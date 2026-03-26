import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import WebSocket

from src.config import CHUNK_BYTES, SILENCE_DURATION
from src.engines.vad import VADEngine
from src.engines.stt import STTEngine
from src.engines.tts import TTSEngine
from src.engines.llm import LLMAgent
from src.services.bitrix import send_to_bitrix
from src.services.phone_normalizer import normalize_phone

logger = logging.getLogger(__name__)

GREETING = (
    "Здравствуйте! Вы позвонили в приемную комиссию. "
    "Меня зовут Ксения, чем я могу вам помочь?"
)


class DialogSession:
    """Управляет одним WebSocket-диалогом.

    FIX: is_processing предотвращает конкурентные process_turn.
    FIX: обработка исключений в фоновых задачах.
    """

    def __init__(
        self,
        websocket: WebSocket,
        vad: VADEngine,
        stt: STTEngine,
        tts: TTSEngine,
        llm: LLMAgent,
    ) -> None:
        self.websocket = websocket
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self.llm = llm

        self.session_id: str = uuid.uuid4().hex[:12]
        self.chat_history: list[dict[str, str]] = []
        self.is_processing: bool = False
        self._start_time: float = time.time()

        self._incoming_buffer = bytearray()
        self._speech_buffer = bytearray()
        self._is_speaking = False
        self._silence_start: float | None = None

    async def run(self) -> None:
        """Основной цикл диалога: приветствие -> прием аудио -> обработка."""
        await self._send_greeting()

        while True:
            data = await self.websocket.receive_bytes()
            self._incoming_buffer.extend(data)

            while len(self._incoming_buffer) >= CHUNK_BYTES:
                chunk = bytes(self._incoming_buffer[:CHUNK_BYTES])
                del self._incoming_buffer[:CHUNK_BYTES]
                await self._handle_chunk(chunk)

    async def _send_greeting(self) -> None:
        self.chat_history.append({"role": "assistant", "content": GREETING})
        logger.info("Агент: %s", GREETING)
        audio = await self.tts.synthesize(GREETING)
        await self.websocket.send_bytes(audio)

    async def _handle_chunk(self, chunk: bytes) -> None:
        has_speech = await self.vad.is_speech(chunk)

        if not self._is_speaking:
            if has_speech:
                self._is_speaking = True
                self._silence_start = None
                self._speech_buffer.clear()
                self._speech_buffer.extend(chunk)
        else:
            self._speech_buffer.extend(chunk)

            if has_speech:
                self._silence_start = None
            else:
                if self._silence_start is None:
                    self._silence_start = time.time()
                elif (time.time() - self._silence_start) >= SILENCE_DURATION:
                    audio_to_process = bytes(self._speech_buffer)
                    self._speech_buffer.clear()
                    self._is_speaking = False
                    self._silence_start = None
                    await self._maybe_process(audio_to_process)

    async def _maybe_process(self, audio_bytes: bytes) -> None:
        """Запускает обработку фразы, если не занято.

        FIX: is_processing не дает запустить параллельный process_turn.
        Аудио, пришедшее во время обработки, будет потеряно — в будущем
        можно накапливать в очередь.
        """
        if self.is_processing:
            logger.warning("Пропуск фразы: обработка предыдущей еще не завершена")
            return

        task = asyncio.create_task(self._process_turn(audio_bytes))
        task.add_done_callback(self._on_task_done)

    async def _process_turn(self, audio_bytes: bytes) -> None:
        self.is_processing = True
        try:
            t0 = time.time()

            # 1. STT
            user_text = await self.stt.transcribe(audio_bytes)
            t_stt = time.time()

            if not user_text:
                return

            logger.info("Пользователь: %s", user_text)
            self.chat_history.append({"role": "user", "content": user_text})

            # 2. LLM
            reply_text, ticket_data = await self.llm.get_response(self.chat_history)
            t_llm = time.time()

            if reply_text:
                logger.info("Агент: %s", reply_text)
                self.chat_history.append({"role": "assistant", "content": reply_text})

                # 3. TTS
                audio_reply = await self.tts.synthesize(reply_text)
                t_tts = time.time()
                await self.websocket.send_bytes(audio_reply)

                # Метрики латентности
                logger.info(
                    "LATENCY stt=%dms llm=%dms tts=%dms total=%dms",
                    int((t_stt - t0) * 1000),
                    int((t_llm - t_stt) * 1000),
                    int((t_tts - t_llm) * 1000),
                    int((t_tts - t0) * 1000),
                )

            # 4. Bitrix24 (с проверкой телефона)
            if ticket_data:
                phone = normalize_phone(ticket_data.get("phone", ""))
                if phone:
                    bitrix_task = asyncio.create_task(send_to_bitrix(ticket_data))
                    bitrix_task.add_done_callback(self._on_task_done)
                else:
                    # Телефон невалиден — переспрашиваем
                    retry_text = (
                        "Извините, я не разобрала номер телефона. "
                        "Повторите, пожалуйста, по цифрам."
                    )
                    logger.warning("Невалидный телефон: %s", ticket_data.get("phone"))
                    self.chat_history.append({"role": "assistant", "content": retry_text})
                    audio = await self.tts.synthesize(retry_text)
                    await self.websocket.send_bytes(audio)

        finally:
            self.is_processing = False

    def save_log(self) -> None:
        """Сохраняет историю диалога в JSON-файл."""
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "sessions")
        os.makedirs(log_dir, exist_ok=True)

        log_data = {
            "session_id": self.session_id,
            "start_time": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self._start_time)
            ),
            "duration_sec": round(time.time() - self._start_time, 1),
            "turns": len([m for m in self.chat_history if m["role"] == "user"]),
            "chat_history": self.chat_history,
        }

        filename = f"{self.session_id}_{int(self._start_time)}.json"
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.info("Лог диалога сохранен: %s", filepath)
        except Exception as e:
            logger.error("Не удалось сохранить лог диалога: %s", e)

    @staticmethod
    def _on_task_done(task: asyncio.Task[Any]) -> None:
        """Логирует необработанные исключения из фоновых задач."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Ошибка в фоновой задаче: %s", exc, exc_info=exc)
