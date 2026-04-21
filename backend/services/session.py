import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import WebSocket

from backend.config import CHUNK_BYTES, SAMPLE_RATE, SILENCE_DURATION, FATAL_CONSECUTIVE_ERRORS
from backend.engines.vad import VADEngine
from backend.engines.stt import STTEngine
from backend.engines.tts import TTSEngine
from backend.engines.llm import LLMAgent
from backend.services.bitrix import send_to_bitrix
from backend.services.phone_normalizer import normalize_phone

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
        caller_phone: str | None = None,
        enum_ids: dict[str, int] | None = None,
    ) -> None:
        self.websocket = websocket
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.caller_phone = caller_phone
        self.enum_ids = enum_ids

        self.session_id: str = uuid.uuid4().hex[:12]
        self.chat_history: list[dict[str, str]] = []
        self.is_processing: bool = False
        self._start_time: float = time.time()
        self._consecutive_errors: int = 0

        self._incoming_buffer = bytearray()
        self._speech_buffer = bytearray()
        self._is_speaking = False
        self._silence_start: float | None = None

    async def run(self) -> None:
        """Основной цикл диалога: приветствие -> прием аудио -> обработка."""
        await self._send_greeting()
        chunks_received = 0

        while True:
            msg = await self.websocket.receive()

            # Текстовое сообщение — управляющий сигнал
            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "end_of_speech":
                        await self._flush_speech()
                except (json.JSONDecodeError, KeyError):
                    pass
                continue

            data = msg.get("bytes", b"")
            if not data:
                continue

            self._incoming_buffer.extend(data)
            chunks_received += 1
            if chunks_received == 1:
                logger.info("Первые данные от клиента: %d байт", len(data))

            while len(self._incoming_buffer) >= CHUNK_BYTES:
                chunk = bytes(self._incoming_buffer[:CHUNK_BYTES])
                del self._incoming_buffer[:CHUNK_BYTES]
                await self._handle_chunk(chunk)

    async def _flush_speech(self) -> None:
        """Принудительно завершает фразу (push-to-talk)."""
        if self._is_speaking and len(self._speech_buffer) > 0:
            audio_to_process = bytes(self._speech_buffer)
            duration_ms = len(audio_to_process) / 2 / SAMPLE_RATE * 1000
            logger.info("Push-to-talk: конец фразы, %.0f мс аудио", duration_ms)
            self._speech_buffer.clear()
            self._is_speaking = False
            self._silence_start = None
            await self._maybe_process(audio_to_process)
        elif len(self._speech_buffer) == 0:
            logger.info("Push-to-talk: пустой буфер, речь не обнаружена")
            await self._send_text("system", "")

    async def _send_text(self, role: str, text: str) -> None:
        """Отправляет текстовое JSON-сообщение в WebSocket для отображения в UI."""
        await self.websocket.send_text(json.dumps(
            {"type": "text", "role": role, "content": text},
            ensure_ascii=False,
        ))

    async def _send_greeting(self) -> None:
        self.chat_history.append({"role": "assistant", "content": GREETING})
        logger.info("Агент: %s", GREETING)
        await self._send_text("assistant", GREETING)
        audio = await self.tts.synthesize(GREETING)
        await self.websocket.send_bytes(audio)

    async def _handle_chunk(self, chunk: bytes) -> None:
        has_speech = await self.vad.is_speech(chunk)

        if not self._is_speaking:
            if has_speech:
                logger.info("VAD: речь обнаружена, начинаю запись")
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
                    duration_ms = len(audio_to_process) / 2 / SAMPLE_RATE * 1000
                    logger.info("VAD: конец фразы, %.0f мс аудио", duration_ms)
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
                logger.info("STT: пустая транскрипция, пропуск")
                await self._send_text("system", "")
                return

            logger.info("Пользователь: %s", user_text)
            self.chat_history.append({"role": "user", "content": user_text})
            await self._send_text("user", user_text)

            # 2. LLM
            reply_text, ticket_data = await self.llm.get_response(
                self.chat_history, caller_phone=self.caller_phone,
            )
            self._consecutive_errors = 0
            t_llm = time.time()

            if reply_text:
                logger.info("Агент: %s", reply_text)
                self.chat_history.append({"role": "assistant", "content": reply_text})
                await self._send_text("assistant", reply_text)

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

            # 4. Bitrix24 (с учётом caller_phone из АТС и fatal_fallback ветки)
            if ticket_data:
                phone_for_check = ticket_data.get("phone") or self.caller_phone or ""
                phone = normalize_phone(phone_for_check) if phone_for_check else ""
                if phone or ticket_data.get("request_type") == "fatal_fallback":
                    bitrix_task = asyncio.create_task(
                        send_to_bitrix(
                            ticket_data,
                            self.chat_history,
                            self.enum_ids,
                            self.caller_phone,
                        )
                    )
                    bitrix_task.add_done_callback(self._on_task_done)
                else:
                    # Телефон невалиден — переспрашиваем
                    retry_text = (
                        "Извините, я не разобрала номер телефона. "
                        "Повторите, пожалуйста, по цифрам."
                    )
                    logger.warning("Невалидный телефон: %s", ticket_data.get("phone"))
                    self.chat_history.append({"role": "assistant", "content": retry_text})
                    await self._send_text("assistant", retry_text)
                    audio = await self.tts.synthesize(retry_text)
                    await self.websocket.send_bytes(audio)

        finally:
            self.is_processing = False

    _FATAL_APOLOGY = (
        "Извините, у меня технические сложности. "
        "Сейчас зафиксирую вашу заявку — оператор обязательно вам перезвонит."
    )
    _FATAL_PHONE_PROMPT = (
        "Продиктуйте, пожалуйста, номер телефона для связи."
    )
    _FATAL_GOODBYE = "До свидания."

    async def _fatal_fallback(self, reason: str) -> None:
        """Последний резерв: при любой неисправности бота создаём срочный лид
        и вежливо закрываем звонок. В будущей итерации с Asterisk первым шагом
        будет реальная попытка SIP-transfer."""
        logger.error("FATAL_FALLBACK triggered: %s", reason)

        try:
            await self._speak_safe(self._FATAL_APOLOGY)

            phone = self.caller_phone or ""
            if not phone:
                await self._speak_safe(self._FATAL_PHONE_PROMPT)
                phone = await self._collect_phone_best_effort()

            name = self._extract_name_from_history()

            ticket_data = {
                "name": name,
                "phone": phone,
                "intent": f"СРОЧНО — технический сбой бота: {reason}",
                "admission_year": None,
                "request_type": "fatal_fallback",
            }

            await send_to_bitrix(
                ticket_data,
                self.chat_history,
                self.enum_ids,
                self.caller_phone,
            )

            await self._speak_safe(self._FATAL_GOODBYE)
        except Exception as exc:
            logger.error("Ошибка в _fatal_fallback: %s", exc, exc_info=True)
        finally:
            try:
                await self.websocket.close()
            except Exception:
                pass

    async def _speak_safe(self, text: str) -> None:
        """TTS + WebSocket send с подавлением ошибок (мы уже в fatal-режиме)."""
        try:
            self.chat_history.append({"role": "assistant", "content": text})
            await self._send_text("assistant", text)
            audio = await self.tts.synthesize(text)
            await self.websocket.send_bytes(audio)
        except Exception as exc:
            logger.warning("Не удалось озвучить в fatal-режиме: %s", exc)

    async def _collect_phone_best_effort(self, timeout_sec: float = 10.0) -> str:
        """Одна попытка получить телефон через одну короткую речевую реплику.
        Если не получилось — возвращает пустую строку."""
        try:
            self._speech_buffer.clear()
            self._is_speaking = False
            self._silence_start = None
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(
                        self.websocket.receive(), timeout=timeout_sec,
                    )
                except asyncio.TimeoutError:
                    break
                data = msg.get("bytes", b"")
                if not data:
                    continue
                self._incoming_buffer.extend(data)
                while len(self._incoming_buffer) >= CHUNK_BYTES:
                    chunk = bytes(self._incoming_buffer[:CHUNK_BYTES])
                    del self._incoming_buffer[:CHUNK_BYTES]
                    has_speech = await self.vad.is_speech(chunk)
                    if has_speech:
                        self._speech_buffer.extend(chunk)
                        self._is_speaking = True
                        self._silence_start = None
                    elif self._is_speaking:
                        if self._silence_start is None:
                            self._silence_start = time.time()
                        elif time.time() - self._silence_start >= SILENCE_DURATION:
                            audio = bytes(self._speech_buffer)
                            text = await self.stt.transcribe(audio)
                            phone = normalize_phone(text or "")
                            return phone or ""
            return ""
        except Exception as exc:
            logger.warning("Не удалось собрать телефон best-effort: %s", exc)
            return ""

    def _extract_name_from_history(self) -> str:
        """Best-effort: ищет в user-репликах одиночное слово похожее на имя.
        Если не нашли — возвращает пустую строку."""
        for msg in self.chat_history:
            if msg["role"] != "user":
                continue
            words = msg["content"].strip().split()
            if len(words) == 1 and words[0][0].isupper():
                return words[0]
        return ""

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
