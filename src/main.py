import os
import asyncio
import logging
import time
import json
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
import onnxruntime as ort
from openai import AsyncOpenAI
import torch

# Очистка прокси
for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(var, None)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHUNK_BYTES = CHUNK_SIZE * 2
SILENCE_DURATION = 1.0
VAD_THRESHOLD = 0.0001
TTS_SAMPLE_RATE = 24000 # Silero v4 лучше звучит на 24kHz

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("1/4: Загрузка Silero VAD (ONNX) на CPU...")
    app.state.vad_session = ort.InferenceSession(os.path.join(os.path.dirname(__file__), "silero_vad.onnx"), providers=['CPUExecutionProvider'])

    logger.info("2/4: Загрузка Faster-Whisper на CUDA (RTX 5090)...")
    app.state.stt_model = WhisperModel("small", device="cuda", compute_type="float16")
    
    logger.info("3/4: Загрузка Silero TTS (Text-to-Speech) на CUDA...")
    # Скачиваем и кешируем модель Silero TTS v4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language='ru',
                                  speaker='v4_ru')
    tts_model.to(device)
    app.state.tts_model = tts_model
    app.state.tts_device = device

    logger.info(f"4/4: Инициализация клиента LLM ({LLM_BASE_URL})...")
    app.state.llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="local-key")
    
    logger.info("✅ Все модели успешно загружены. Сервер готов к диалогам.")
    yield

app = FastAPI(lifespan=lifespan)

# --- ДВИЖКИ STT и VAD (Остались без изменений) ---
class VADEngine:
    def __init__(self, session):
        self.session = session
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    async def is_speech(self, audio_chunk: bytes) -> bool:
        return await asyncio.to_thread(self._detect, audio_chunk)

    def _detect(self, audio_chunk: bytes) -> bool:
        if len(audio_chunk) != CHUNK_BYTES: return False
        audio_float = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        ort_inputs = {'input': np.expand_dims(audio_float, axis=0), 'sr': np.array([SAMPLE_RATE], dtype=np.int64), 'state': self.state}
        out, self.state = self.session.run(None, ort_inputs)
        return out[0][0] > VAD_THRESHOLD

class STTEngine:
    def __init__(self, model):
        self.model = model

    async def transcribe(self, audio_bytes: bytes) -> str:
        return await asyncio.to_thread(self._transcribe, audio_bytes)

    def _transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes: return ""
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_float, beam_size=5, language="ru")
        return " ".join([s.text for s in segments]).strip()

# --- НОВЫЙ ДВИЖОК TTS (Синтез речи) ---
class TTSEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        # Silero TTS чувствителен к длинным текстам и спецсимволам, делаем базовую очистку
        clean_text = re.sub(r'[^а-яА-ЯёЁ0-9\s.,!?\-]', '', text)
        if not clean_text.strip():
            return b""
            
        audio_tensor = self.model.apply_tts(text=clean_text,
                                            speaker='xenia', # Приятный женский голос
                                            sample_rate=TTS_SAMPLE_RATE)
        
        # Переводим тензор обратно в байты PCM 16-bit
        audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
        return audio_np.tobytes()

# --- ОБНОВЛЕННЫЙ ДВИЖОК LLM (Диалог и вызов функций) ---
class LLMAgent:
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.system_prompt = (
            "Ты — приветливый голосовой ассистент приемной комиссии колледжа (девушка по имени Ксения). "
            "Твоя задача — консультировать абитуриентов. Общайся вежливо, коротко и естественно (максимум 2-3 предложения), "
            "так как твой ответ будет озвучиваться голосом.\n"
            "Если пользователь просит перезвонить, записать его, или задает сложный вопрос, "
            "мягко спроси его ИМЯ и ТЕЛЕФОН. Как только ты получишь эти данные, вызови инструмент (tool) 'create_ticket'."
        )
        
        # Описание функции для Битрикс24
        self.tools = [{
            "type": "function",
            "function": {
                "name": "create_ticket",
                "description": "Создать заявку/лид на обратный звонок или поступление",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Имя абонента"},
                        "phone": {"type": "string", "description": "Контактный телефон"},
                        "intent": {"type": "string", "description": "Краткая суть вопроса (1 предложение)"}
                    },
                    "required": ["name", "phone", "intent"]
                }
            }
        }]

    async def get_response(self, messages: list) -> tuple[str, dict]:
        """Возвращает (Текст для озвучки, Данные тикета если нужно)"""
        try:
            # Отправляем весь контекст диалога
            response = await self.client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "system", "content": self.system_prompt}] + messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.4,
                max_tokens=150
            )
            
            message = response.choices[0].message
            reply_text = message.content or ""
            ticket_data = None
            
            # Проверяем, решила ли модель вызвать функцию (создать лид)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "create_ticket":
                        ticket_data = json.loads(tool_call.function.arguments)
                        logger.info(f"⚡ LLM ИНИЦИИРОВАЛА СОЗДАНИЕ ТИКЕТА: {ticket_data}")
                        # Если LLM ничего не сказала текстом, добавим подтверждение сами
                        if not reply_text:
                            reply_text = "Я записала ваши данные. Наши специалисты свяжутся с вами в ближайшее время! Чем-то еще могу помочь?"
            
            return reply_text, ticket_data
            
        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")
            return "Извините, у меня небольшая заминка со связью.", None

# --- Имитация отправки в Битрикс24 ---
async def send_to_bitrix(ticket_data: dict):
    logger.info(f"🌍 [WEBHOOK] Отправка лида в Битрикс24: {ticket_data}")
    # Здесь будет реальный httpx.post(BITRIX_WEBHOOK_URL, json=...)
    await asyncio.sleep(1) 
    logger.info("✅ Лид успешно создан в CRM!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("📞 Входящий звонок: Клиент подключен")

    vad_engine = VADEngine(app.state.vad_session)
    stt_engine = STTEngine(app.state.stt_model)
    tts_engine = TTSEngine(app.state.tts_model, app.state.tts_device)
    llm_agent = LLMAgent(app.state.llm_client)

    # Память текущего диалога (привязывается к конкретному WebSocket-соединению)
    chat_history = []
    
    incoming_buffer = bytearray()
    speech_buffer = bytearray()
    is_speaking = False
    silence_start_time = None

    try:
        # Приветствуем пользователя при подключении
        greeting = "Здравствуйте! Вы позвонили в приемную комиссию. Меня зовут Ксения, чем я могу вам помочь?"
        chat_history.append({"role": "assistant", "content": greeting})
        logger.info(f"🤖 Агент: {greeting}")
        
        # Озвучиваем приветствие и шлем клиенту
        audio_reply = await tts_engine.synthesize(greeting)
        await websocket.send_bytes(audio_reply)

        while True:
            # Получаем аудио от клиента
            data = await websocket.receive_bytes()
            incoming_buffer.extend(data)

            while len(incoming_buffer) >= CHUNK_BYTES:
                chunk = bytes(incoming_buffer[:CHUNK_BYTES])
                del incoming_buffer[:CHUNK_BYTES]

                has_speech = await vad_engine.is_speech(chunk)

                if not is_speaking:
                    if has_speech:
                        is_speaking = True
                        silence_start_time = None
                        speech_buffer.clear()
                        speech_buffer.extend(chunk)
                else:
                    speech_buffer.extend(chunk)
                    
                    if has_speech:
                        silence_start_time = None 
                    else:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif (time.time() - silence_start_time) >= SILENCE_DURATION:
                            audio_to_process = bytes(speech_buffer)
                            speech_buffer.clear()
                            is_speaking = False
                            silence_start_time = None
                            
                            # Начинаем обработку фразы
                            async def process_turn(audio_bytes):
                                # 1. Услышали (STT)
                                user_text = await stt_engine.transcribe(audio_bytes)
                                if not user_text:
                                    return
                                    
                                logger.info(f"👤 Пользователь: {user_text}")
                                chat_history.append({"role": "user", "content": user_text})
                                
                                # 2. Подумали (LLM)
                                reply_text, ticket_data = await llm_agent.get_response(chat_history)
                                
                                if reply_text:
                                    logger.info(f"🤖 Агент: {reply_text}")
                                    chat_history.append({"role": "assistant", "content": reply_text})
                                    
                                    # 3. Ответили голосом (TTS)
                                    audio_reply = await tts_engine.synthesize(reply_text)
                                    await websocket.send_bytes(audio_reply)
                                
                                # 4. Выполнили фоновую задачу (Битрикс24)
                                if ticket_data:
                                    asyncio.create_task(send_to_bitrix(ticket_data))
                            
                            asyncio.create_task(process_turn(audio_to_process))

    except WebSocketDisconnect:
        logger.info("☎️ Звонок завершен: Клиент отключился")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}", exc_info=True)