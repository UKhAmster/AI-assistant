import os
import asyncio
import logging
import time
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
import onnxruntime as ort
from openai import AsyncOpenAI

# Принудительно очищаем переменные прокси для этого скрипта
for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(var, None)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы для аудио
SAMPLE_RATE = 16000
CHUNK_SIZE = 512             # Размер чанка в сэмплах
CHUNK_BYTES = CHUNK_SIZE * 2 # Размер чанка в байтах (1024 байта)
SILENCE_DURATION = 1.2       # Секунды тишины для отсечки фразы
VAD_THRESHOLD = 0.0001       # Порог уверенности VAD

# Настройки LLM через переменные окружения (с фолбеками)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация ML-моделей при старте сервера"""
    logger.info("Загрузка Silero VAD (ONNX) на CPU...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(current_dir, "silero_vad.onnx")
    
    app.state.vad_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    logger.info("Загрузка Faster-Whisper на CUDA (RTX 5090)...")
    # ПЕРЕВЕДЕНО НА GPU для максимальной скорости
    app.state.stt_model = WhisperModel("small", device="cuda", compute_type="float16")
    
    logger.info(f"Инициализация клиента LLM ({LLM_BASE_URL})...")
    app.state.llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="local-key-not-checked")
    
    logger.info("Модели успешно загружены. Сервер готов.")
    yield
    
    logger.info("Очистка ресурсов...")
    app.state.vad_session = None
    app.state.stt_model = None
    app.state.llm_client = None

app = FastAPI(lifespan=lifespan)

class VADEngine:
    """Обертка для Silero VAD v5 через ONNX Runtime"""
    def __init__(self, session):
        self.session = session
        self.reset_states()

    def reset_states(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    async def is_speech(self, audio_chunk: bytes) -> bool:
        return await asyncio.to_thread(self._detect_speech_sync, audio_chunk)

    def _detect_speech_sync(self, audio_chunk: bytes) -> bool:
        if len(audio_chunk) != CHUNK_BYTES:
            return False
            
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        ort_inputs = {
            'input': np.expand_dims(audio_float, axis=0),
            'sr': np.array([SAMPLE_RATE], dtype=np.int64),
            'state': self.state
        }
        
        out, state = self.session.run(None, ort_inputs)
        self.state = state
        
        return out[0][0] > VAD_THRESHOLD

class STTEngine:
    """Обертка для Faster-Whisper"""
    def __init__(self, model):
        self.model = model

    async def transcribe(self, audio_bytes: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes)

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""
            
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            audio_float,
            beam_size=5,
            language="ru",
            condition_on_previous_text=False
        )
        
        return " ".join([segment.text for segment in segments]).strip()

class LLMEngine:
    """Обертка для работы с локальной LLM (Qwen) через vLLM"""
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.system_prompt = (
            "Ты — ИИ-ассистент приемной комиссии учебного заведения. "
            "Твоя задача — проанализировать текст пользователя и вернуть результат СТРОГО в формате JSON. "
            "Извлеки следующие ключи:\n"
            "- name (Имя человека, если есть, иначе null)\n"
            "- phone (Телефон, если есть, иначе null)\n"
            "- intent (Цель: 'Поступление', 'Расписание', 'Справка', или 'Иное')\n"
            "- department (Направление или факультет, если упоминается, иначе null)\n"
            "- summary (Краткое содержание вопроса в 1 предложении)\n\n"
            "Не пиши никаких пояснений, только валидный JSON."
        )

    async def extract_entities(self, text: str) -> dict:
        try:
            response = await self.client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1, # Низкая температура для стабильного JSON
                max_tokens=256
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Очистка от возможных маркдаун-тегов, если LLM решит их добавить
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3]
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3]
                
            return json.loads(raw_content)
        except json.JSONDecodeError:
            logger.error(f"Ошибка парсинга JSON. Сырой ответ LLM: {raw_content}")
            return {"error": "Invalid JSON from LLM", "raw_text": text}
        except Exception as e:
            logger.error(f"Ошибка LLM-процессинга: {e}")
            return {"error": "Не удалось извлечь сущности", "raw_text": text}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Клиент подключен по WebSocket")

    vad_engine = VADEngine(app.state.vad_session)
    stt_engine = STTEngine(app.state.stt_model)
    llm_engine = LLMEngine(app.state.llm_client)

    incoming_buffer = bytearray()
    speech_buffer = bytearray()
    
    is_speaking = False
    silence_start_time = None

    try:
        while True:
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
                        logger.info("Начало речи обнаружено.")
                else:
                    speech_buffer.extend(chunk)
                    
                    if has_speech:
                        silence_start_time = None 
                    else:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif (time.time() - silence_start_time) >= SILENCE_DURATION:
                            logger.info("Конец фразы. Обработка STT -> LLM...")
                            
                            audio_to_process = bytes(speech_buffer)
                            speech_buffer.clear()
                            is_speaking = False
                            silence_start_time = None
                            
                            # Асинхронный пайплайн
                            async def process_pipeline(audio_bytes):
                                # 1. Speech-to-Text
                                text = await stt_engine.transcribe(audio_bytes)
                                if text:
                                    logger.info(f"Распознано: {text}")
                                    
                                    # 2. LLM Entity Extraction
                                    structured_data = await llm_engine.extract_entities(text)
                                    logger.info(f"LLM Результат: {structured_data}")
                                    
                                    # 3. Отправка клиенту
                                    response = {
                                        "status": "success", 
                                        "transcription": text,
                                        "entities": structured_data
                                    }
                                    try:
                                        await websocket.send_text(json.dumps(response, ensure_ascii=False))
                                    except Exception as e:
                                        logger.error(f"Ошибка отправки клиенту: {e}")
                            
                            asyncio.create_task(process_pipeline(audio_to_process))

    except WebSocketDisconnect:
        logger.info("Клиент отключился")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка WebSocket: {e}", exc_info=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)