import os
import asyncio
import logging
import time
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
import onnxruntime as ort

# Принудительно очищаем переменные прокси для этого скрипта
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
SAMPLE_RATE = 16000
CHUNK_SIZE = 512             # Размер чанка в сэмплах
CHUNK_BYTES = CHUNK_SIZE * 2 # Размер чанка в байтах (1024 байта)
SILENCE_DURATION = 1.2       # Секунды тишины для отсечки фразы
VAD_THRESHOLD = 0.0001          # Порог уверенности VAD

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация ML-моделей при старте сервера"""
    logger.info("Загрузка Silero VAD (ONNX)...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(current_dir, "silero_vad.onnx")
    
    app.state.vad_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    logger.info("Загрузка Faster-Whisper...")
    app.state.stt_model = WhisperModel("small", device="cpu", compute_type="float32")
    
    logger.info("Модели успешно загружены. Сервер готов.")
    yield
    
    logger.info("Очистка ресурсов...")
    app.state.vad_session = None
    app.state.stt_model = None

app = FastAPI(lifespan=lifespan)

class VADEngine:
    """Обертка для Silero VAD v5 через ONNX Runtime с режимом Рентгена"""
    def __init__(self, session):
        self.session = session
        self.reset_states()

    def reset_states(self):
        """Сброс состояний RNN для версии v5"""
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
        
        confidence = out[0][0]
        max_amplitude = float(np.max(np.abs(audio_float)))
        
        # ДЕБАГ: Если в аудио есть хоть какой-то звук (громкость > 1%), выводим в лог
        if max_amplitude > 0.01:
            logger.info(f"[VAD РЕНТГЕН] Громкость: {max_amplitude:.4f} | Уверенность сети: {confidence:.4f}")
        
        return confidence > VAD_THRESHOLD

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Клиент подключен по WebSocket")

    vad_engine = VADEngine(app.state.vad_session)
    stt_engine = STTEngine(app.state.stt_model)

    incoming_buffer = bytearray()
    speech_buffer = bytearray()
    
    is_speaking = False
    silence_chunk_count = 0

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
                            logger.info("Конец фразы. Отправка на распознавание...")
                            
                            audio_to_process = bytes(speech_buffer)
                            speech_buffer.clear()
                            is_speaking = False
                            silence_start_time = None
                            
                            # Запускаем распознавание в отдельной задаче, 
                            # чтобы не блокировать получение аудио из WebSocket
                            async def background_transcribe(audio_bytes):
                                text = await stt_engine.transcribe(audio_bytes)
                                if text:
                                    try:
                                        response = {"status": "success", "text": text}
                                        await websocket.send_text(json.dumps(response, ensure_ascii=False))
                                        logger.info(f"Распознано: {text}")
                                    except Exception as e:
                                        logger.error(f"Ошибка отправки клиенту: {e}")
                            
                            asyncio.create_task(background_transcribe(audio_to_process))

    except WebSocketDisconnect:
        logger.info("Клиент отключился")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}", exc_info=True)
    finally:
        if is_speaking and len(speech_buffer) > 0:
            logger.info("Обработка остатков аудио после отключения...")
            audio_to_process = bytes(speech_buffer)
            async def final_transcribe(audio_bytes):
                text = await stt_engine.transcribe(audio_bytes)
                if text:
                    logger.info(f"Распознано напоследок: {text}")
            asyncio.create_task(final_transcribe(audio_to_process))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)