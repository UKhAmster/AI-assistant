import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import onnxruntime as ort
from openai import AsyncOpenAI
import torch

from src.config import LLM_BASE_URL, WHISPER_MODEL_SIZE
from src.engines import VADEngine, STTEngine, TTSEngine, LLMAgent
from src.knowledge.loader import build_index
from src.knowledge.retriever import KnowledgeRetriever
from src.services.session import DialogSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("1/4: Загрузка Silero VAD (ONNX) на CPU...")
    vad_path = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")
    app.state.vad_session = ort.InferenceSession(
        vad_path, providers=["CPUExecutionProvider"]
    )

    logger.info("2/4: Загрузка Faster-Whisper на CUDA...")
    app.state.stt_model = WhisperModel(
        WHISPER_MODEL_SIZE, device="cuda", compute_type="float16"
    )

    logger.info("3/4: Загрузка Silero TTS на CUDA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
    )
    tts_model.to(device)
    app.state.tts_model = tts_model
    app.state.tts_device = device

    logger.info("4/5: Инициализация клиента LLM (%s)...", LLM_BASE_URL)
    app.state.llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="local-key")

    logger.info("5/5: Загрузка базы знаний колледжа...")
    knowledge_index = build_index()
    app.state.retriever = KnowledgeRetriever(knowledge_index)
    if app.state.retriever.is_available:
        logger.info("База знаний загружена, RAG активен")
    else:
        logger.info("База знаний не найдена, RAG отключен (положите .md файлы в src/data/)")

    logger.info("Все модели загружены. Сервер готов.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_available": app.state.retriever.is_available,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Входящий звонок: клиент подключен")

    session = DialogSession(
        websocket=websocket,
        vad=VADEngine(app.state.vad_session),
        stt=STTEngine(app.state.stt_model),
        tts=TTSEngine(app.state.tts_model, app.state.tts_device),
        llm=LLMAgent(app.state.llm_client, retriever=app.state.retriever),
    )

    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info("Звонок завершен: клиент отключился")
    except Exception as e:
        logger.error("Ошибка WebSocket: %s", e, exc_info=True)
    finally:
        session.save_log()
