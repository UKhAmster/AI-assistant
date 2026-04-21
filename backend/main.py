import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse
from faster_whisper import WhisperModel
import onnxruntime as ort
from openai import AsyncOpenAI
from backend.config import (
    LLM_BASE_URL,
    WHISPER_MODEL_SIZE,
    F5_TTS_REPO,
    F5_TTS_CKPT_FILE,
    F5_TTS_VOCAB_FILE,
    TTS_VOICE_REF,
    TTS_VOICE_REF_TEXT,
    BITRIX_WEBHOOK_URL,
)
from backend.engines import VADEngine, STTEngine, TTSEngine, LLMAgent
from backend.knowledge.loader import build_index
from backend.knowledge.retriever import KnowledgeRetriever
from backend.services.session import DialogSession
from backend.services.bitrix import load_ai_quality_enum_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SSELogHandler(logging.Handler):
    """Рассылает логи всем подключённым SSE-клиентам."""

    def __init__(self) -> None:
        super().__init__()
        self.subscribers: list[asyncio.Queue[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        dead: list[asyncio.Queue[str]] = []
        for q in self.subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.subscribers.remove(q)

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)


sse_handler = SSELogHandler()
sse_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(sse_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("1/6: Загрузка Silero VAD (ONNX) на CPU...")
    vad_path = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")
    app.state.vad_session = ort.InferenceSession(
        vad_path, providers=["CPUExecutionProvider"]
    )

    logger.info("2/6: Загрузка Faster-Whisper на CUDA...")
    app.state.stt_model = WhisperModel(
        WHISPER_MODEL_SIZE, device="cuda", compute_type="float16"
    )

    logger.info("3/6: Загрузка F5-TTS на CUDA...")
    from huggingface_hub import hf_hub_download
    from f5_tts.api import F5TTS

    ckpt_path = hf_hub_download(F5_TTS_REPO, F5_TTS_CKPT_FILE)
    vocab_path = hf_hub_download(F5_TTS_REPO, F5_TTS_VOCAB_FILE)
    logger.info("F5-TTS checkpoint: %s", ckpt_path)

    tts_model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=ckpt_path,
        vocab_file=vocab_path,
        device="cuda",
    )

    ref_path = os.path.join(os.path.dirname(__file__), TTS_VOICE_REF)
    if not os.path.exists(ref_path):
        raise RuntimeError(
            f"F5-TTS требует voice_ref аудио для клонирования. "
            f"Файл не найден: {ref_path}. "
            f"Сконвертируй референс в 16kHz mono WAV и положи по этому пути."
        )
    logger.info("Голосовой референс: %s (text=%r)", ref_path, TTS_VOICE_REF_TEXT)

    app.state.tts_model = tts_model
    app.state.tts_ref_audio = ref_path
    app.state.tts_ref_text = TTS_VOICE_REF_TEXT

    logger.info("4/6: Инициализация клиента LLM (%s)...", LLM_BASE_URL)
    app.state.llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="local-key")

    logger.info("5/6: Загрузка базы знаний колледжа...")
    knowledge_index = build_index()
    app.state.retriever = KnowledgeRetriever(knowledge_index)
    if app.state.retriever.is_available:
        logger.info("База знаний загружена, RAG активен")
    else:
        logger.info("База знаний не найдена, RAG отключен (положите .md файлы в backend/data/)")

    # 6/6: Загрузка enum IDs для UF_CRM_AI_QUALITY
    app.state.bitrix_enum_ids = None
    if BITRIX_WEBHOOK_URL:
        logger.info("6/6: Резолв Bitrix enum IDs...")
        app.state.bitrix_enum_ids = await load_ai_quality_enum_ids(BITRIX_WEBHOOK_URL)
        logger.info("Bitrix enum IDs: %s", app.state.bitrix_enum_ids)
    else:
        logger.warning("BITRIX_WEBHOOK_URL не задан, лиды в Bitrix не будут создаваться")

    logger.info("Все модели загружены. Сервер готов.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_available": app.state.retriever.is_available,
    }


@app.get("/logs")
async def logs_stream():
    q = sse_handler.subscribe()

    async def event_generator():
        try:
            while True:
                msg = await q.get()
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            sse_handler.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    caller_phone_raw = websocket.query_params.get("caller_phone")
    caller_phone: str | None = None
    if caller_phone_raw:
        from backend.services.phone_normalizer import normalize_phone
        caller_phone = normalize_phone(caller_phone_raw)
        if not caller_phone:
            logger.warning(
                "Невалидный caller_phone в query-param: %r (игнорирую)",
                caller_phone_raw,
            )

    logger.info(
        "Входящий звонок: клиент подключен, caller_phone=%s",
        caller_phone or "<неизвестен>",
    )

    session = DialogSession(
        websocket=websocket,
        vad=VADEngine(app.state.vad_session),
        stt=STTEngine(app.state.stt_model),
        tts=TTSEngine(
            app.state.tts_model,
            ref_audio_path=app.state.tts_ref_audio,
            ref_text=app.state.tts_ref_text,
        ),
        llm=LLMAgent(app.state.llm_client, retriever=app.state.retriever),
        caller_phone=caller_phone,
        enum_ids=app.state.bitrix_enum_ids,
    )

    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info("Звонок завершен: клиент отключился")
    except Exception as e:
        logger.error("Ошибка WebSocket: %s", e, exc_info=True)
    finally:
        session.save_log()
