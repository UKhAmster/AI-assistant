import os

# Очистка прокси (нужно до импорта httpx/openai)
for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(var, None)

# Аудио
SAMPLE_RATE: int = 16000
CHUNK_SIZE: int = 512
CHUNK_BYTES: int = CHUNK_SIZE * 2
TTS_SAMPLE_RATE: int = 16000  # Qwen3-TTS выдаёт 16kHz

# VAD
VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.0001"))
SILENCE_DURATION: float = float(os.getenv("SILENCE_DURATION", "1.0"))

# LLM
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:8002/v1")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")
MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "20"))

# Whisper
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")

# TTS (F5-TTS, русский fine-tune с voice cloning)
F5_TTS_REPO: str = os.getenv("F5_TTS_REPO", "Misha24-10/F5-TTS_RUSSIAN")
F5_TTS_CKPT_FILE: str = os.getenv(
    "F5_TTS_CKPT_FILE",
    "F5TTS_v1_Base_v2/model_last_inference.safetensors",
)
F5_TTS_VOCAB_FILE: str = os.getenv("F5_TTS_VOCAB_FILE", "F5TTS_v1_Base/vocab.txt")
TTS_VOICE_REF: str = os.getenv("TTS_VOICE_REF", "voice_ref.wav")
TTS_VOICE_REF_TEXT: str = os.getenv(
    "TTS_VOICE_REF_TEXT",
    "Здравствуйте, вы позвонили в приёмную комиссию колледжа КЭСИ.",
)

# Bitrix24
BITRIX_WEBHOOK_URL: str = os.getenv("BITRIX_WEBHOOK_URL", "")

# Fatal-fallback threshold
FATAL_CONSECUTIVE_ERRORS: int = int(os.getenv("FATAL_CONSECUTIVE_ERRORS", "3"))
