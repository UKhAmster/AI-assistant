import os

# Очистка прокси (нужно до импорта httpx/openai)
for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(var, None)

# Аудио
SAMPLE_RATE: int = 16000
CHUNK_SIZE: int = 512
CHUNK_BYTES: int = CHUNK_SIZE * 2
TTS_SAMPLE_RATE: int = 24000

# VAD
VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.0001"))
SILENCE_DURATION: float = float(os.getenv("SILENCE_DURATION", "1.0"))

# LLM
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")
MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "20"))

# Whisper
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")

# Bitrix24
BITRIX_WEBHOOK_URL: str = os.getenv("BITRIX_WEBHOOK_URL", "")
