import asyncio
import logging

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTEngine:
    """Faster-Whisper (CUDA, float16) — транскрипция речи в текст."""

    def __init__(self, model: WhisperModel) -> None:
        self.model = model

    async def transcribe(self, audio_bytes: bytes) -> str:
        return await asyncio.to_thread(self._transcribe, audio_bytes)

    def _transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""
        audio_float = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        segments, _ = self.model.transcribe(audio_float, beam_size=5, language="ru")
        return " ".join(s.text for s in segments).strip()
