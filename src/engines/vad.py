import asyncio
import logging

import numpy as np
import onnxruntime as ort

from src.config import CHUNK_BYTES, SAMPLE_RATE, VAD_THRESHOLD

logger = logging.getLogger(__name__)


class VADEngine:
    """Silero VAD (ONNX, CPU) — детекция речи/тишины.

    FIX: asyncio.Lock защищает self.state от конкурентного доступа.
    ONNX RNN state — мутабельный numpy-массив, без лока параллельные
    вызовы через asyncio.to_thread могут его повредить.
    """

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self._lock = asyncio.Lock()

    async def is_speech(self, audio_chunk: bytes) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._detect, audio_chunk)

    def _detect(self, audio_chunk: bytes) -> bool:
        if len(audio_chunk) != CHUNK_BYTES:
            return False
        audio_float = (
            np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )
        ort_inputs = {
            "input": np.expand_dims(audio_float, axis=0),
            "sr": np.array([SAMPLE_RATE], dtype=np.int64),
            "state": self.state,
        }
        out, self.state = self.session.run(None, ort_inputs)
        return out[0][0] > VAD_THRESHOLD
