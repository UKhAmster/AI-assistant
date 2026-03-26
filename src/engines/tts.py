import asyncio
import logging
import re

import numpy as np
import torch

from src.config import TTS_SAMPLE_RATE

logger = logging.getLogger(__name__)


class TTSEngine:
    """Silero TTS v4 (CUDA) — синтез русской речи, голос xenia, 24kHz."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        clean_text = re.sub(r"[^а-яА-ЯёЁ0-9\s.,!?\-]", "", text)
        if not clean_text.strip():
            return b""

        audio_tensor = self.model.apply_tts(
            text=clean_text,
            speaker="xenia",
            sample_rate=TTS_SAMPLE_RATE,
        )

        audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
        return audio_np.tobytes()
