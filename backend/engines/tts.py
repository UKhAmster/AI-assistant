import asyncio
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


class TTSEngine:
    """Qwen3-TTS — синтез русской речи с описанием голоса или клонированием."""

    def __init__(self, model, voice_prompt=None) -> None:
        self.model = model
        self.voice_prompt = voice_prompt

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        clean_text = re.sub(r"[^а-яА-ЯёЁ0-9\s.,!?\-:;()\"']", "", text)
        if not clean_text.strip():
            return b""

        if self.voice_prompt is not None:
            # Клонирование голоса из референса
            wavs, sr = self.model.generate_voice_clone(
                text=clean_text,
                language="Russian",
                voice_clone_prompt=self.voice_prompt,
            )
        else:
            # Генерация голоса по текстовому описанию.
            # Инструкция на английском — модель обучена на EN/ZH корпусе, EN надёжнее.
            wavs, sr = self.model.generate_voice_design(
                text=clean_text,
                language="Russian",
                instruct=(
                    "young bright female voice, high-pitched, fast tempo, "
                    "cheerful and energetic tone, warm and friendly delivery, "
                    "clear articulation, Russian native speaker"
                ),
                temperature=0.9,
                top_p=0.9,
            )

        audio_np = (wavs[0] * 32767).astype(np.int16)
        return audio_np.tobytes()
