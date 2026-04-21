import asyncio
import logging

import numpy as np
import torch
import torchaudio.functional as AF

logger = logging.getLogger(__name__)

TARGET_SR = 16000  # WebSocket-клиент ожидает 16kHz


class TTSEngine:
    """F5-TTS с voice cloning по ref_audio + ref_text.

    Интерфейс: synthesize(text) -> 16-bit PCM bytes @ 16kHz mono.
    """

    def __init__(self, f5_model, ref_audio_path: str, ref_text: str) -> None:
        self.model = f5_model
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        if not text.strip():
            return b""

        wav, sr, _ = self.model.infer(
            ref_file=self.ref_audio_path,
            ref_text=self.ref_text,
            gen_text=text,
            show_info=lambda *a, **k: None,
            # progress=tqdm by default — печатает progress bar в stdout,
            # это ок для отладки; передача lambda/None падает т.к. f5-tts
            # дёргает progress.tqdm() где это модуль.
            speed=1.0,
            remove_silence=False,
        )

        wav_tensor = torch.from_numpy(np.asarray(wav)).float()
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.squeeze()
        if sr != TARGET_SR:
            wav_tensor = AF.resample(wav_tensor, orig_freq=sr, new_freq=TARGET_SR)

        audio_np = (wav_tensor.numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
        return audio_np.tobytes()
