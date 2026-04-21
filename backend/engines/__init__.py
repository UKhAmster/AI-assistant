"""Lazy-loading package to avoid importing heavy ML deps at import time."""
from __future__ import annotations

__all__ = ["VADEngine", "STTEngine", "TTSEngine", "LLMAgent"]

_module_map = {
    "VADEngine": "backend.engines.vad",
    "STTEngine": "backend.engines.stt",
    "TTSEngine": "backend.engines.tts",
    "LLMAgent": "backend.engines.llm",
}


def __getattr__(name: str):  # noqa: ANN001, ANN202
    if name in _module_map:
        import importlib
        module = importlib.import_module(_module_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
