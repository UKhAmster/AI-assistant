"""Загрузчик и индексатор документов базы знаний колледжа.

Читает Markdown-файлы из src/data/, разбивает на чанки,
строит FAISS-индекс через sentence-transformers.

Если sentence-transformers / faiss не установлены или файлов нет,
возвращает пустой индекс — приложение работает без RAG.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Путь к директории с данными колледжа
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Параметры чанкирования
CHUNK_SIZE = 400  # примерное кол-во символов на чанк
CHUNK_OVERLAP = 80


@dataclass(frozen=True)
class Chunk:
    """Один фрагмент документа."""
    text: str
    source: str  # имя файла-источника


@dataclass
class KnowledgeIndex:
    """FAISS-индекс + список чанков. Может быть пустым."""
    chunks: list[Chunk] = field(default_factory=list)
    index: object = None  # faiss.IndexFlatIP или None
    model: object = None  # SentenceTransformer или None

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0


def _split_text(text: str, source: str) -> list[Chunk]:
    """Разбивает текст на перекрывающиеся чанки."""
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, source=source))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _load_documents() -> list[Chunk]:
    """Читает все .md файлы из DATA_DIR и разбивает на чанки."""
    if not DATA_DIR.exists():
        logger.info("Директория данных %s не найдена, RAG отключен", DATA_DIR)
        return []

    md_files = sorted(DATA_DIR.glob("*.md"))
    if not md_files:
        logger.info("Нет .md файлов в %s, RAG отключен", DATA_DIR)
        return []

    all_chunks: list[Chunk] = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = _split_text(text, source=md_file.name)
        all_chunks.extend(chunks)
        logger.info("Загружено %d чанков из %s", len(chunks), md_file.name)

    logger.info("Всего чанков базы знаний: %d", len(all_chunks))
    return all_chunks


def build_index() -> KnowledgeIndex:
    """Строит FAISS-индекс из документов. Безопасен при отсутствии зависимостей."""
    chunks = _load_documents()
    if not chunks:
        return KnowledgeIndex()

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except ImportError as e:
        logger.warning(
            "sentence-transformers или faiss не установлены (%s), RAG отключен", e
        )
        return KnowledgeIndex()

    logger.info("Построение индекса: загрузка модели эмбеддингов...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Inner product на нормализованных = cosine similarity
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    logger.info("FAISS-индекс построен: %d векторов, dim=%d", index.ntotal, embeddings_np.shape[1])
    return KnowledgeIndex(chunks=chunks, index=index, model=model)
