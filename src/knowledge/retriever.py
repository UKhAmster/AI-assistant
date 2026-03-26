"""RAG-ретривер: ищет релевантные чанки по запросу пользователя.

Если индекс пуст (нет данных или зависимостей) — возвращает пустой список.
LLMAgent работает в обоих случаях: с контекстом и без.
"""

import logging

from src.knowledge.loader import KnowledgeIndex

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Поиск по базе знаний колледжа через FAISS."""

    def __init__(self, knowledge_index: KnowledgeIndex) -> None:
        self._index = knowledge_index

    @property
    def is_available(self) -> bool:
        return not self._index.is_empty

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Возвращает top_k наиболее релевантных фрагментов.

        Возвращает пустой список если:
        - индекс пуст (нет документов)
        - зависимости не установлены
        - запрос пустой
        """
        if self._index.is_empty or not query.strip():
            return []

        try:
            import numpy as np

            query_embedding = self._index.model.encode(
                [query], normalize_embeddings=True
            )
            query_np = np.array(query_embedding, dtype=np.float32)

            scores, indices = self._index.index.search(query_np, min(top_k, len(self._index.chunks)))

            results: list[str] = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0:
                    continue
                chunk = self._index.chunks[idx]
                results.append(chunk.text)
                logger.debug(
                    "RAG [%d]: score=%.3f source=%s text=%s...",
                    i, score, chunk.source, chunk.text[:60],
                )

            return results

        except Exception as e:
            logger.error("Ошибка RAG-поиска: %s", e)
            return []
