import json
import logging
from typing import Any

from openai import AsyncOpenAI

from src.config import LLM_MODEL_NAME, MAX_CHAT_HISTORY
from src.knowledge.retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BASE = (
    "Ты — приветливый голосовой ассистент приемной комиссии колледжа (девушка по имени Ксения). "
    "Твоя задача — консультировать абитуриентов. Общайся вежливо, коротко и естественно (максимум 2-3 предложения), "
    "так как твой ответ будет озвучиваться голосом.\n"
    "Если пользователь просит перезвонить, записать его, или задает сложный вопрос, "
    "мягко спроси его ИМЯ и ТЕЛЕФОН. Как только ты получишь эти данные, вызови инструмент (tool) 'create_ticket'."
)

RAG_CONTEXT_TEMPLATE = (
    "\n\nИспользуй следующую информацию о колледже для ответов:\n"
    "---\n{context}\n---\n"
    "Если информация не найдена в контексте выше, скажи что уточнишь и предложи оставить заявку."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Создать заявку/лид на обратный звонок или поступление",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя абонента"},
                    "phone": {"type": "string", "description": "Контактный телефон"},
                    "intent": {
                        "type": "string",
                        "description": "Краткая суть вопроса (1 предложение)",
                    },
                },
                "required": ["name", "phone", "intent"],
            },
        },
    }
]


class LLMAgent:
    """Qwen 2.5-14B через vLLM — диалог и вызов функций (create_ticket).

    FIX: chat_history обрезается до MAX_CHAT_HISTORY последних сообщений,
    чтобы не превысить max_model_len=4096 vLLM.

    RAG: если KnowledgeRetriever доступен, добавляет контекст из базы
    знаний колледжа в системный промпт.
    """

    def __init__(
        self, client: AsyncOpenAI, retriever: KnowledgeRetriever | None = None
    ) -> None:
        self.client = client
        self.retriever = retriever

    def _build_system_prompt(self, user_text: str) -> str:
        """Собирает системный промпт, при наличии RAG — с контекстом."""
        prompt = SYSTEM_PROMPT_BASE

        if self.retriever and self.retriever.is_available and user_text:
            chunks = self.retriever.retrieve(user_text, top_k=3)
            if chunks:
                context = "\n---\n".join(chunks)
                prompt += RAG_CONTEXT_TEMPLATE.format(context=context)

        return prompt

    async def get_response(
        self, chat_history: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any] | None]:
        """Возвращает (текст для озвучки, данные тикета или None)."""
        try:
            # Обрезка: system prompt + последние N сообщений
            trimmed = chat_history[-MAX_CHAT_HISTORY:]

            # Последнее сообщение пользователя для RAG-поиска
            last_user_text = ""
            for msg in reversed(trimmed):
                if msg["role"] == "user":
                    last_user_text = msg["content"]
                    break

            system_prompt = self._build_system_prompt(last_user_text)
            messages = [{"role": "system", "content": system_prompt}] + trimmed

            response = await self.client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.4,
                max_tokens=150,
            )

            message = response.choices[0].message
            reply_text = message.content or ""
            ticket_data = None

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "create_ticket":
                        ticket_data = json.loads(tool_call.function.arguments)
                        logger.info(f"LLM инициировала создание тикета: {ticket_data}")
                        if not reply_text:
                            reply_text = (
                                "Я записала ваши данные. Наши специалисты свяжутся "
                                "с вами в ближайшее время! Чем-то еще могу помочь?"
                            )

            return reply_text, ticket_data

        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")
            return "Извините, у меня небольшая заминка со связью.", None
