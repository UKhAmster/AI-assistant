import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from backend.config import LLM_MODEL_NAME, MAX_CHAT_HISTORY
from backend.knowledge.retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Системный промпт: архитектура голосового агента
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Ты — Ксения, сотрудник приёмной комиссии колледжа КЭСИ. Ты отвечаешь на телефонные звонки.
Ты консультируешь абитуриентов по вопросам поступления, специальностей, стоимости и документов.
Ты НЕ преподаватель, НЕ репетитор, НЕ готовишь к экзаменам. Ты работаешь только в приёмной комиссии.
Если вопрос не касается поступления в колледж КЭСИ — вежливо скажи, что это не в твоей компетенции.

═══ ТВОЯ ЦЕЛЬ ═══
Каждый разговор должен закончиться конкретным результатом:
• Абитуриент получил полный ответ на свой вопрос, ИЛИ
• Абитуриент записан на обратный звонок (create_ticket), ИЛИ
• Абитуриент переведён на оператора (transfer_to_operator).

═══ КАК ВЕСТИ РАЗГОВОР ═══
1. ИНИЦИАТИВА — ты ведёшь диалог, а не просто отвечаешь. После каждого ответа \
задай уточняющий вопрос, который продвигает разговор вперёд:
   — «Вы планируете поступать после 9 или 11 класса?»
   — «Какое направление вам ближе — техническое или гуманитарное?»
   — «Вам удобнее очная форма или хотите совмещать с работой?»
   Вопрос должен логически вытекать из того, что сказал абитуриент. \
   НИКОГДА не заканчивай фразой «Что-то ещё подсказать?» — это звучит как автоответчик.

2. КРАТКОСТЬ — твои ответы озвучиваются голосом. Говори 2–3 предложения + вопрос. \
   Не перечисляй длинные списки — лучше спроси, что именно интересно, и расскажи конкретно.

3. ЭМПАТИЯ — если вопрос касается стоимости, сроков или документов, \
   начни с нормализующей фразы: «Это один из самых частых вопросов» \
   или «Разберёмся, всё не так сложно».

4. МОСТИКИ — при переходе к новой теме свяжи с предыдущим: \
   «Кстати, раз вы интересуетесь программированием — у нас есть новое направление по ИИ...»

5. ПАМЯТЬ — если абитуриент уже называл имя, класс, интересующую специальность — \
   используй эту информацию в следующих ответах. Не переспрашивай то, что уже знаешь.

═══ ЭТАПЫ РАЗГОВОРА ═══
{stage_instruction}

═══ ЕСЛИ НЕ ПОНИМАЕШЬ ВОПРОС ═══
• НИКОГДА не переводи на оператора если не поняла вопрос. Вместо этого переспроси:
  — «Извините, я не совсем поняла. Вы спрашиваете о специальностях, стоимости или документах?»
  — «Можете повторить вопрос другими словами? Я хочу помочь.»
• Предложи варианты: «Я могу рассказать о направлениях, формах обучения, стоимости или документах для поступления. Что вас интересует?»

═══ ПЕРЕВОД НА ОПЕРАТОРА ═══
• Вызывай transfer_to_operator ТОЛЬКО если абитуриент ДОСЛОВНО попросил: \
  «позовите оператора», «переведите на человека», «хочу поговорить с живым человеком».
• Если просто непонятный вопрос — НЕ переводи, а переспроси.
• Если вопрос сложный но по теме — попробуй ответить из контекста, или скажи «К сожалению, \
  у меня нет точной информации по этому вопросу, но я могу уточнить. Что ещё вас интересует?»

═══ СБОР КОНТАКТОВ ═══
• Собирай имя и телефон ТОЛЬКО когда абитуриент сам попросил перезвонить, \
   или когда ты предложила и он согласился. Тогда вызови create_ticket.
• НИКОГДА не спрашивай контакты в первых репликах.

═══ ЯЗЫК ═══
ВСЕГДА отвечай ТОЛЬКО на русском. Никакого китайского, английского или других языков.\
"""

# Инструкции по этапам разговора
STAGE_DISCOVERY = (
    "ТЕКУЩИЙ ЭТАП: Знакомство и выявление потребностей.\n"
    "Задача: узнать, что интересует абитуриента — класс (9 или 11), "
    "интересующее направление, форму обучения. "
    "Задавай открытые вопросы. Не перегружай информацией."
)

STAGE_INFORM = (
    "ТЕКУЩИЙ ЭТАП: Информирование и помощь.\n"
    "Задача: давать конкретные ответы по специальностям, стоимости, срокам. "
    "Используй данные из контекста. После ответа задай уточняющий вопрос, "
    "который углубляет интерес. Продолжай помогать столько, сколько нужно."
)

RAG_CONTEXT_TEMPLATE = (
    "\n\n═══ ИНФОРМАЦИЯ О КОЛЛЕДЖЕ ═══\n{context}"
)

TRANSFER_REPLY = "Конечно, сейчас переведу, пожалуйста ожидайте."

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "transfer_to_operator",
            "description": (
                "Перевести звонок на оператора. Вызывай ТОЛЬКО когда "
                "собеседник ЯВНО и ДОСЛОВНО попросил оператора или живого человека. "
                "НЕ вызывай если просто не понял вопрос — переспроси вместо этого."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Краткая причина перевода",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "Создать заявку на обратный звонок. Вызывай когда "
                "собеседник согласился оставить контакты для обратного звонка."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя абонента"},
                    "phone": {"type": "string", "description": "Контактный телефон"},
                    "intent": {
                        "type": "string",
                        "description": "Краткая суть вопроса",
                    },
                },
                "required": ["name", "phone", "intent"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Пост-обработка текста
# ---------------------------------------------------------------------------

_NON_RUSSIAN_RE = re.compile(r"[\u2E80-\u9FFF\uF900-\uFAFF\uFE30-\uFE4F]")
_STUTTER_RE = re.compile(r"(\w{2,}?)\1{2,}")
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _sanitize(text: str) -> str:
    """Чистит ответ LLM: think-блоки, заикания, нерусский текст."""
    text = _THINK_RE.sub("", text).strip()
    text = _STUTTER_RE.sub(r"\1", text)
    if not _NON_RUSSIAN_RE.search(text):
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = [s for s in sentences if not _NON_RUSSIAN_RE.search(s)]
    if clean:
        logger.warning("Обрезан нерусский текст из ответа LLM")
        return " ".join(clean)
    logger.warning("Весь ответ LLM на нерусском, заменён заглушкой")
    return "Извините, я немного запуталась. Повторите, пожалуйста, ваш вопрос?"


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """Голосовой агент приёмной комиссии — ведёт диалог по этапам."""

    def __init__(
        self, client: AsyncOpenAI, retriever: KnowledgeRetriever | None = None
    ) -> None:
        self.client = client
        self.retriever = retriever

    @staticmethod
    def _get_stage(user_turn_count: int) -> str:
        """Определяет этап разговора по количеству реплик."""
        if user_turn_count <= 2:
            return STAGE_DISCOVERY
        return STAGE_INFORM

    def _build_system_prompt(self, user_text: str, user_turn_count: int) -> str:
        """Собирает системный промпт с RAG-контекстом и этапом разговора."""
        stage_instruction = self._get_stage(user_turn_count)
        prompt = SYSTEM_PROMPT.format(stage_instruction=stage_instruction)

        if self.retriever and self.retriever.is_available and user_text:
            chunks = self.retriever.retrieve(user_text, top_k=2)
            if chunks:
                # Ограничиваем контекст ~1500 символов
                context_parts = []
                total = 0
                for c in chunks:
                    if total + len(c) > 1500:
                        break
                    context_parts.append(c)
                    total += len(c)
                context = "\n---\n".join(context_parts)
                prompt += RAG_CONTEXT_TEMPLATE.format(context=context)

        return prompt

    async def get_response(
        self, chat_history: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any] | None]:
        """Возвращает (текст для озвучки, данные тикета или None)."""
        try:
            trimmed = chat_history[-MAX_CHAT_HISTORY:]
            user_turn_count = sum(1 for m in chat_history if m["role"] == "user")

            last_user_text = ""
            for msg in reversed(trimmed):
                if msg["role"] == "user":
                    last_user_text = msg["content"]
                    break

            system_prompt = self._build_system_prompt(last_user_text, user_turn_count)
            messages = [{"role": "system", "content": system_prompt}] + trimmed

            response = await self.client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.4,
                max_tokens=250,
                frequency_penalty=0.3,
                presence_penalty=0.2,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            message = response.choices[0].message
            reply_text = _sanitize(message.content or "")
            ticket_data = None

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "create_ticket":
                        ticket_data = json.loads(tool_call.function.arguments)
                        logger.info("LLM: создание тикета: %s", ticket_data)
                        if not reply_text:
                            reply_text = (
                                "Записала! Наши специалисты свяжутся с вами "
                                "в ближайшее время."
                            )
                    elif tool_call.function.name == "transfer_to_operator":
                        reason = json.loads(tool_call.function.arguments).get("reason", "")
                        logger.info("LLM: перевод на оператора: %s", reason)
                        reply_text = TRANSFER_REPLY

            return reply_text, ticket_data

        except Exception as e:
            logger.error("Ошибка LLM: %s", e)
            return "Извините, у меня небольшая заминка со связью.", None
