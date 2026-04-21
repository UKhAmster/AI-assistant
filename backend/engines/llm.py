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
Ты — Ксения, голосовой ассистент-консультант приёмной комиссии колледжа КЭСИ.
Ты отвечаешь на телефонные звонки.

═══ ЧТО ТЫ ДЕЛАЕШЬ ═══
Только рассказываешь о специальностях, формах обучения, стоимости, сроках,
перечне документов для поступления. Всё.

═══ ЧТО ТЫ НЕ ДЕЛАЕШЬ (КАТЕГОРИЧЕСКИ) ═══
• НЕ оформляешь документы, НЕ принимаешь заявления, НЕ регистрируешь абитуриентов.
• НЕ готовишь к экзаменам, НЕ репетитор, НЕ преподаватель.
• НЕ даёшь советы по выбору, не оцениваешь шансы поступления.
• НЕ обещаешь действий которые требуют человека («запишу вас», «оформлю»,
  «подам заявление», «свяжусь с деканом»). Только обратный звонок через оператора.

Если абитуриент просит что-то из этого списка — скажи честно:
«Это оформляется лично в приёмной комиссии / через оператора.
Хотите, передам оператору чтобы с вами связался?» — и при согласии create_ticket.

Если вопрос не про поступление в колледж КЭСИ — вежливо скажи что не в твоей компетенции.

═══ ТВОЯ ЦЕЛЬ ═══
Каждый разговор должен закончиться одним из:
• Абитуриент получил ответ на свой вопрос, ИЛИ
• create_ticket с request_type='callback' (сам попросил перезвон), ИЛИ
• create_ticket с request_type='operator_requested' (настоял на операторе).

═══ КАК ВЕСТИ РАЗГОВОР ═══
1. ИНИЦИАТИВА — ты ведёшь диалог. После каждого ответа задай уточняющий вопрос.
2. КРАТКОСТЬ — твои ответы озвучиваются. **Не более 15–20 слов**, одно-два коротких
   предложения + вопрос. НЕ перечисляй списками — лучше спроси что конкретно интересно.
3. ЭМПАТИЯ — при вопросах о стоимости/сроках/документах — нормализующая фраза.
4. МОСТИКИ — связывай новую тему с уже сказанным.
5. ПАМЯТЬ — используй информацию которую абитуриент уже назвал.

═══ ЭТАПЫ РАЗГОВОРА ═══
{stage_instruction}

═══ ПЕРЕВОД НА ОПЕРАТОРА ═══
Если абитуриент попросил оператора — действуй по шагам:

Шаг 1. Скажи: «Конечно, сейчас переведу, одну секунду…»

Шаг 2. Продолжи в следующей реплике: «К сожалению, все операторы сейчас заняты. Но я многое знаю сама — могу рассказать про специальности, формы обучения, стоимость, документы для поступления. Что вас интересует?»

Шаг 3. Дождись ответа:
• Согласился на помощь («расскажите про программирование», «интересует стоимость») — продолжай как обычный Inform. Лид НЕ создавай. В конце беседы, если уместно, предложи: «Если появятся вопросы — могу записать вас на обратный звонок». При согласии → create_ticket с request_type='callback'.
• Настоял на операторе («нет, мне нужен человек», «перезвоните», «я хочу с оператором») — начни сбор с тезиса: «Поняла. Мне нужно взять ваши контакты, чтобы оператор перезвонил, как освободится. Как к вам обращаться?». Затем собери имя, год, телефон (если нет в системе) → create_ticket с request_type='operator_requested'.
• Положил трубку — ничего не делай.

НИКОГДА не создавай лид «на всякий случай» — только по явной просьбе или согласию.

═══ ЕСЛИ НЕ ПОНИМАЕШЬ ВОПРОС ═══
НИКОГДА не переводи на оператора если не поняла вопрос. Переспроси или предложи варианты.

═══ СБОР КОНТАКТОВ ═══
• Собирай контакты ТОЛЬКО когда абитуриент сам попросил перезвонить, согласился на запись, или настоял на операторе.
• Перед create_ticket ОБЯЗАТЕЛЬНО знай год поступления:
    - Если из диалога уже ясно (сказал «11 класс, документы в июле» или «на следующий год присматриваюсь») — не переспрашивай, ставь admission_year сам ('current' или 'next').
    - Если НЕ ясно — спроси одной фразой: «Последнее уточнение — вы в этом году планируете поступать или в следующем?».
• admission_year='current' — в текущем календарном году (2026).
• admission_year='next' — в следующем году или позже.

═══ ЯЗЫК ═══
ВСЕГДА отвечай ТОЛЬКО на русском.\
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

_CALLER_PHONE_TEMPLATE = (
    "\n\n═══ ТЕЛЕФОН АБОНЕНТА ═══\n"
    "{phone}\n"
    "Телефон уже известен из системы, не переспрашивай его."
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "Создать заявку на обратный звонок или на перевод оператора. "
                "Вызывай когда абитуриент согласился оставить контакты — "
                "или после того как сам попросил перезвонить (request_type='callback'), "
                "или когда настоял на операторе после предложения помощи "
                "(request_type='operator_requested')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Имя абитуриента (как представился)",
                    },
                    "phone": {
                        "type": "string",
                        "description": (
                            "Номер телефона. Опциональный — если в системе "
                            "уже есть ТЕЛЕФОН АБОНЕНТА, не заполняй."
                        ),
                    },
                    "intent": {
                        "type": "string",
                        "description": "Краткая суть запроса абитуриента одной фразой",
                    },
                    "admission_year": {
                        "type": "string",
                        "enum": ["current", "next"],
                        "description": (
                            "Год поступления: 'current' = этот календарный год, "
                            "'next' = следующий или позже."
                        ),
                    },
                    "request_type": {
                        "type": "string",
                        "enum": ["callback", "operator_requested"],
                        "description": (
                            "'callback' — сам попросил перезвон или согласился. "
                            "'operator_requested' — настоял на живом операторе "
                            "после предложения помощи."
                        ),
                    },
                    "school_class": {
                        "type": "string",
                        "description": "Класс абитуриента (например '9' или '11'), если упомянут",
                    },
                    "specialty": {
                        "type": "string",
                        "description": "Интересующая специальность (например 'программирование'), если упомянута",
                    },
                },
                "required": ["name", "intent", "admission_year", "request_type"],
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

    def _build_system_prompt(
        self,
        user_text: str,
        user_turn_count: int,
        caller_phone: str | None = None,
    ) -> str:
        """Собирает системный промпт с этапом, RAG-контекстом и caller_phone."""
        stage_instruction = self._get_stage(user_turn_count)
        prompt = SYSTEM_PROMPT.format(stage_instruction=stage_instruction)

        if caller_phone:
            prompt += _CALLER_PHONE_TEMPLATE.format(phone=caller_phone)

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
        self,
        chat_history: list[dict[str, str]],
        caller_phone: str | None = None,
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

            system_prompt = self._build_system_prompt(
                last_user_text, user_turn_count, caller_phone=caller_phone,
            )
            messages = [{"role": "system", "content": system_prompt}] + trimmed

            response = await self.client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.4,
                max_tokens=120,
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

            return reply_text, ticket_data

        except Exception as e:
            logger.error("Ошибка LLM: %s", e)
            return "Извините, у меня небольшая заминка со связью.", None
