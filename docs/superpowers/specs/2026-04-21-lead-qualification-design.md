# Design: Lead Qualification & Operator-Transfer Fallback

- **Date:** 2026-04-21
- **Branch:** v1.0
- **Author:** dev (user 277 in Bitrix, vasonov04@gmail.com)
- **Status:** Approved — ready for implementation plan

---

## 1. Context

Голосовой ассистент «Ксения» принимает звонки в приёмную комиссию колледжа КЭСИ. Текущая логика диалога и интеграция с Bitrix24 устарели:

- `transfer_to_operator` — лишь голосовая фраза, ничего не делает
- `create_ticket` — пишет в Bitrix только имя, телефон, intent; кастомные поля не заполняются
- Нет квалификации лидов на «горячие» / «холодные» — операторы обрабатывают все одинаково и тратят время на тех, кто поступает не в текущем году
- Нет fallback-логики на случай технических сбоев — при поломке бота абонент просто «повисает»

Бизнес-цель — чтобы ассистент был полноценным помощником первой линии: отвечал на вопросы по RAG-базе, фильтровал нерелевантные звонки, а лиды в CRM шли уже квалифицированными и с полным контекстом разговора.

## 2. Goals / Non-goals

### Goals

- Переписать логику «перевод на оператора»: сначала предложить помощь, только при настойчивости — завести лид
- Квалифицировать лиды по году поступления через кастомное поле `UF_CRM_AI_QUALITY` (значения «Качественный» / «Некачественный»)
- Передавать в Bitrix класс, специальность и полный транскрипт диалога
- Обеспечить fallback на оператора при любой технической неисправности бота (в текущей итерации — через срочный лид, после Asterisk-интеграции — через реальный SIP-transfer)
- Подготовить код к будущему получению `caller_phone` из АТС (Caller ID) — сделать параметр опциональным

### Non-goals

- Реальный SIP-transfer на оператора (ждёт интеграции Мегафон АТС + Asterisk, см. [research doc](../../research/2026-04-21-megafon-pbx-integration.md))
- Дедупликация лидов по телефону (отложено)
- Durable queue для неотправленных лидов (отложено)
- Barge-in / прерывание TTS (отложено)

## 3. Обсуждённые и зафиксированные решения

| # | Решение | Обоснование |
|---|---|---|
| 1 | Реальной интеграции с АТС **не делаем** в этой итерации | Мегафон не имеет public programmable-voice API уровня Twilio. Нужен Asterisk-слой — отдельный проект. |
| 2 | `transfer_to_operator` tool **удаляется**, его работу берёт `create_ticket` с `request_type` | Единая точка входа в CRM, меньше state в `session.py` |
| 3 | Маркер качества = кастомное поле `UF_CRM_AI_QUALITY` (ID=415), enum: «Качественный» (ID=173) / «Некачественный» (ID=175) | Не трогаем воронку (не ломаем отчёты операторов), квалификация ортогональна статусу |
| 4 | Вопрос про год задаётся **перед `create_ticket`**, с адаптивностью (если из диалога уже ясно — не переспрашивать) | Баланс между естественностью и гарантией заполнения поля |
| 5 | При настойчивости на операторе — сбор контактов начинается с тезиса **«мне нужно взять ваши контакты, чтобы оператор перезвонил как освободится»** | Объясняет цель сбора данных, снижает сопротивление |
| 6 | При отказе бота **принудительно идём через fallback на оператора** (в заглушке — срочный лид `[СРОЧНО]`) | Безопасность абонента — ни один звонок не должен остаться без реакции |
| 7 | `caller_phone` опционален в `create_ticket`, берётся из `DialogSession`, в веб-клиенте доступен через query-param `?caller_phone=+7...` | Готовим код к пересадке на АТС без переделки |

## 4. Architecture

Три слоя изменений, все в рамках существующей pipeline `VAD → STT → LLM → TTS → send_to_bitrix`:

1. **LLM** (`backend/engines/llm.py`): переписанный системный промпт + расширенная схема `create_ticket`
2. **Bitrix-слой** (`backend/services/bitrix.py`): расширенный payload + резолвер enum-ID при старте
3. **Session-слой** (`backend/services/session.py`): `caller_phone` параметр, передача `chat_history` в `send_to_bitrix`, `_fatal_fallback`

Весь flow VAD → STT → LLM → TTS остаётся без изменений. Новое — только в обработке результата LLM (tool_call) и в построении payload для CRM.

## 5. Components

### 5.1 `backend/engines/llm.py`

**Изменения в промпте:**

Секция `═══ ПЕРЕВОД НА ОПЕРАТОРА ═══` переписывается по шагам:

> Шаг 1. Скажи: «Конечно, сейчас переведу, одну секунду…»
>
> Шаг 2. Продолжи в следующей реплике: «К сожалению, все операторы сейчас заняты. Но я многое знаю сама — могу рассказать про специальности, формы обучения, стоимость, документы для поступления. Что вас интересует?»
>
> Шаг 3. Дождись ответа:
> - **Согласился на помощь** — продолжай как Inform. Лид не создавай. В конце, если появилась возможность, предложи: «Если появятся вопросы — могу записать вас на обратный звонок». При согласии → `create_ticket(request_type='callback')`.
> - **Настоял на операторе** — сбор контактов начни с тезиса «мне нужно взять ваши контакты, чтобы оператор перезвонил, как освободится». Собери имя, год, телефон (если `caller_phone` неизвестен) → `create_ticket(request_type='operator_requested')`.
> - **Положил трубку** — ничего не делай, WebSocket закроется сам.
>
> **Никогда** не создавай лид «на всякий случай» — только по явной просьбе или согласию.

Секция `═══ СБОР КОНТАКТОВ ═══` дополняется правилом:

> Перед `create_ticket` обязательно знай год поступления. Если из диалога уже ясно («я в 11 классе», «на следующий год присматриваюсь») — не переспрашивай. Если не ясно — спроси одной фразой: «Последнее уточнение — вы в этом году планируете или в следующем?».

Динамический блок `═══ ТЕЛЕФОН АБОНЕНТА ═══` вставляется в system prompt только если `caller_phone is not None`:

> ТЕЛЕФОН АБОНЕНТА: +7XXXXXXXXXX. Он уже известен из системы, не переспрашивай.

**Изменения в `TOOLS`:**

- **Удалить** `transfer_to_operator`.
- **Расширить** `create_ticket`:

```python
{
    "name": "create_ticket",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "phone": {"type": "string"},  # теперь опциональный
            "intent": {"type": "string"},
            "admission_year": {"type": "string", "enum": ["current", "next"]},
            "request_type": {"type": "string", "enum": ["callback", "operator_requested"]},
            "school_class": {"type": "string"},   # "9" / "11" / ...
            "specialty": {"type": "string"},
        },
        "required": ["name", "intent", "admission_year", "request_type"],
    },
}
```

### 5.2 `backend/services/bitrix.py`

**Новая функция:**

```python
async def load_ai_quality_enum_ids(webhook_url: str) -> dict[str, int]:
    """Резолвит enum-ID для Качественный/Некачественный при старте.

    Возвращает {"current": <id_Качественный>, "next": <id_Некачественный>}.
    Бросает RuntimeError если не нашло ожидаемые VALUE-строки.
    """
```

Идентификация по строгому совпадению `VALUE == "Качественный"` / `"Некачественный"` (XML_ID не работает через REST Bitrix — это известное ограничение).

**Изменения в `send_to_bitrix(ticket_data, chat_history, enum_ids, caller_phone)`:**

Новые параметры:
- `chat_history: list[dict]` — для транскрипта в COMMENTS
- `enum_ids: dict[str, int]` — заранее загруженный mapping
- `caller_phone: str | None` — используется если `ticket_data["phone"]` пустой

Payload для обычных `callback` / `operator_requested`:

```python
{
    "fields": {
        "TITLE": f"Звонок: {ticket_data['intent'][:60]}",
        "NAME": ticket_data["name"],
        "PHONE": [{"VALUE": phone, "VALUE_TYPE": "MOBILE"}],
        "SOURCE_ID": "CALLBACK",
        "COMMENTS": _format_comments(ticket_data, chat_history),
        "UF_CRM_AI_QUALITY": enum_ids[ticket_data["admission_year"]],
        "UF_CRM_KAKOIKLASSVIZ": ticket_data.get("school_class", ""),
        "UF_CRM_KAKAYASPETSIA": ticket_data.get("specialty", ""),
    }
}
```

Для `fatal_fallback`:
- `TITLE` = `"СРОЧНО: сбой бота — оператор срочно перезвонить"`
- COMMENTS начинается с `[СРОЧНО: технический сбой]` + reason + транскрипт
- `UF_CRM_AI_QUALITY` **не устанавливается** (квалификация не сделана)

`_format_comments`: префикс по `request_type` → короткое intent → разделитель → транскрипт с «Оператор:» / «Абитуриент:».

Phone-логика: если `ticket_data["phone"]` есть — нормализуем. Иначе подставляем `caller_phone`. Если нет ни того ни другого и `request_type != "fatal_fallback"` → warning + лид не создаётся.

### 5.3 `backend/services/session.py`

**`DialogSession.__init__`:**
- Новый параметр `caller_phone: str | None = None`
- Новый параметр `enum_ids: dict[str, int] | None = None`
- Новое поле `self._consecutive_errors: int = 0` для подсчёта триггеров `_fatal_fallback`

**`_process_turn`:**
- При получении `ticket_data` — передать в `send_to_bitrix(ticket_data, self.chat_history, self.enum_ids, self.caller_phone)`
- При исключении от LLM — увеличить `self._consecutive_errors`; при `≥ FATAL_CONSECUTIVE_ERRORS` → `_fatal_fallback("LLM repeatedly failed")`
- При успехе — сбросить счётчик

**Новый метод `_fatal_fallback(reason: str)`:**
- Озвучить hardcoded apology
- Если `caller_phone` отсутствует — одна попытка собрать голосом
- Собрать best-effort `ticket_data` с `request_type="fatal_fallback"` и `admission_year=None`
- Вызвать `send_to_bitrix`
- Попрощаться, закрыть WebSocket

**`run()`:** любое unhandled exception → `_fatal_fallback(str(exc))` перед закрытием.

### 5.4 `backend/main.py`

В `lifespan` после инициализации `retriever`:

```python
app.state.bitrix_enum_ids = None
if BITRIX_WEBHOOK_URL:
    app.state.bitrix_enum_ids = await load_ai_quality_enum_ids(BITRIX_WEBHOOK_URL)
    logger.info("Bitrix enum IDs: %s", app.state.bitrix_enum_ids)
```

Если `BITRIX_WEBHOOK_URL` задан, но резолвер упал — `RuntimeError` и приложение не стартует (fail-fast).

В `websocket_endpoint`:

```python
query_params = dict(websocket.query_params)
caller_phone = query_params.get("caller_phone")  # для теста в веб-клиенте

session = DialogSession(
    websocket=websocket,
    vad=VADEngine(...),
    stt=STTEngine(...),
    tts=TTSEngine(...),
    llm=LLMAgent(...),
    caller_phone=caller_phone,
    enum_ids=app.state.bitrix_enum_ids,
)
```

### 5.5 Константы

В `backend/config.py`:

```python
FATAL_CONSECUTIVE_ERRORS: int = int(os.getenv("FATAL_CONSECUTIVE_ERRORS", "3"))
```

## 6. Data Flow

### Сценарий A — callback без перевода (базовый)

Абитуриент сам просит перезвонить. LLM собирает имя/телефон/год → `create_ticket(request_type='callback')` → лид с `[Обратный звонок]` в COMMENTS.

### Сценарий B — просьба оператора, помощь принята (лид **не** создаётся)

```
Абитуриент: "Я бы хотел с оператором поговорить"
Агент: "Конечно, сейчас переведу, одну секунду…"
Агент: "К сожалению, операторы заняты. Но я многое знаю — специальности,
        формы, стоимость, документы. Что интересует?"
Абитуриент: "Расскажите про программирование"
Агент: [обычный Inform + RAG]
...
Агент (в конце): "Если появятся вопросы — могу записать на обратный звонок."
Абитуриент: "Да, запишите"
→ create_ticket(request_type='callback', admission_year=..., ...)
```

### Сценарий C — просьба оператора, настоял (лид создаётся)

```
Абитуриент: "Я бы хотел с оператором поговорить"
Агент: "Конечно, сейчас переведу, одну секунду…"
Агент: "К сожалению, операторы заняты. Но я могу помочь — специальности,
        формы, стоимость, документы. Что интересует?"
Абитуриент: "Нет, мне именно оператора, пусть перезвонят"
Агент: "Поняла. Мне нужно взять ваши контакты, чтобы оператор перезвонил,
        как освободится. Как к вам обращаться?"
Абитуриент: "Иван"
Агент: "Иван, продиктуйте телефон."   ← пропускается если caller_phone задан
Абитуриент: "8 900…"
Агент: "Записала. Вы в этом году поступаете или в следующем?"
Абитуриент: "В этом, после 11 класса, хочу на программирование"
Агент: "Спасибо, Иван! Оператор вам перезвонит."
→ create_ticket(request_type='operator_requested',
                admission_year='current', school_class='11',
                specialty='программирование')
```

Bitrix payload:

```json
{
  "fields": {
    "TITLE": "Звонок: Просил оператора, хочет на программирование",
    "NAME": "Иван",
    "PHONE": [{"VALUE": "+79001234567", "VALUE_TYPE": "MOBILE"}],
    "SOURCE_ID": "CALLBACK",
    "COMMENTS": "[Запрошен оператор]\nПросил оператора, хочет на программирование\n\n— Транскрипт ——\n...",
    "UF_CRM_AI_QUALITY": 173,
    "UF_CRM_KAKOIKLASSVIZ": "11",
    "UF_CRM_KAKAYASPETSIA": "программирование"
  }
}
```

### Сценарий D — fatal fallback

Любая критическая ошибка → TTS «технические сложности, зафиксирую заявку» → best-effort сбор контактов → лид `[СРОЧНО]` без `UF_CRM_AI_QUALITY` → закрытие WebSocket.

## 7. Error Handling

### 7.1 Webhook Bitrix недоступен
3 попытки exp-backoff (1→2→4с). После — ERROR в лог с полным payload. Лид теряется (durable queue в future work).

### 7.2 Enum IDs не резолвятся
Приложение не стартует. RuntimeError с инструкцией «проверь UI: CRM → Настройки → …».

### 7.3 Невалидный телефон
`normalize_phone` вернул `None` → LLM переспрашивает. После второй неудачи → warning, лид не создаётся. Для АТС-phone — та же логика.

### 7.4 `admission_year` не задан
Поле `UF_CRM_AI_QUALITY` **не добавляется** в payload (остаётся пустым в Bitrix). WARNING в лог. Пустое поле = явный сигнал «AI не квалифицировал», оператор разбирается вручную. Не делаем дефолт — это скрыло бы баг.

### 7.5 `request_type` не задан
Default `callback` (самое мягкое предположение).

### 7.6 Прерванный звонок
WebSocket disconnect → session.save_log() сохраняет JSON в `logs/sessions/`. Лид не создаётся, оператор при необходимости может посмотреть файл.

### 7.7 Мусор от LLM
Существующий `_sanitize` (think-блоки, заикания, иероглифы). Не меняем.

### 7.8 Двойной вызов `create_ticket`
Known issue — создадутся 2 лида. Redemption: флаг `self._ticket_created` (future work, пока YAGNI).

### 7.9 Fatal fallback (принцип)

Триггеры:

| Событие | Порог |
|---|---|
| LLM-запрос падает network/timeout | 3 подряд |
| STT пустой на речь (VAD-сработал) | 3 подряд |
| TTS бросил исключение | 1 раз |
| Unhandled exception в `run` / `_process_turn` | 1 раз |
| Счётчик «Извините, заминка» | ≥ 3 |

Порог `FATAL_CONSECUTIVE_ERRORS=3` в config.

Поведение-заглушка: TTS apology → best-effort phone → лид `[СРОЧНО]` → hangup.

В следующей итерации (Asterisk): `_fatal_fallback` сначала пытается SIP REFER на hunt-группу с таймаутом 20 секунд, fallback на текущую логику только если никто не ответил.

## 8. Testing

### 8.1 Unit-тесты

| Файл | Что проверяет |
|---|---|
| `test/test_phone_normalizer.py` | уже есть |
| `test/test_bitrix_payload.py` | все комбинации request_type × admission_year × optional → правильный JSON |
| `test/test_enum_resolver.py` | `load_ai_quality_enum_ids` на mock httpx: успех, missing field, weird values → RuntimeError |
| `test/test_llm_prompt.py` | `_build_system_prompt`: caller_phone блок есть/нет, блок оператора присутствует |

Pytest + mock httpx через `httpx.MockTransport`.

### 8.2 Live-интеграционные тесты (extend `test/test_bitrix_live.py`)

- `test_hot_callback()` — admission_year='current', request_type='callback'
- `test_cold_callback()` — admission_year='next'
- `test_operator_requested()` — `[Запрошен оператор]` в COMMENTS
- `test_fatal_fallback()` — `[СРОЧНО]` + пустой `UF_CRM_AI_QUALITY`

После каждого теста — `crm.lead.delete` (cleanup). Не в CI.

### 8.3 Manual QA через веб-клиент (`docs/manual-qa-checklist.md`)

Пять сценариев:
1. Callback (прямая просьба перезвонить)
2. Operator-help-accepted (просьба оператора → согласие на помощь → callback в конце)
3. Operator-insisted (просьба оператора → настоял → лид)
4. Fatal fallback (остановить vLLM → сделать 3 реплики → лид `[СРОЧНО]`)
5. caller_phone query-param (`/ws?caller_phone=+79001234567` → LLM не спрашивает телефон)

### 8.4 Что не тестируем

- Asterisk SIP (его нет)
- Нагрузочное
- Качество LLM-ответов (отдельный QA-процесс на текстах)

## 9. Future Work

1. **🔴 Asterisk + Мегафон SIP — критично для продакшна.** См. [research doc](../../research/2026-04-21-megafon-pbx-integration.md). Без этого `_fatal_fallback` не даёт реального оператора, `caller_phone` всегда неизвестен.
2. Durable queue для неотправленных лидов (файловая очередь + worker)
3. Дедупликация лидов по телефону через `crm.lead.list?filter[PHONE]=...`
4. Защита от двойного `create_ticket` в одной сессии
5. `UF_CRM_PREDPOCHTITEL` — если операторы подтвердят смысл, добавить `preferred_form` в tool
6. Barge-in / прерывание TTS
7. Prometheus-экспорт метрик latency
8. Проверить что `MAX_CHAT_HISTORY` автокомпакт реально работает

## 10. Known issues

- `UF_CRM_AI_QUALITY` enum XML_ID не сохраняется через REST Bitrix → резолвим по строковому `VALUE`. Хрупко к переименованию значения в UI.
- `SHOW_FILTER: "S"` не принялся при `crm.lead.userfield.add` (вернулся как `"N"`) — не критично, поле работает в фильтре всё равно.
- «Общий вид карточки» для нового поля настраивается **только в UI Bitrix**, REST-метода нет. Это один ручной шаг при добавлении каждого нового UF_*-поля.

## 11. Approvals

| Шаг | Дата | Статус |
|---|---|---|
| § 1 Architecture | 2026-04-21 | ✅ approved |
| § 2 Components | 2026-04-21 | ✅ approved |
| § 3 Data flow (с правкой фразы про сбор контактов) | 2026-04-21 | ✅ approved |
| § 4 Error handling (+ 4.9 fatal fallback) | 2026-04-21 | ✅ approved |
| § 5 Testing | 2026-04-21 | ✅ approved |
| § 6 Future work | 2026-04-21 | ✅ approved |
