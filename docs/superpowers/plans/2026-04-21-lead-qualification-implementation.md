# Lead Qualification & Operator-Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать логику квалификации лидов по году поступления, переписать «перевод на оператора» по принципу «сначала предложить помощь», добавить fatal-fallback при любых ошибках бота, подготовить код к получению `caller_phone` из АТС.

**Architecture:** Изменения в 3-х слоях существующей pipeline (`VAD → STT → LLM → TTS → send_to_bitrix`):
- LLM (новый промпт + расширенный tool `create_ticket`, удалён `transfer_to_operator`)
- Bitrix-слой (резолвер enum-ID при старте, расширенный payload с UF_* и транскриптом)
- Session-слой (`caller_phone`, `enum_ids`, счётчик ошибок, `_fatal_fallback`)

**Tech Stack:** Python 3.10, FastAPI, pytest + pytest-asyncio + pytest-mock, httpx 0.27 (уже в проекте), existing Whisper/Silero/vLLM pipeline.

**Reference:** `docs/superpowers/specs/2026-04-21-lead-qualification-design.md`

---

## File Structure

### Create

| Файл | Ответственность |
|---|---|
| `test/__init__.py` | маркер пакета для pytest |
| `test/conftest.py` | общие pytest-фикстуры (mock httpx, sample payloads) |
| `test/test_phone_normalizer.py` | unit-тесты для `normalize_phone` (покрытие кириллических числительных, форматирования) |
| `test/test_bitrix_payload.py` | unit-тесты для чистых функций `_build_lead_payload` и `_format_comments` |
| `test/test_enum_resolver.py` | unit-тесты для `load_ai_quality_enum_ids` (mock httpx) |
| `test/test_llm_prompt.py` | unit-тесты для `LLMAgent._build_system_prompt` (caller_phone branching) |
| `pytest.ini` | pytest-конфиг (asyncio_mode, testpaths, добавить backend/ в pythonpath) |
| `docs/manual-qa-checklist.md` | чек-лист из § 5.3 спеки |

### Modify

| Файл | Что меняется |
|---|---|
| `requirements.txt` | +pytest, +pytest-asyncio, +pytest-mock, +respx (для мока httpx) |
| `backend/config.py` | +`FATAL_CONSECUTIVE_ERRORS: int` |
| `backend/services/bitrix.py` | новые `_format_comments`, `_build_lead_payload`, `load_ai_quality_enum_ids`; сигнатура `send_to_bitrix` расширена |
| `backend/engines/llm.py` | переписанный системный промпт, расширенная `create_ticket` schema, удалённый `transfer_to_operator`, обновлённый `_build_system_prompt` (caller_phone) |
| `backend/services/session.py` | `DialogSession.__init__` принимает `caller_phone` и `enum_ids`; `_consecutive_errors` счётчик; метод `_fatal_fallback`; exception-хендлеры вызывают fallback |
| `backend/main.py` | lifespan резолвит enum_ids; websocket_endpoint читает `caller_phone` из query_params |
| `backend/web/index.html` | JS читает `caller_phone` из URL и пробрасывает в WebSocket URL |
| `test/test_bitrix_live.py` | 4 новых сценария: hot_callback, cold_callback, operator_requested, fatal_fallback |

---

## Task 0: Set up pytest infrastructure

**Files:**
- Create: `pytest.ini`, `test/__init__.py`, `test/conftest.py`
- Modify: `requirements.txt`

- [ ] **Step 0.1: Добавить dev-зависимости в requirements.txt**

Открыть `requirements.txt`, добавить в конец:

```
# --- dev/test deps ---
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-mock==3.14.0
respx==0.21.1
```

`respx` — стандартный MockTransport для httpx 0.27.

- [ ] **Step 0.2: Установить локально**

```bash
pip install pytest==8.3.3 pytest-asyncio==0.24.0 pytest-mock==3.14.0 respx==0.21.1
```

Expected: установка без ошибок.

- [ ] **Step 0.3: Создать `pytest.ini`**

```ini
[pytest]
testpaths = test
python_files = test_*.py
asyncio_mode = auto
addopts = -v --tb=short
```

- [ ] **Step 0.4: Создать `test/__init__.py`** (пустой файл, нужен для pytest discovery)

```python
```

- [ ] **Step 0.5: Создать `test/conftest.py` с общими фикстурами**

```python
"""Общие pytest-фикстуры для тестов."""
import pytest


@pytest.fixture
def sample_chat_history() -> list[dict]:
    return [
        {"role": "assistant", "content": "Здравствуйте! Чем могу помочь?"},
        {"role": "user", "content": "Хочу на программирование после 11 класса"},
        {"role": "assistant", "content": "Отличный выбор. Записать на обратный звонок?"},
        {"role": "user", "content": "Да, Иван, 8 900 123 45 67"},
    ]


@pytest.fixture
def sample_ticket_data_callback() -> dict:
    return {
        "name": "Иван",
        "phone": "+79001234567",
        "intent": "Интересует программирование",
        "admission_year": "current",
        "request_type": "callback",
        "school_class": "11",
        "specialty": "программирование",
    }


@pytest.fixture
def sample_ticket_data_operator() -> dict:
    return {
        "name": "Пётр",
        "phone": "+79007654321",
        "intent": "Просил оператора",
        "admission_year": "next",
        "request_type": "operator_requested",
        "school_class": "9",
        "specialty": "дизайн",
    }


@pytest.fixture
def sample_ticket_data_fatal() -> dict:
    return {
        "name": "",
        "phone": "",
        "intent": "Сбой бота",
        "admission_year": None,
        "request_type": "fatal_fallback",
        "school_class": None,
        "specialty": None,
    }


@pytest.fixture
def enum_ids() -> dict[str, int]:
    return {"current": 173, "next": 175}
```

- [ ] **Step 0.6: Проверить pytest запускается**

```bash
pytest --collect-only
```

Expected: `collected 0 items` (ни одного теста пока нет, но pytest не падает).

- [ ] **Step 0.7: Commit**

```bash
git add requirements.txt pytest.ini test/__init__.py test/conftest.py
git commit -m "chore: set up pytest infrastructure

- pytest, pytest-asyncio, pytest-mock, respx в requirements
- pytest.ini с asyncio_mode=auto
- test/conftest.py с общими фикстурами для ticket_data, chat_history, enum_ids"
```

---

## Task 1: Add FATAL_CONSECUTIVE_ERRORS to config

**Files:**
- Modify: `backend/config.py`

- [ ] **Step 1.1: Добавить константу в `backend/config.py`**

В конец файла (после `BITRIX_WEBHOOK_URL`):

```python

# Fatal-fallback threshold
FATAL_CONSECUTIVE_ERRORS: int = int(os.getenv("FATAL_CONSECUTIVE_ERRORS", "3"))
```

- [ ] **Step 1.2: Проверить импортируется**

```bash
python -c "from backend.config import FATAL_CONSECUTIVE_ERRORS; print(FATAL_CONSECUTIVE_ERRORS)"
```

Expected: `3`

- [ ] **Step 1.3: Commit**

```bash
git add backend/config.py
git commit -m "feat: add FATAL_CONSECUTIVE_ERRORS config constant"
```

---

## Task 2: Create baseline tests for phone_normalizer

**Files:**
- Create: `test/test_phone_normalizer.py`

- [ ] **Step 2.1: Написать тест на цифровой формат**

Создать `test/test_phone_normalizer.py`:

```python
"""Тесты для normalize_phone: приводит любой формат к +7XXXXXXXXXX."""
import pytest

from backend.services.phone_normalizer import normalize_phone


class TestDigitFormats:
    def test_eleven_digits_starting_with_7(self):
        assert normalize_phone("79001234567") == "+79001234567"

    def test_eleven_digits_starting_with_8(self):
        assert normalize_phone("89001234567") == "+79001234567"

    def test_ten_digits_without_country_code(self):
        assert normalize_phone("9001234567") == "+79001234567"

    def test_formatted_with_plus(self):
        assert normalize_phone("+7 900 123 45 67") == "+79001234567"

    def test_formatted_with_dashes(self):
        assert normalize_phone("8-900-123-45-67") == "+79001234567"

    def test_formatted_with_parens(self):
        assert normalize_phone("8 (900) 123-45-67") == "+79001234567"


class TestRussianNumerals:
    def test_simple_numerals(self):
        assert normalize_phone(
            "восемь девятьсот один два три четыре пять шесть семь"
        ) in ("+79001234567",)  # зависит от реализации парсинга

    def test_mixed_digits_and_numerals(self):
        assert normalize_phone("восемь 900 123 45 67") == "+79001234567"


class TestInvalidInputs:
    def test_empty_string(self):
        assert normalize_phone("") is None

    def test_whitespace_only(self):
        assert normalize_phone("   ") is None

    def test_too_short(self):
        assert normalize_phone("123") is None

    def test_too_long(self):
        assert normalize_phone("123456789012345") is None

    def test_wrong_country_code(self):
        assert normalize_phone("99001234567") is None  # 11 цифр, но начинается с 9
```

- [ ] **Step 2.2: Запустить, убедиться что часть тестов проходит, часть может падать**

```bash
pytest test/test_phone_normalizer.py -v
```

Expected: большинство тестов проходят (т.к. normalize_phone уже реализован). Возможны единичные провалы на тестах русских числительных — зависит от точности алгоритма. Если падают только они — оставить как известные ограничения и пометить `@pytest.mark.xfail(reason="partial numerals parsing")`.

- [ ] **Step 2.3: Commit**

```bash
git add test/test_phone_normalizer.py
git commit -m "test: baseline unit tests for phone_normalizer"
```

---

## Task 3: Implement `_format_comments` helper (TDD)

**Files:**
- Create: `test/test_bitrix_payload.py`
- Modify: `backend/services/bitrix.py`

- [ ] **Step 3.1: Написать failing-тест в `test/test_bitrix_payload.py`**

```python
"""Тесты чистых функций bitrix.py: _format_comments, _build_lead_payload."""
import pytest

from backend.services.bitrix import _format_comments


class TestFormatComments:
    def test_callback_request_prefix(self, sample_chat_history, sample_ticket_data_callback):
        text = _format_comments(sample_ticket_data_callback, sample_chat_history)
        assert text.startswith("[Обратный звонок]")

    def test_operator_requested_prefix(self, sample_chat_history, sample_ticket_data_operator):
        text = _format_comments(sample_ticket_data_operator, sample_chat_history)
        assert text.startswith("[Запрошен оператор]")

    def test_fatal_fallback_prefix(self, sample_chat_history, sample_ticket_data_fatal):
        text = _format_comments(sample_ticket_data_fatal, sample_chat_history)
        assert text.startswith("[СРОЧНО: технический сбой]")

    def test_intent_included(self, sample_chat_history, sample_ticket_data_callback):
        text = _format_comments(sample_ticket_data_callback, sample_chat_history)
        assert "Интересует программирование" in text

    def test_transcript_with_role_labels(self, sample_chat_history, sample_ticket_data_callback):
        text = _format_comments(sample_ticket_data_callback, sample_chat_history)
        assert "Оператор:" in text
        assert "Абитуриент:" in text
        assert "Хочу на программирование после 11 класса" in text

    def test_transcript_separator(self, sample_chat_history, sample_ticket_data_callback):
        text = _format_comments(sample_ticket_data_callback, sample_chat_history)
        assert "—— Транскрипт ——" in text
```

- [ ] **Step 3.2: Запустить, убедиться что упало**

```bash
pytest test/test_bitrix_payload.py::TestFormatComments -v
```

Expected: `ImportError: cannot import name '_format_comments'` — функция ещё не существует.

- [ ] **Step 3.3: Реализовать `_format_comments` в `backend/services/bitrix.py`**

Добавить (перед `send_to_bitrix`):

```python
_REQUEST_TYPE_PREFIX: dict[str, str] = {
    "callback": "[Обратный звонок]",
    "operator_requested": "[Запрошен оператор]",
    "fatal_fallback": "[СРОЧНО: технический сбой]",
}


def _format_comments(
    ticket_data: dict[str, Any], chat_history: list[dict[str, str]]
) -> str:
    """Формирует текст для поля COMMENTS лида в Bitrix.

    Структура: [префикс по request_type] -> intent -> разделитель -> транскрипт.
    """
    request_type = ticket_data.get("request_type", "callback")
    prefix = _REQUEST_TYPE_PREFIX.get(request_type, "[Обратный звонок]")
    intent = ticket_data.get("intent", "")

    lines: list[str] = [prefix]
    if intent:
        lines.append(intent)
    lines.append("")
    lines.append("—— Транскрипт ——")
    for msg in chat_history:
        role_label = "Оператор" if msg["role"] == "assistant" else "Абитуриент"
        lines.append(f"{role_label}: {msg['content']}")

    return "\n".join(lines)
```

- [ ] **Step 3.4: Запустить, убедиться что тесты проходят**

```bash
pytest test/test_bitrix_payload.py::TestFormatComments -v
```

Expected: 6 тестов passed.

- [ ] **Step 3.5: Commit**

```bash
git add test/test_bitrix_payload.py backend/services/bitrix.py
git commit -m "feat(bitrix): add _format_comments helper for COMMENTS field

Формирует текст с префиксом по request_type, intent и форматированным
транскриптом диалога с подписями Оператор: / Абитуриент:"
```

---

## Task 4: Implement `_build_lead_payload` helper (TDD)

**Files:**
- Modify: `test/test_bitrix_payload.py`, `backend/services/bitrix.py`

- [ ] **Step 4.1: Добавить тесты в `test/test_bitrix_payload.py`**

Дописать в конец файла:

```python
from backend.services.bitrix import _build_lead_payload


class TestBuildLeadPayload:
    def test_callback_basic_structure(
        self, sample_ticket_data_callback, sample_chat_history, enum_ids
    ):
        payload = _build_lead_payload(
            sample_ticket_data_callback,
            sample_chat_history,
            enum_ids,
            phone="+79001234567",
        )
        fields = payload["fields"]
        assert fields["NAME"] == "Иван"
        assert fields["SOURCE_ID"] == "CALLBACK"
        assert fields["PHONE"] == [
            {"VALUE": "+79001234567", "VALUE_TYPE": "MOBILE"}
        ]
        assert fields["UF_CRM_AI_QUALITY"] == 173
        assert fields["UF_CRM_KAKOIKLASSVIZ"] == "11"
        assert fields["UF_CRM_KAKAYASPETSIA"] == "программирование"
        assert "[Обратный звонок]" in fields["COMMENTS"]

    def test_operator_requested_payload(
        self, sample_ticket_data_operator, sample_chat_history, enum_ids
    ):
        payload = _build_lead_payload(
            sample_ticket_data_operator,
            sample_chat_history,
            enum_ids,
            phone="+79007654321",
        )
        fields = payload["fields"]
        assert fields["UF_CRM_AI_QUALITY"] == 175  # next = Некачественный
        assert "[Запрошен оператор]" in fields["COMMENTS"]

    def test_fatal_fallback_no_quality_field(
        self, sample_ticket_data_fatal, sample_chat_history, enum_ids
    ):
        payload = _build_lead_payload(
            sample_ticket_data_fatal,
            sample_chat_history,
            enum_ids,
            phone="",
        )
        fields = payload["fields"]
        assert "UF_CRM_AI_QUALITY" not in fields  # при fatal не ставим
        assert fields["TITLE"].startswith("СРОЧНО")
        assert "[СРОЧНО" in fields["COMMENTS"]

    def test_title_truncated_from_intent(
        self, sample_chat_history, enum_ids
    ):
        data = {
            "name": "Тест",
            "phone": "+79001234567",
            "intent": "Очень длинный интент " * 20,
            "admission_year": "current",
            "request_type": "callback",
        }
        payload = _build_lead_payload(data, sample_chat_history, enum_ids, "+79001234567")
        assert len(payload["fields"]["TITLE"]) < 120

    def test_optional_school_class_omitted_when_absent(
        self, sample_chat_history, enum_ids
    ):
        data = {
            "name": "Тест",
            "phone": "+79001234567",
            "intent": "intent",
            "admission_year": "current",
            "request_type": "callback",
        }
        payload = _build_lead_payload(data, sample_chat_history, enum_ids, "+79001234567")
        assert "UF_CRM_KAKOIKLASSVIZ" not in payload["fields"]
        assert "UF_CRM_KAKAYASPETSIA" not in payload["fields"]

    def test_admission_year_none_omits_quality_field(
        self, sample_chat_history, enum_ids
    ):
        data = {
            "name": "Тест",
            "phone": "+79001234567",
            "intent": "intent",
            "admission_year": None,
            "request_type": "callback",
        }
        payload = _build_lead_payload(data, sample_chat_history, enum_ids, "+79001234567")
        assert "UF_CRM_AI_QUALITY" not in payload["fields"]
```

- [ ] **Step 4.2: Запустить, убедиться упало**

```bash
pytest test/test_bitrix_payload.py::TestBuildLeadPayload -v
```

Expected: `ImportError: cannot import name '_build_lead_payload'`.

- [ ] **Step 4.3: Реализовать `_build_lead_payload` в `backend/services/bitrix.py`**

Добавить после `_format_comments`:

```python
_MAX_TITLE_LEN = 100


def _build_lead_payload(
    ticket_data: dict[str, Any],
    chat_history: list[dict[str, str]],
    enum_ids: dict[str, int],
    phone: str,
) -> dict[str, Any]:
    """Собирает JSON-payload для crm.lead.add.

    phone: уже нормализованный телефон (или пустая строка для fatal_fallback).
    enum_ids: {"current": <id_Качественный>, "next": <id_Некачественный>}.
    """
    request_type = ticket_data.get("request_type", "callback")
    intent = ticket_data.get("intent", "без темы")

    if request_type == "fatal_fallback":
        title = "СРОЧНО: сбой бота — оператор срочно перезвонить"
    else:
        title = f"Звонок: {intent}"
    title = title[:_MAX_TITLE_LEN]

    fields: dict[str, Any] = {
        "TITLE": title,
        "NAME": ticket_data.get("name", ""),
        "SOURCE_ID": "CALLBACK",
        "COMMENTS": _format_comments(ticket_data, chat_history),
    }

    if phone:
        fields["PHONE"] = [{"VALUE": phone, "VALUE_TYPE": "MOBILE"}]

    admission_year = ticket_data.get("admission_year")
    if request_type != "fatal_fallback" and admission_year in ("current", "next"):
        fields["UF_CRM_AI_QUALITY"] = enum_ids[admission_year]

    school_class = ticket_data.get("school_class")
    if school_class:
        fields["UF_CRM_KAKOIKLASSVIZ"] = school_class

    specialty = ticket_data.get("specialty")
    if specialty:
        fields["UF_CRM_KAKAYASPETSIA"] = specialty

    return {"fields": fields}
```

- [ ] **Step 4.4: Запустить тесты**

```bash
pytest test/test_bitrix_payload.py -v
```

Expected: все 12 тестов passed (6 из Task 3 + 6 новых).

- [ ] **Step 4.5: Commit**

```bash
git add test/test_bitrix_payload.py backend/services/bitrix.py
git commit -m "feat(bitrix): add _build_lead_payload with UF_* mapping

Чистая функция собирает payload для crm.lead.add. Покрывает:
- callback/operator_requested/fatal_fallback через _format_comments
- UF_CRM_AI_QUALITY резолвится из enum_ids[admission_year]
- опциональные UF_CRM_KAKOIKLASSVIZ и UF_CRM_KAKAYASPETSIA
- fatal_fallback не заполняет UF_CRM_AI_QUALITY (явный сигнал что не квалифицировано)
- TITLE обрезается до 100 символов"
```

---

## Task 5: Implement `load_ai_quality_enum_ids` (TDD with respx)

**Files:**
- Create: `test/test_enum_resolver.py`
- Modify: `backend/services/bitrix.py`

- [ ] **Step 5.1: Написать failing-тест в `test/test_enum_resolver.py`**

```python
"""Тесты резолвера enum ID для UF_CRM_AI_QUALITY."""
import pytest
import respx
from httpx import Response

from backend.services.bitrix import load_ai_quality_enum_ids


WEBHOOK = "https://example.bitrix24.ru/rest/1/token"


@respx.mock
@pytest.mark.asyncio
async def test_resolver_success():
    respx.post(f"{WEBHOOK}/crm.lead.userfield.list.json").mock(
        return_value=Response(200, json={
            "result": [{
                "ID": "415",
                "FIELD_NAME": "UF_CRM_AI_QUALITY",
                "USER_TYPE_ID": "enumeration",
                "LIST": [
                    {"ID": "173", "VALUE": "Качественный"},
                    {"ID": "175", "VALUE": "Некачественный"},
                ],
            }],
        })
    )
    ids = await load_ai_quality_enum_ids(WEBHOOK)
    assert ids == {"current": 173, "next": 175}


@respx.mock
@pytest.mark.asyncio
async def test_resolver_field_not_found():
    respx.post(f"{WEBHOOK}/crm.lead.userfield.list.json").mock(
        return_value=Response(200, json={"result": []})
    )
    with pytest.raises(RuntimeError, match="UF_CRM_AI_QUALITY"):
        await load_ai_quality_enum_ids(WEBHOOK)


@respx.mock
@pytest.mark.asyncio
async def test_resolver_missing_value():
    respx.post(f"{WEBHOOK}/crm.lead.userfield.list.json").mock(
        return_value=Response(200, json={
            "result": [{
                "FIELD_NAME": "UF_CRM_AI_QUALITY",
                "LIST": [
                    {"ID": "173", "VALUE": "Качественный"},
                    # нет "Некачественный"
                ],
            }],
        })
    )
    with pytest.raises(RuntimeError, match="Некачественный"):
        await load_ai_quality_enum_ids(WEBHOOK)


@respx.mock
@pytest.mark.asyncio
async def test_resolver_http_error():
    respx.post(f"{WEBHOOK}/crm.lead.userfield.list.json").mock(
        return_value=Response(500, text="Internal Server Error")
    )
    with pytest.raises(RuntimeError):
        await load_ai_quality_enum_ids(WEBHOOK)
```

- [ ] **Step 5.2: Запустить, убедиться упало**

```bash
pytest test/test_enum_resolver.py -v
```

Expected: ImportError на `load_ai_quality_enum_ids`.

- [ ] **Step 5.3: Реализовать `load_ai_quality_enum_ids` в `backend/services/bitrix.py`**

Добавить в верхнюю часть файла (рядом с другими функциями):

```python
async def load_ai_quality_enum_ids(webhook_url: str) -> dict[str, int]:
    """Резолвит enum-ID значений UF_CRM_AI_QUALITY при старте приложения.

    Identify by VALUE strict match: "Качественный" → "current", "Некачественный" → "next".
    XML_ID не используем — он не сохраняется через REST Bitrix.

    Returns:
        {"current": <id_Качественный>, "next": <id_Некачественный>}

    Raises:
        RuntimeError: если поле UF_CRM_AI_QUALITY не существует, либо не найдено
            одно из обязательных значений, либо webhook недоступен.
    """
    url = f"{webhook_url.rstrip('/')}/crm.lead.userfield.list.json"
    payload = {"filter": {"FIELD_NAME": "UF_CRM_AI_QUALITY"}}

    try:
        async with httpx.AsyncClient(timeout=BASE_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"Не удалось достучаться до Bitrix webhook при загрузке enum IDs: {e}"
        ) from e

    data = response.json()
    if "error" in data:
        raise RuntimeError(
            f"Bitrix вернул ошибку {data['error']}: {data.get('error_description', '')}"
        )

    fields = data.get("result", [])
    if not fields:
        raise RuntimeError(
            "Поле UF_CRM_AI_QUALITY не найдено в Bitrix. "
            "Проверь UI: CRM → Настройки → Настройки форм и отчётов → "
            "Пользовательские поля → Лиды."
        )

    items = fields[0].get("LIST", [])
    mapping: dict[str, int] = {}
    for item in items:
        value = item.get("VALUE", "").strip()
        item_id = int(item["ID"])
        if value == "Качественный":
            mapping["current"] = item_id
        elif value == "Некачественный":
            mapping["next"] = item_id

    missing = [k for k in ("current", "next") if k not in mapping]
    if missing:
        expected = {"current": "Качественный", "next": "Некачественный"}
        missing_names = ", ".join(expected[m] for m in missing)
        raise RuntimeError(
            f"В поле UF_CRM_AI_QUALITY не найдены значения: {missing_names}. "
            "Проверь UI: CRM → Настройки → Пользовательские поля → "
            "AI: Качество лида → список значений."
        )

    return mapping
```

- [ ] **Step 5.4: Запустить тесты**

```bash
pytest test/test_enum_resolver.py -v
```

Expected: 4 теста passed.

- [ ] **Step 5.5: Commit**

```bash
git add test/test_enum_resolver.py backend/services/bitrix.py
git commit -m "feat(bitrix): add load_ai_quality_enum_ids startup resolver

Резолвит ID значений enum-поля UF_CRM_AI_QUALITY по строковому VALUE
(XML_ID не работает через REST Bitrix — см. spec § 10).

Fail-fast: если поле или значения не найдены — RuntimeError с подсказкой
где исправить в UI."
```

---

## Task 6: Refactor `send_to_bitrix` signature

**Files:**
- Modify: `backend/services/bitrix.py`

- [ ] **Step 6.1: Обновить сигнатуру и логику `send_to_bitrix`**

Заменить текущую функцию `send_to_bitrix` на:

```python
async def send_to_bitrix(
    ticket_data: dict[str, Any],
    chat_history: list[dict[str, str]],
    enum_ids: dict[str, int] | None,
    caller_phone: str | None = None,
) -> dict[str, Any] | None:
    """Создает лид в Bitrix24 CRM через REST webhook.

    ticket_data: результат tool_call create_ticket от LLM.
    chat_history: полная история диалога для транскрипта в COMMENTS.
    enum_ids: резолв UF_CRM_AI_QUALITY значений (см. load_ai_quality_enum_ids).
        None если webhook не был задан при старте — тогда лид не отправляется.
    caller_phone: телефон из АТС / query-param. Используется если
        ticket_data["phone"] пустой.

    Возвращает ответ Bitrix24 при успехе, None при ошибке (включая отсутствие
    webhook или невалидный телефон). Никогда не бросает исключений.
    """
    if not BITRIX_WEBHOOK_URL or enum_ids is None:
        logger.warning("BITRIX_WEBHOOK_URL не задан, лид не отправлен: %s", ticket_data)
        return None

    request_type = ticket_data.get("request_type", "callback")

    # Phone: сначала из ticket_data, если нет — caller_phone. fatal_fallback
    # может идти с пустым телефоном — не блокирует создание лида.
    raw_phone = ticket_data.get("phone") or caller_phone or ""
    phone = normalize_phone(raw_phone) if raw_phone else ""

    if not phone and request_type != "fatal_fallback":
        logger.warning(
            "Невалидный/пустой телефон (%r), лид не создан. ticket_data=%s",
            raw_phone, ticket_data,
        )
        return None

    payload = _build_lead_payload(ticket_data, chat_history, enum_ids, phone)
    url = f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.add.json"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=BASE_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                lead_id = result.get("result")
                logger.info(
                    "Лид создан в Bitrix24, ID=%s, type=%s", lead_id, request_type,
                )
                return result

        except httpx.TimeoutException:
            logger.warning(
                "Таймаут Bitrix24 (попытка %d/%d)", attempt, MAX_RETRIES,
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "Bitrix24 вернул ошибку %s (попытка %d/%d): %s",
                e.response.status_code, attempt, MAX_RETRIES,
                e.response.text[:200],
            )
        except Exception as e:
            logger.error(
                "Ошибка отправки в Bitrix24 (попытка %d/%d): %s",
                attempt, MAX_RETRIES, e,
            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(2 ** attempt)

    logger.error(
        "Не удалось создать лид после %d попыток. payload=%s",
        MAX_RETRIES, payload,
    )
    return None
```

Импорт `asyncio` должен быть вверху модуля — проверить. Если нет, добавить `import asyncio` рядом с `import logging`.

- [ ] **Step 6.2: Удалить дублирующийся блок `import asyncio` внутри функции** (был в оригинале как `if attempt < MAX_RETRIES: import asyncio; await asyncio.sleep(...)`).

- [ ] **Step 6.3: Убедиться что существующие тесты не сломались**

```bash
pytest test/test_bitrix_payload.py test/test_enum_resolver.py -v
```

Expected: все проходят (они не затрагивают send_to_bitrix напрямую).

- [ ] **Step 6.4: Commit**

```bash
git add backend/services/bitrix.py
git commit -m "refactor(bitrix): extend send_to_bitrix signature

Новые параметры: chat_history (для транскрипта), enum_ids (для UF_AI_QUALITY),
caller_phone (fallback если phone пустой в ticket_data).

Логика:
- payload строится через _build_lead_payload (pure function)
- Если phone не резолвится и это не fatal_fallback — лид не отправляется
- Для fatal_fallback допускается пустой телефон"
```

---

## Task 7: Update LLM system prompt (operator logic + year + caller_phone block)

**Files:**
- Create: `test/test_llm_prompt.py`
- Modify: `backend/engines/llm.py`

- [ ] **Step 7.1: Написать failing-тесты в `test/test_llm_prompt.py`**

```python
"""Тесты рендеринга системного промпта."""
import pytest

from backend.engines.llm import LLMAgent


@pytest.fixture
def agent_no_retriever():
    class DummyClient:
        pass
    return LLMAgent(DummyClient(), retriever=None)


class TestSystemPrompt:
    def test_no_caller_phone_means_no_phone_block(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=1, caller_phone=None,
        )
        assert "ТЕЛЕФОН АБОНЕНТА" not in prompt

    def test_caller_phone_inserts_block(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=1, caller_phone="+79001234567",
        )
        assert "ТЕЛЕФОН АБОНЕНТА" in prompt
        assert "+79001234567" in prompt
        assert "не переспрашивай" in prompt.lower()

    def test_operator_block_present(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=1, caller_phone=None,
        )
        assert "ПЕРЕВОД НА ОПЕРАТОРА" in prompt
        assert "предложи" in prompt.lower() or "могу помочь" in prompt.lower()

    def test_year_instruction_present(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=3, caller_phone=None,
        )
        assert "год" in prompt.lower()
        assert "admission_year" in prompt or "в этом году" in prompt.lower()

    def test_discovery_stage_for_first_turns(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=1, caller_phone=None,
        )
        assert "Знакомство" in prompt or "Discovery" in prompt

    def test_inform_stage_for_later_turns(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=5, caller_phone=None,
        )
        assert "Информирование" in prompt or "Inform" in prompt
```

- [ ] **Step 7.2: Запустить, убедиться упало**

```bash
pytest test/test_llm_prompt.py -v
```

Expected: тесты падают т.к. `_build_system_prompt` не принимает `caller_phone` + промпт не содержит новых блоков.

- [ ] **Step 7.3: Переписать SYSTEM_PROMPT в `backend/engines/llm.py`**

Заменить текущий `SYSTEM_PROMPT` на новую версию:

```python
SYSTEM_PROMPT = """\
Ты — Ксения, сотрудник приёмной комиссии колледжа КЭСИ. Ты отвечаешь на телефонные звонки.
Ты консультируешь абитуриентов по вопросам поступления, специальностей, стоимости и документов.
Ты НЕ преподаватель, НЕ репетитор, НЕ готовишь к экзаменам. Ты работаешь только в приёмной комиссии.
Если вопрос не касается поступления в колледж КЭСИ — вежливо скажи, что это не в твоей компетенции.

═══ ТВОЯ ЦЕЛЬ ═══
Каждый разговор должен закончиться конкретным результатом:
• Абитуриент получил полный ответ на свой вопрос, ИЛИ
• Абитуриент записан на обратный звонок (create_ticket с request_type='callback'), ИЛИ
• Абитуриент настоял на операторе — записан (create_ticket с request_type='operator_requested').

═══ КАК ВЕСТИ РАЗГОВОР ═══
1. ИНИЦИАТИВА — ты ведёшь диалог. После каждого ответа задай уточняющий вопрос.
2. КРАТКОСТЬ — твои ответы озвучиваются. 2–3 предложения + вопрос.
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
```

- [ ] **Step 7.4: Обновить `_build_system_prompt` для поддержки `caller_phone`**

Найти метод `_build_system_prompt` и заменить на:

```python
_CALLER_PHONE_TEMPLATE = (
    "\n\n═══ ТЕЛЕФОН АБОНЕНТА ═══\n"
    "{phone}\n"
    "Телефон уже известен из системы, не переспрашивай его."
)


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
```

- [ ] **Step 7.5: Запустить тесты промпта**

```bash
pytest test/test_llm_prompt.py -v
```

Expected: 6 passed.

- [ ] **Step 7.6: Commit**

```bash
git add test/test_llm_prompt.py backend/engines/llm.py
git commit -m "feat(llm): new system prompt + caller_phone block

- Блок 'Перевод на оператора': теперь сначала предложение помощи, лид
  создаётся только при настойчивости или явном согласии на callback.
- Блок 'Сбор контактов': обязательный admission_year с адаптивностью
  (не переспрашивать если ясно из контекста).
- _build_system_prompt принимает caller_phone. Если задан — вставляет
  блок 'ТЕЛЕФОН АБОНЕНТА' чтобы LLM не переспрашивала."
```

---

## Task 8: Update `create_ticket` tool schema (remove transfer_to_operator)

**Files:**
- Modify: `backend/engines/llm.py`

- [ ] **Step 8.1: Обновить TOOLS**

Заменить текущий `TOOLS` list в `backend/engines/llm.py`:

```python
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
```

- [ ] **Step 8.2: Удалить константы `TRANSFER_REPLY` и все упоминания `transfer_to_operator`**

В файле `backend/engines/llm.py`:
- Удалить строку `TRANSFER_REPLY = "..."`
- В методе `get_response` удалить ветку обработки `tool_call.function.name == "transfer_to_operator"`

- [ ] **Step 8.3: Запустить существующие тесты чтобы ничего не сломалось**

```bash
pytest test/ -v
```

Expected: все предыдущие тесты passed.

- [ ] **Step 8.4: Commit**

```bash
git add backend/engines/llm.py
git commit -m "feat(llm): extend create_ticket schema, remove transfer_to_operator

create_ticket теперь требует admission_year и request_type.
school_class и specialty — опциональные.
transfer_to_operator больше не tool — его работу делает create_ticket
с request_type='operator_requested'."
```

---

## Task 9: Extend `DialogSession` constructor (caller_phone, enum_ids, counter)

**Files:**
- Modify: `backend/services/session.py`

- [ ] **Step 9.1: Обновить `__init__` в `backend/services/session.py`**

Заменить текущий `__init__`:

```python
def __init__(
    self,
    websocket: WebSocket,
    vad: VADEngine,
    stt: STTEngine,
    tts: TTSEngine,
    llm: LLMAgent,
    caller_phone: str | None = None,
    enum_ids: dict[str, int] | None = None,
) -> None:
    self.websocket = websocket
    self.vad = vad
    self.stt = stt
    self.tts = tts
    self.llm = llm
    self.caller_phone = caller_phone
    self.enum_ids = enum_ids

    self.session_id: str = uuid.uuid4().hex[:12]
    self.chat_history: list[dict[str, str]] = []
    self.is_processing: bool = False
    self._start_time: float = time.time()
    self._consecutive_errors: int = 0

    self._incoming_buffer = bytearray()
    self._speech_buffer = bytearray()
    self._is_speaking = False
    self._silence_start: float | None = None
```

- [ ] **Step 9.2: Обновить вызов `LLMAgent.get_response` в `_process_turn`**

Найти в `_process_turn`:

```python
reply_text, ticket_data = await self.llm.get_response(self.chat_history)
```

**Контекст:** `LLMAgent.get_response` сейчас принимает только chat_history. После Task 7 мы изменили `_build_system_prompt` чтобы принимать `caller_phone`. Сейчас нужно:

(a) Обновить `LLMAgent.get_response` чтобы принимал `caller_phone` параметр и передавал в `_build_system_prompt`.
(b) Session передаёт `self.caller_phone`.

В `backend/engines/llm.py` метод `get_response`:

Заменить сигнатуру и вызов `_build_system_prompt`:

```python
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
        # ... остальное без изменений
```

В `session.py` обновить вызов:

```python
reply_text, ticket_data = await self.llm.get_response(
    self.chat_history, caller_phone=self.caller_phone,
)
```

- [ ] **Step 9.3: Обновить вызов `send_to_bitrix` в `_process_turn`**

Найти блок `if ticket_data:` в `_process_turn`. Заменить:

```python
# 4. Bitrix24 (с проверкой телефона)
if ticket_data:
    phone_for_check = ticket_data.get("phone") or self.caller_phone or ""
    phone = normalize_phone(phone_for_check) if phone_for_check else ""
    if phone or ticket_data.get("request_type") == "fatal_fallback":
        bitrix_task = asyncio.create_task(
            send_to_bitrix(
                ticket_data,
                self.chat_history,
                self.enum_ids,
                self.caller_phone,
            )
        )
        bitrix_task.add_done_callback(self._on_task_done)
    else:
        # Телефон невалиден — переспрашиваем
        retry_text = (
            "Извините, я не разобрала номер телефона. "
            "Повторите, пожалуйста, по цифрам."
        )
        logger.warning("Невалидный телефон: %s", ticket_data.get("phone"))
        self.chat_history.append({"role": "assistant", "content": retry_text})
        await self._send_text("assistant", retry_text)
        audio = await self.tts.synthesize(retry_text)
        await self.websocket.send_bytes(audio)
```

- [ ] **Step 9.4: Reset counter on successful LLM reply**

В `_process_turn`, сразу после успешного `reply_text, ticket_data = await self.llm.get_response(...)`, добавить:

```python
self._consecutive_errors = 0
```

- [ ] **Step 9.5: Запустить все unit-тесты**

```bash
pytest test/ -v --ignore=test/test_bitrix_live.py
```

Expected: все unit-тесты passed.

- [ ] **Step 9.6: Commit**

```bash
git add backend/services/session.py backend/engines/llm.py
git commit -m "feat(session): wire caller_phone + enum_ids through DialogSession

- __init__ принимает caller_phone и enum_ids
- _consecutive_errors счётчик для fatal-fallback триггера
- LLMAgent.get_response пробрасывает caller_phone в _build_system_prompt
- _process_turn резетит счётчик при успешной реплике и использует
  self.caller_phone как fallback если ticket_data.phone пустой"
```

---

## Task 10: Implement `_fatal_fallback` method

**Files:**
- Modify: `backend/services/session.py`

- [ ] **Step 10.1: Добавить константу в начало файла**

В `backend/services/session.py` обновить импорты, добавить:

```python
from backend.config import CHUNK_BYTES, SAMPLE_RATE, SILENCE_DURATION, FATAL_CONSECUTIVE_ERRORS
```

- [ ] **Step 10.2: Добавить метод `_fatal_fallback` в `DialogSession`**

Добавить перед методом `save_log`:

```python
_FATAL_APOLOGY = (
    "Извините, у меня технические сложности. "
    "Сейчас зафиксирую вашу заявку — оператор обязательно вам перезвонит."
)
_FATAL_PHONE_PROMPT = (
    "Продиктуйте, пожалуйста, номер телефона для связи."
)
_FATAL_GOODBYE = "До свидания."


async def _fatal_fallback(self, reason: str) -> None:
    """Последний резерв: при любой неисправности бота создаём срочный лид
    и вежливо закрываем звонок. В будущей итерации с Asterisk первым шагом
    будет реальная попытка SIP-transfer."""
    logger.error("FATAL_FALLBACK triggered: %s", reason)

    try:
        await self._speak_safe(self._FATAL_APOLOGY)

        phone = self.caller_phone or ""
        if not phone:
            await self._speak_safe(self._FATAL_PHONE_PROMPT)
            phone = await self._collect_phone_best_effort()

        name = self._extract_name_from_history()

        ticket_data = {
            "name": name,
            "phone": phone,
            "intent": f"СРОЧНО — технический сбой бота: {reason}",
            "admission_year": None,
            "request_type": "fatal_fallback",
        }

        await send_to_bitrix(
            ticket_data,
            self.chat_history,
            self.enum_ids,
            self.caller_phone,
        )

        await self._speak_safe(self._FATAL_GOODBYE)
    except Exception as exc:
        logger.error("Ошибка в _fatal_fallback: %s", exc, exc_info=True)
    finally:
        try:
            await self.websocket.close()
        except Exception:
            pass


async def _speak_safe(self, text: str) -> None:
    """TTS + WebSocket send с подавлением ошибок (мы уже в fatal-режиме)."""
    try:
        self.chat_history.append({"role": "assistant", "content": text})
        await self._send_text("assistant", text)
        audio = await self.tts.synthesize(text)
        await self.websocket.send_bytes(audio)
    except Exception as exc:
        logger.warning("Не удалось озвучить в fatal-режиме: %s", exc)


async def _collect_phone_best_effort(self, timeout_sec: float = 10.0) -> str:
    """Одна попытка получить телефон через одну короткую речевую реплику.
    Если не получилось — возвращает пустую строку."""
    import asyncio as _asyncio
    try:
        # Ждём одну VAD-активность; если абонент молчит — сдаёмся
        self._speech_buffer.clear()
        self._is_speaking = False
        self._silence_start = None
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                msg = await _asyncio.wait_for(
                    self.websocket.receive(), timeout=timeout_sec,
                )
            except _asyncio.TimeoutError:
                break
            data = msg.get("bytes", b"")
            if not data:
                continue
            self._incoming_buffer.extend(data)
            while len(self._incoming_buffer) >= CHUNK_BYTES:
                chunk = bytes(self._incoming_buffer[:CHUNK_BYTES])
                del self._incoming_buffer[:CHUNK_BYTES]
                has_speech = await self.vad.is_speech(chunk)
                if has_speech:
                    self._speech_buffer.extend(chunk)
                    self._is_speaking = True
                    self._silence_start = None
                elif self._is_speaking:
                    if self._silence_start is None:
                        self._silence_start = time.time()
                    elif time.time() - self._silence_start >= SILENCE_DURATION:
                        audio = bytes(self._speech_buffer)
                        text = await self.stt.transcribe(audio)
                        phone = normalize_phone(text or "")
                        return phone or ""
        return ""
    except Exception as exc:
        logger.warning("Не удалось собрать телефон best-effort: %s", exc)
        return ""


def _extract_name_from_history(self) -> str:
    """Best-effort: ищет в user-репликах одиночное слово похожее на имя.
    Если не нашли — возвращает пустую строку."""
    for msg in self.chat_history:
        if msg["role"] != "user":
            continue
        words = msg["content"].strip().split()
        if len(words) == 1 and words[0][0].isupper():
            return words[0]
    return ""
```

- [ ] **Step 10.3: Импортировать `send_to_bitrix` и `normalize_phone` в session.py** (они уже импортируются — проверить)

- [ ] **Step 10.4: Проверить синтаксис**

```bash
python -c "from backend.services.session import DialogSession; print('ok')"
```

Expected: `ok`.

- [ ] **Step 10.5: Commit**

```bash
git add backend/services/session.py
git commit -m "feat(session): add _fatal_fallback method

Последний резерв при технических сбоях бота. Шаги:
1. Apology TTS: 'извините, технические сложности, зафиксирую заявку'
2. Если caller_phone отсутствует — одна best-effort попытка собрать
   голосом (timeout 10 сек)
3. Extract name from history best-effort
4. send_to_bitrix с request_type='fatal_fallback' (в payload идёт без
   UF_CRM_AI_QUALITY, TITLE = 'СРОЧНО: сбой бота')
5. Goodbye + close WebSocket

Все операции wrapped в try/except — в fatal-режиме нельзя падать дальше."
```

---

## Task 11: Wire fatal_fallback into exception handlers

**Files:**
- Modify: `backend/services/session.py`

- [ ] **Step 11.1: Обновить `run()` для вызова fatal_fallback**

Найти метод `run` в `DialogSession`. Обернуть тело цикла в try/except:

```python
async def run(self) -> None:
    """Основной цикл диалога: приветствие -> прием аудио -> обработка."""
    try:
        await self._send_greeting()
        chunks_received = 0

        while True:
            msg = await self.websocket.receive()

            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "end_of_speech":
                        await self._flush_speech()
                except (json.JSONDecodeError, KeyError):
                    pass
                continue

            data = msg.get("bytes", b"")
            if not data:
                continue

            self._incoming_buffer.extend(data)
            chunks_received += 1
            if chunks_received == 1:
                logger.info("Первые данные от клиента: %d байт", len(data))

            while len(self._incoming_buffer) >= CHUNK_BYTES:
                chunk = bytes(self._incoming_buffer[:CHUNK_BYTES])
                del self._incoming_buffer[:CHUNK_BYTES]
                await self._handle_chunk(chunk)
    except WebSocketDisconnect:
        raise  # прокидываем — websocket_endpoint обработает
    except Exception as exc:
        await self._fatal_fallback(f"unhandled in run: {exc}")
        raise
```

Убедиться что `from fastapi import WebSocketDisconnect` уже импортирован наверху файла. Если нет — добавить:

```python
from starlette.websockets import WebSocketDisconnect
```

- [ ] **Step 11.2: Обновить `_process_turn` для инкремента счётчика и триггера**

Найти блок try/finally в `_process_turn`. Обновить:

```python
async def _process_turn(self, audio_bytes: bytes) -> None:
    self.is_processing = True
    try:
        t0 = time.time()

        # 1. STT
        user_text = await self.stt.transcribe(audio_bytes)
        t_stt = time.time()

        if not user_text:
            logger.info("STT: пустая транскрипция, пропуск")
            await self._send_text("system", "")
            return

        logger.info("Пользователь: %s", user_text)
        self.chat_history.append({"role": "user", "content": user_text})
        await self._send_text("user", user_text)

        # 2. LLM
        try:
            reply_text, ticket_data = await self.llm.get_response(
                self.chat_history, caller_phone=self.caller_phone,
            )
            self._consecutive_errors = 0
        except Exception as exc:
            self._consecutive_errors += 1
            logger.error("LLM error #%d: %s", self._consecutive_errors, exc)
            if self._consecutive_errors >= FATAL_CONSECUTIVE_ERRORS:
                await self._fatal_fallback(
                    f"LLM failed {self._consecutive_errors} consecutive times"
                )
                return
            reply_text = "Извините, у меня небольшая заминка со связью."
            ticket_data = None

        t_llm = time.time()

        # ... остальное без изменений (TTS, latency metrics, Bitrix)
    finally:
        self.is_processing = False
```

- [ ] **Step 11.3: Проверить импорт FATAL_CONSECUTIVE_ERRORS наверху файла**

Убедиться что строка в импортах есть:

```python
from backend.config import CHUNK_BYTES, SAMPLE_RATE, SILENCE_DURATION, FATAL_CONSECUTIVE_ERRORS
```

- [ ] **Step 11.4: Syntax check**

```bash
python -c "from backend.services.session import DialogSession; print('ok')"
```

Expected: `ok`.

- [ ] **Step 11.5: Запустить все unit-тесты**

```bash
pytest test/ -v --ignore=test/test_bitrix_live.py
```

Expected: все passed.

- [ ] **Step 11.6: Commit**

```bash
git add backend/services/session.py
git commit -m "feat(session): trigger _fatal_fallback on unrecoverable errors

- LLM error streak >= FATAL_CONSECUTIVE_ERRORS → fatal_fallback
- Unhandled exception в run() → fatal_fallback + re-raise
- WebSocketDisconnect не триггерит fallback (нормальное завершение)
- Счётчик _consecutive_errors сбрасывается при успешной реплике LLM"
```

---

## Task 12: Update `main.py` lifespan to load enum_ids

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 12.1: Обновить импорты в `backend/main.py`**

Добавить:

```python
from backend.config import LLM_BASE_URL, WHISPER_MODEL_SIZE, TTS_MODEL_NAME, TTS_VOICE_REF, BITRIX_WEBHOOK_URL
from backend.services.bitrix import load_ai_quality_enum_ids
```

- [ ] **Step 12.2: Обновить lifespan — добавить загрузку enum_ids**

Найти функцию `lifespan`. После строки `app.state.retriever = KnowledgeRetriever(knowledge_index)`:

```python
    # 6/6: Загрузка enum IDs для UF_CRM_AI_QUALITY
    app.state.bitrix_enum_ids = None
    if BITRIX_WEBHOOK_URL:
        logger.info("6/6: Резолв Bitrix enum IDs...")
        app.state.bitrix_enum_ids = await load_ai_quality_enum_ids(BITRIX_WEBHOOK_URL)
        logger.info("Bitrix enum IDs: %s", app.state.bitrix_enum_ids)
    else:
        logger.warning("BITRIX_WEBHOOK_URL не задан, лиды в Bitrix не будут создаваться")

    logger.info("Все модели загружены. Сервер готов.")
    yield
```

Обновить нумерацию этапов — изменить `"5/5: Загрузка базы знаний..."` на `"5/6: Загрузка базы знаний..."`.

- [ ] **Step 12.3: Syntax check**

```bash
python -c "from backend.main import app; print('ok')" 2>&1 | head -20
```

Ожидается ошибка если vLLM не запущен — это норм, смотрим только что импорт не падает. Если импорт чистый — ok. Если падает на TTS/Whisper — это тоже норм, lifespan запустится при старте uvicorn.

- [ ] **Step 12.4: Commit**

```bash
git add backend/main.py
git commit -m "feat(main): resolve Bitrix enum IDs at startup (fail-fast)

Если BITRIX_WEBHOOK_URL задан, но резолвер не находит
UF_CRM_AI_QUALITY или значения Качественный/Некачественный —
приложение упадёт на старте с понятным RuntimeError."
```

---

## Task 13: Update `main.py` websocket_endpoint for caller_phone

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 13.1: Обновить websocket_endpoint**

Заменить тело функции:

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    caller_phone_raw = websocket.query_params.get("caller_phone")
    caller_phone: str | None = None
    if caller_phone_raw:
        from backend.services.phone_normalizer import normalize_phone
        caller_phone = normalize_phone(caller_phone_raw)
        if not caller_phone:
            logger.warning(
                "Невалидный caller_phone в query-param: %r (игнорирую)",
                caller_phone_raw,
            )

    logger.info(
        "Входящий звонок: клиент подключен, caller_phone=%s",
        caller_phone or "<неизвестен>",
    )

    session = DialogSession(
        websocket=websocket,
        vad=VADEngine(app.state.vad_session),
        stt=STTEngine(app.state.stt_model),
        tts=TTSEngine(app.state.tts_model, app.state.tts_voice_prompt),
        llm=LLMAgent(app.state.llm_client, retriever=app.state.retriever),
        caller_phone=caller_phone,
        enum_ids=app.state.bitrix_enum_ids,
    )

    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info("Звонок завершен: клиент отключился")
    except Exception as e:
        logger.error("Ошибка WebSocket: %s", e, exc_info=True)
    finally:
        session.save_log()
```

- [ ] **Step 13.2: Commit**

```bash
git add backend/main.py
git commit -m "feat(main): pass caller_phone query-param to DialogSession

Нормализует caller_phone через normalize_phone, невалидный — игнорирует.
Всегда передаёт enum_ids в DialogSession (None если webhook отсутствует)."
```

---

## Task 14: Update web client to forward caller_phone

**Files:**
- Modify: `backend/web/index.html`

- [ ] **Step 14.1: Найти место создания WebSocket в `backend/web/index.html`**

Поиск по файлу:

```bash
grep -n "new WebSocket" backend/web/index.html
```

- [ ] **Step 14.2: Обновить URL WebSocket**

В найденной строке, заменить:

```javascript
const ws = new WebSocket(`${protocol}//${location.host}/ws`);
```

на:

```javascript
const queryPhone = new URLSearchParams(window.location.search).get('caller_phone');
const wsUrl = `${protocol}//${location.host}/ws${queryPhone ? `?caller_phone=${encodeURIComponent(queryPhone)}` : ''}`;
const ws = new WebSocket(wsUrl);
```

Если точный синтаксис создания WebSocket отличается — адаптировать.

- [ ] **Step 14.3: Manual test**

Запустить uvicorn (если vLLM доступен), открыть `http://localhost:8001/?caller_phone=%2B79001234567` в браузере. Проверить в логах FastAPI что видит `caller_phone=+79001234567`.

Если не можешь запустить локально — пропустить, проверится в QA-фазе.

- [ ] **Step 14.4: Commit**

```bash
git add backend/web/index.html
git commit -m "feat(web): forward caller_phone query-param into WebSocket URL

Если в URL страницы есть ?caller_phone=+7..., он пробрасывается в
WebSocket URL. Используется для ручного тестирования без АТС."
```

---

## Task 15: Extend test_bitrix_live.py with new scenarios

**Files:**
- Modify: `test/test_bitrix_live.py`

- [ ] **Step 15.1: Добавить функции для 4 сценариев**

В конец файла `test/test_bitrix_live.py`, перед `if __name__ == "__main__"` блоком, добавить:

```python
async def delete_lead(lead_id: int) -> None:
    """Удалить лид (cleanup после тестов)."""
    url = f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.delete.json"
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(url, json={"id": lead_id})


async def scenario_hot_callback() -> None:
    print("\n--- Сценарий: hot_callback ---")
    enum_ids = await load_ai_quality_enum_ids(BITRIX_WEBHOOK_URL)
    ticket_data = {
        "name": "Тест HotCallback",
        "phone": "+79001111111",
        "intent": "SMOKE: hot callback",
        "admission_year": "current",
        "request_type": "callback",
        "school_class": "11",
        "specialty": "программирование",
    }
    chat_history = [
        {"role": "assistant", "content": "Здравствуйте!"},
        {"role": "user", "content": "Хочу на программирование"},
    ]
    result = await send_to_bitrix(ticket_data, chat_history, enum_ids)
    lead_id = result.get("result") if result else None
    print(f"  lead_id = {lead_id}")
    if lead_id:
        # verify: UF_CRM_AI_QUALITY = 173, COMMENTS starts with [Обратный звонок]
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.get.json?id={lead_id}"
            )
        lead = r.json()["result"]
        assert str(lead.get("UF_CRM_AI_QUALITY")) == str(enum_ids["current"]), (
            f"UF_CRM_AI_QUALITY={lead.get('UF_CRM_AI_QUALITY')}, expected {enum_ids['current']}"
        )
        assert "[Обратный звонок]" in lead["COMMENTS"]
        print("  ✓ payload проверен")
        await delete_lead(lead_id)
        print("  ✓ cleanup (lead удалён)")


async def scenario_cold_callback() -> None:
    print("\n--- Сценарий: cold_callback ---")
    enum_ids = await load_ai_quality_enum_ids(BITRIX_WEBHOOK_URL)
    ticket_data = {
        "name": "Тест ColdCallback",
        "phone": "+79002222222",
        "intent": "SMOKE: cold callback",
        "admission_year": "next",
        "request_type": "callback",
    }
    chat_history = [
        {"role": "assistant", "content": "Здравствуйте!"},
        {"role": "user", "content": "Интересуюсь, буду поступать в следующем году"},
    ]
    result = await send_to_bitrix(ticket_data, chat_history, enum_ids)
    lead_id = result.get("result") if result else None
    print(f"  lead_id = {lead_id}")
    if lead_id:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.get.json?id={lead_id}"
            )
        lead = r.json()["result"]
        assert str(lead.get("UF_CRM_AI_QUALITY")) == str(enum_ids["next"])
        print("  ✓ UF_CRM_AI_QUALITY = Некачественный")
        await delete_lead(lead_id)


async def scenario_operator_requested() -> None:
    print("\n--- Сценарий: operator_requested ---")
    enum_ids = await load_ai_quality_enum_ids(BITRIX_WEBHOOK_URL)
    ticket_data = {
        "name": "Тест Operator",
        "phone": "+79003333333",
        "intent": "SMOKE: настоял на операторе",
        "admission_year": "current",
        "request_type": "operator_requested",
    }
    chat_history = [
        {"role": "user", "content": "Нет, мне именно оператора"},
    ]
    result = await send_to_bitrix(ticket_data, chat_history, enum_ids)
    lead_id = result.get("result") if result else None
    print(f"  lead_id = {lead_id}")
    if lead_id:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.get.json?id={lead_id}"
            )
        lead = r.json()["result"]
        assert "[Запрошен оператор]" in lead["COMMENTS"]
        print("  ✓ COMMENTS содержит [Запрошен оператор]")
        await delete_lead(lead_id)


async def scenario_fatal_fallback() -> None:
    print("\n--- Сценарий: fatal_fallback ---")
    enum_ids = await load_ai_quality_enum_ids(BITRIX_WEBHOOK_URL)
    ticket_data = {
        "name": "",
        "phone": "",
        "intent": "SMOKE: имитация fatal_fallback",
        "admission_year": None,
        "request_type": "fatal_fallback",
    }
    chat_history = [
        {"role": "assistant", "content": "Здравствуйте!"},
    ]
    result = await send_to_bitrix(
        ticket_data, chat_history, enum_ids, caller_phone="+79009999999",
    )
    lead_id = result.get("result") if result else None
    print(f"  lead_id = {lead_id}")
    if lead_id:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.get.json?id={lead_id}"
            )
        lead = r.json()["result"]
        # UF_CRM_AI_QUALITY должно быть пустым (fatal не квалифицирует)
        assert not lead.get("UF_CRM_AI_QUALITY") or lead.get("UF_CRM_AI_QUALITY") in (
            False, "", "0", 0,
        ), f"UF_CRM_AI_QUALITY должно быть пустым, got {lead.get('UF_CRM_AI_QUALITY')}"
        assert "СРОЧНО" in lead["TITLE"]
        assert "[СРОЧНО" in lead["COMMENTS"]
        print("  ✓ TITLE содержит СРОЧНО, UF_CRM_AI_QUALITY пустой")
        await delete_lead(lead_id)


async def run_all_scenarios() -> None:
    if not await ping_webhook():
        print("FAIL: webhook недоступен")
        return
    await scenario_hot_callback()
    await scenario_cold_callback()
    await scenario_operator_requested()
    await scenario_fatal_fallback()
    print("\n✓ Все сценарии отработали")
```

Обновить импорт в начале файла:

```python
from backend.services.bitrix import send_to_bitrix, load_ai_quality_enum_ids
```

Заменить main блок в конце:

```python
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--scenarios":
        asyncio.run(run_all_scenarios())
    else:
        asyncio.run(main())
```

- [ ] **Step 15.2: Запустить сценарии live**

Требует рабочего Bitrix webhook (BITRIX_WEBHOOK_URL в .env).

```bash
python -m test.test_bitrix_live --scenarios
```

Expected: 4 сценария проходят, каждый создаёт и удаляет лид. В Bitrix после запуска ничего из тестов не остаётся.

- [ ] **Step 15.3: Commit**

```bash
git add test/test_bitrix_live.py
git commit -m "test: extend live-integration scenarios for bitrix

4 новых сценария с автоматическим cleanup:
- hot_callback: admission_year='current' → UF_CRM_AI_QUALITY=173
- cold_callback: admission_year='next' → UF_CRM_AI_QUALITY=175
- operator_requested: [Запрошен оператор] в COMMENTS
- fatal_fallback: TITLE начинается с СРОЧНО, UF_CRM_AI_QUALITY пустой

Запускается через: python -m test.test_bitrix_live --scenarios"
```

---

## Task 16: Write manual QA checklist

**Files:**
- Create: `docs/manual-qa-checklist.md`

- [ ] **Step 16.1: Создать файл**

```markdown
# Manual QA Checklist

Ручные сценарии после деплоя. Каждый сценарий — реальный звонок через
веб-клиент или АТС (после интеграции Asterisk).

**Подготовка:**
- Приложение запущено (docker compose up -d)
- Bitrix webhook в .env корректный, скоуп `crm` выставлен
- Поле `UF_CRM_AI_QUALITY` настроено в «Общий вид карточки»
  (см. docs/superpowers/specs/2026-04-21-lead-qualification-design.md § 10)

---

## Сценарий 1: Callback (базовый)

**Шаги:**
1. Открой веб-клиент: `http://<host>:8001/`
2. Нажми «Начать звонок»
3. После приветствия скажи: «Здравствуйте! Меня зовут Анна, я в 11-м классе,
   интересует дизайн. Можете перезвонить сегодня вечером?»
4. Ассистент должен:
   - Уточнить детали (если нужно)
   - Спросить номер телефона
   - Перед `create_ticket` спросить: «Вы в этом году поступаете или в следующем?»
5. Ответь: «В этом»
6. Убедись, что ассистент попрощался

**Проверка в Bitrix:**
- Открой CRM → Лиды → отсортируй по дате
- Новый лид:
  - NAME = «Анна»
  - PHONE содержит введённый номер
  - STATUS_ID = NEW (по умолчанию)
  - UF_CRM_AI_QUALITY = «Качественный»
  - UF_CRM_KAKOIKLASSVIZ = «11»
  - UF_CRM_KAKAYASPETSIA = «дизайн»
  - COMMENTS начинается с `[Обратный звонок]`
  - COMMENTS содержит транскрипт с «Оператор:» / «Абитуриент:»

---

## Сценарий 2: Operator — согласился на помощь (лид НЕ создаётся)

**Шаги:**
1. Начни звонок
2. Скажи: «Переведите меня на оператора»
3. Ассистент должен:
   - Сказать «Конечно, сейчас переведу…»
   - Через короткую паузу/в следующей реплике: «К сожалению, операторы
     заняты. Но я могу помочь — специальности, формы, стоимость. Что интересует?»
4. Ответь: «Расскажите про программирование»
5. Ассистент рассказывает из RAG
6. В конце беседы, когда уместно — ассистент предлагает callback
7. Ответь: «Нет, спасибо, пока не надо» и положи трубку

**Проверка в Bitrix:**
- Новых лидов **НЕ должно быть**
- Лог: в `logs/sessions/` появится JSON-файл диалога

---

## Сценарий 3: Operator — настоял (лид создаётся)

**Шаги:**
1. Начни звонок
2. Скажи: «Переведите на оператора»
3. Ассистент предлагает помощь
4. Скажи: «Нет, мне именно оператора нужно, пусть человек перезвонит»
5. Ассистент должен сказать что-то вроде: «Поняла. Мне нужно взять ваши
   контакты, чтобы оператор перезвонил, как освободится. Как к вам обращаться?»
6. Продиктуй имя, телефон, год

**Проверка в Bitrix:**
- Новый лид:
  - UF_CRM_AI_QUALITY = в зависимости от года
  - COMMENTS начинается с `[Запрошен оператор]`
  - В транскрипте видна настойчивость абитуриента на операторе

---

## Сценарий 4: Fatal fallback

**Шаги:**
1. `docker compose stop vllm-server` — остановить LLM-сервер
2. Начни звонок в веб-клиенте
3. Скажи любую фразу
4. Ассистент ответит «Извините, у меня небольшая заминка со связью»
5. Скажи ещё 2 фразы — ассистент повторит ту же ошибку
6. На 3-ю ошибку (счётчик `FATAL_CONSECUTIVE_ERRORS=3`) ассистент должен сказать:
   «Извините, у меня технические сложности. Сейчас зафиксирую вашу заявку —
   оператор обязательно перезвонит.»
7. Ассистент просит продиктовать телефон
8. Продиктуй номер, дождись прощания, звонок завершается

**Проверка в Bitrix:**
- Новый лид:
  - TITLE начинается с «СРОЧНО: сбой бота»
  - COMMENTS начинается с `[СРОЧНО: технический сбой]`
  - UF_CRM_AI_QUALITY **пустое** (квалификация не выполнена)

**После теста:** `docker compose start vllm-server`

---

## Сценарий 5: caller_phone query-param

**Шаги:**
1. Открой: `http://<host>:8001/?caller_phone=%2B79001234567`
2. Начни звонок
3. Скажи: «Привет, хочу оставить заявку на обратный звонок»
4. Ассистент НЕ должен спрашивать телефон — он уже в системе
5. Ассистент просит имя, год, создаёт лид

**Проверка в Bitrix:**
- Новый лид с PHONE = `+79001234567`
- В логе запуска: `Входящий звонок: клиент подключен, caller_phone=+79001234567`

---

## После всех сценариев

Удалить все тестовые лиды вручную (в UI Bitrix — фильтр по TITLE или
по телефону, bulk-delete) либо запустить:

```bash
python -m test.test_bitrix_live --scenarios
```

и проверить что в списке нет зависших тестовых лидов.
```

- [ ] **Step 16.2: Commit**

```bash
git add docs/manual-qa-checklist.md
git commit -m "docs: manual QA checklist for 5 post-deploy scenarios

Покрывает все ветки из spec § 5.3:
- базовый callback
- operator + согласие на помощь (без лида)
- operator + настойчивость (лид [Запрошен оператор])
- fatal_fallback после 3 LLM-ошибок (лид [СРОЧНО])
- caller_phone через query-param"
```

---

## Task 17: Final verification run

**Files:** (ничего не модифицируем, только проверяем)

- [ ] **Step 17.1: Полный прогон unit-тестов**

```bash
pytest test/ -v --ignore=test/test_bitrix_live.py
```

Expected: все тесты passed. Зафиксировать количество.

- [ ] **Step 17.2: Live-интеграционные сценарии**

Требует рабочего Bitrix и корректного .env.

```bash
python -m test.test_bitrix_live --scenarios
```

Expected: 4 сценария проходят с cleanup.

- [ ] **Step 17.3: Деплой на удалённый сервер**

Следуя `feedback_deploy_workflow.md` из memory:

```bash
scp -r backend/ dev@192.168.2.59:~/AI-assistant-remote/AI-assistant/
scp docker-compose.yml dev@192.168.2.59:~/AI-assistant-remote/AI-assistant/
ssh dev@192.168.2.59 "cd ~/AI-assistant-remote/AI-assistant && docker compose restart api-server"
```

- [ ] **Step 17.4: Проверить запуск**

```bash
ssh dev@192.168.2.59 "docker logs voice_api --tail 50"
```

Expected: в логах видно `Bitrix enum IDs: {'current': 173, 'next': 175}` и `Сервер готов`.

- [ ] **Step 17.5: Прогнать Manual QA checklist**

Пройти все 5 сценариев из `docs/manual-qa-checklist.md`. Отмечать в чек-листе что работает, что нет.

- [ ] **Step 17.6: Final commit (если были исправления)**

Если при manual QA всплыли мелкие баги — зафиксировать:

```bash
git add <files>
git commit -m "fix: <concrete bug description from manual QA>"
```

---

## Summary

**Tasks:** 18 (0-17)

**Commits:** ~18 atomic commits, каждая задача один коммит.

**Tests added:** 4 файла (test_phone_normalizer, test_bitrix_payload, test_enum_resolver, test_llm_prompt) + 4 live-сценария.

**Deployment:** через scp + docker restart (без build — см. memory `deploy-workflow`).

**Rollback:** `git revert <commit-sha>` по atomic коммитам, либо откат бранча `v1.0` на `e9db2b08` (последний до этих изменений).

**Key risk areas (manual verification priority):**
1. Task 10-11 `_fatal_fallback` — единственная ветка без автоматических тестов, критична для safety
2. Task 12 — lifespan изменения, могут сломать старт если Bitrix недоступен
3. Task 14 — web-client JS, ломкий к синтаксису текущего файла
