import logging
from typing import Any

import httpx

from backend.config import BITRIX_WEBHOOK_URL
from backend.services.phone_normalizer import normalize_phone

logger = logging.getLogger(__name__)

# Retry: 3 попытки, экспоненциальная задержка
MAX_RETRIES = 3
BASE_TIMEOUT = 10.0

_REQUEST_TYPE_PREFIX: dict[str, str] = {
    "callback": "[Обратный звонок]",
    "operator_requested": "[Запрошен оператор]",
    "fatal_fallback": "[СРОЧНО: технический сбой]",
}


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
        async with httpx.AsyncClient(timeout=BASE_TIMEOUT, trust_env=False) as client:
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


async def send_to_bitrix(ticket_data: dict[str, Any]) -> dict[str, Any] | None:
    """Создает лид в Bitrix24 CRM через REST webhook.

    Возвращает ответ Bitrix24 при успехе, None при ошибке.
    Никогда не бросает исключений — ошибки логируются.
    """
    if not BITRIX_WEBHOOK_URL:
        logger.warning("BITRIX_WEBHOOK_URL не задан, лид не отправлен: %s", ticket_data)
        return None

    # Нормализация телефона
    raw_phone = ticket_data.get("phone", "")
    phone = normalize_phone(raw_phone)
    if not phone:
        logger.warning("Невалидный номер телефона, лид не создан: %s", raw_phone)
        return None

    url = f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.add.json"
    payload = {
        "fields": {
            "TITLE": f"Звонок: {ticket_data.get('intent', 'без темы')}",
            "NAME": ticket_data.get("name", ""),
            "PHONE": [
                {
                    "VALUE": phone,
                    "VALUE_TYPE": "MOBILE",
                }
            ],
            "SOURCE_ID": "CALLBACK",
            "COMMENTS": ticket_data.get("intent", ""),
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=BASE_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                lead_id = result.get("result")
                logger.info("Лид создан в Bitrix24, ID: %s", lead_id)
                return result

        except httpx.TimeoutException:
            logger.warning(
                "Таймаут Bitrix24 (попытка %d/%d)", attempt, MAX_RETRIES
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "Bitrix24 вернул ошибку %s (попытка %d/%d): %s",
                e.response.status_code,
                attempt,
                MAX_RETRIES,
                e.response.text[:200],
            )
        except Exception as e:
            logger.error(
                "Ошибка отправки в Bitrix24 (попытка %d/%d): %s",
                attempt,
                MAX_RETRIES,
                e,
            )

        if attempt < MAX_RETRIES:
            import asyncio
            await asyncio.sleep(2 ** attempt)

    logger.error("Не удалось создать лид после %d попыток: %s", MAX_RETRIES, ticket_data)
    return None
