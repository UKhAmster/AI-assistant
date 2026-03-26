import logging
from typing import Any

import httpx

from src.config import BITRIX_WEBHOOK_URL
from src.services.phone_normalizer import normalize_phone

logger = logging.getLogger(__name__)

# Retry: 3 попытки, экспоненциальная задержка
MAX_RETRIES = 3
BASE_TIMEOUT = 10.0


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
