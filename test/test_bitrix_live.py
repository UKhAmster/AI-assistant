"""Smoke-test интеграции с Bitrix24.

Последовательность:
  1) GET crm.lead.fields.json — проверка webhook (readonly, не создает лид)
  2) POST crm.lead.add.json через send_to_bitrix() — создание тестового лида
  3) Печать ID и прямой ссылки на карточку лида

Запуск из корня проекта:
    python -m test.test_bitrix_live

.env подхватывается автоматически.
"""
import asyncio
import logging
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Минимальный парсер .env — без зависимости от python-dotenv."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

import httpx  # noqa: E402

from backend.config import BITRIX_WEBHOOK_URL  # noqa: E402
from backend.services.bitrix import send_to_bitrix, load_ai_quality_enum_ids  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def ping_webhook() -> bool:
    url = f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.fields.json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
    except httpx.HTTPError as e:
        print(f"FAIL ping: {type(e).__name__}: {e}")
        return False

    if response.status_code != 200:
        print(f"FAIL ping: HTTP {response.status_code}: {response.text[:300]}")
        return False

    data = response.json()
    if "error" in data:
        print(f"FAIL ping: {data['error']} — {data.get('error_description', '')}")
        return False

    fields = data.get("result", {})
    print(f"OK ping: webhook отвечает, {len(fields)} полей в crm.lead.fields")
    uf_fields = sorted(k for k in fields if k.startswith("UF_"))
    if uf_fields:
        print(f"   Кастомные поля (UF_*): {', '.join(uf_fields[:15])}")
    return True


async def create_test_lead() -> None:
    print()
    print("Создаю тестовый лид...")
    result = await send_to_bitrix({
        "name": "Тест Smoke",
        "phone": "+7 900 123 45 67",
        "intent": "SMOKE-TEST: проверка интеграции голосового ассистента с Bitrix24",
    })
    if result is None:
        print("FAIL: лид не создан — смотри логи выше")
        return
    lead_id = result.get("result")
    print(f"OK: лид создан, ID = {lead_id}")
    # Домен вытащим из webhook URL, чтобы ссылка работала для любого Bitrix
    domain = BITRIX_WEBHOOK_URL.split("/rest/")[0]
    print(f"   {domain}/crm/lead/details/{lead_id}/")


async def main() -> None:
    if not BITRIX_WEBHOOK_URL:
        print("BITRIX_WEBHOOK_URL не задан в .env")
        return
    masked = BITRIX_WEBHOOK_URL[:50] + "..." if len(BITRIX_WEBHOOK_URL) > 50 else BITRIX_WEBHOOK_URL
    print(f"Webhook: {masked}")
    print()
    if not await ping_webhook():
        print()
        print("Не удалось достучаться до webhook. Проверь:")
        print("  - URL в .env")
        print("  - Скоуп 'crm' включён на входящем вебхуке")
        return
    await create_test_lead()


async def delete_lead(lead_id: int) -> None:
    """Удалить лид (cleanup после тестов)."""
    url = f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.delete.json"
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
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
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.get.json?id={lead_id}"
            )
        lead = r.json()["result"]
        assert str(lead.get("UF_CRM_AI_QUALITY")) == str(enum_ids["current"]), (
            f"UF_CRM_AI_QUALITY={lead.get('UF_CRM_AI_QUALITY')}, expected {enum_ids['current']}"
        )
        assert "(Обратный звонок)" in lead["COMMENTS"]
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
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
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
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.get.json?id={lead_id}"
            )
        lead = r.json()["result"]
        assert "(Запрошен оператор)" in lead["COMMENTS"]
        print("  ✓ COMMENTS содержит (Запрошен оператор)")
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
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.get.json?id={lead_id}"
            )
        lead = r.json()["result"]
        # UF_CRM_AI_QUALITY должно быть пустым (fatal не квалифицирует)
        assert not lead.get("UF_CRM_AI_QUALITY") or lead.get("UF_CRM_AI_QUALITY") in (
            False, "", "0", 0,
        ), f"UF_CRM_AI_QUALITY должно быть пустым, got {lead.get('UF_CRM_AI_QUALITY')}"
        assert "СРОЧНО" in lead["TITLE"]
        assert "(СРОЧНО" in lead["COMMENTS"]
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--scenarios":
        asyncio.run(run_all_scenarios())
    else:
        asyncio.run(main())
