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
from backend.services.bitrix import send_to_bitrix  # noqa: E402

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


if __name__ == "__main__":
    asyncio.run(main())
