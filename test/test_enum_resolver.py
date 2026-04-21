"""Тесты резолвера enum ID для UF_CRM_AI_QUALITY."""
import pytest
import respx
from httpx import Response

from backend.services.bitrix import load_ai_quality_enum_ids


WEBHOOK = "https://example.bitrix24.ru/rest/1/token"
ENDPOINT = f"{WEBHOOK}/crm.lead.userfield.list.json"


@respx.mock(using="httpx", assert_all_called=False)
@pytest.mark.asyncio
async def test_resolver_success(respx_mock):
    respx_mock.post(ENDPOINT).mock(
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


@respx.mock(using="httpx", assert_all_called=False)
@pytest.mark.asyncio
async def test_resolver_field_not_found(respx_mock):
    respx_mock.post(ENDPOINT).mock(
        return_value=Response(200, json={"result": []})
    )
    with pytest.raises(RuntimeError, match="UF_CRM_AI_QUALITY"):
        await load_ai_quality_enum_ids(WEBHOOK)


@respx.mock(using="httpx", assert_all_called=False)
@pytest.mark.asyncio
async def test_resolver_missing_value(respx_mock):
    respx_mock.post(ENDPOINT).mock(
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


@respx.mock(using="httpx", assert_all_called=False)
@pytest.mark.asyncio
async def test_resolver_http_error(respx_mock):
    respx_mock.post(ENDPOINT).mock(
        return_value=Response(500, text="Internal Server Error")
    )
    with pytest.raises(RuntimeError):
        await load_ai_quality_enum_ids(WEBHOOK)
