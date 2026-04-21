"""Тесты чистых функций bitrix.py: _format_comments, _build_lead_payload."""
import pytest

from backend.services.bitrix import _format_comments, _build_lead_payload


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
