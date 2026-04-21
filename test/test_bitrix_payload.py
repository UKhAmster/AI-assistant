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
