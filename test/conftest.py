"""Общие pytest-фикстуры для тестов."""
import os
import pytest

# Диагностические скрипты — не pytest-тесты (выполняют asyncio.run на уровне модуля).
# Исключаем их из коллекции, чтобы pytest не падал при импорте.
collect_ignore = [
    os.path.join(os.path.dirname(__file__), f)
    for f in [
        "test_bitrix_live.py",
        "test_force_vad.py",
        "test_sine.py",
        "test_vad.py",
        "test_vad2.py",
        "test_vad_agc.py",
        "test_vad_norm.py",
        "test_vad_thresh.py",
        "test_whisper.py",
    ]
]


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
