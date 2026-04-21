"""Нормализация российских телефонных номеров из любого формата.

Поддерживает:
- Прописью: "восемь девятьсот двенадцать тридцать четыре пятьдесят шесть"
- Смешанный: "восемь 900 123 45 56"
- Цифры с разделителями: "8-900-123-45-56", "8 (900) 123-45-56"
- Международный: "+7 900 123 45 56"
"""

import logging
import re

logger = logging.getLogger(__name__)

# Русские числительные -> цифры
_ONES = {
    "ноль": "0", "один": "1", "одна": "1", "два": "2", "две": "2",
    "три": "3", "четыре": "4", "пять": "5", "шесть": "6",
    "семь": "7", "восемь": "8", "девять": "9",
}

_TEENS = {
    "десять": "10", "одиннадцать": "11", "двенадцать": "12",
    "тринадцать": "13", "четырнадцать": "14", "пятнадцать": "15",
    "шестнадцать": "16", "семнадцать": "17", "восемнадцать": "18",
    "девятнадцать": "19",
}

_TENS = {
    "двадцать": "2", "тридцать": "3", "сорок": "4",
    "пятьдесят": "5", "шестьдесят": "6", "семьдесят": "7",
    "восемьдесят": "8", "девяносто": "9",
}

_HUNDREDS = {
    "сто": "1", "двести": "2", "триста": "3", "четыреста": "4",
    "пятьсот": "5", "шестьсот": "6", "семьсот": "7",
    "восемьсот": "8", "девятьсот": "9",
}


def _words_to_digits(text: str) -> str:
    """Заменяет русские числительные на цифры, сохраняя остальной текст."""
    words = text.lower().split()
    result: list[str] = []
    i = 0

    while i < len(words):
        word = words[i]

        # Сотни (900 -> "9")
        if word in _HUNDREDS:
            group = _HUNDREDS[word]
            # Проверяем следующее слово — десятки или единицы
            if i + 1 < len(words) and words[i + 1] in _TEENS:
                group += _TEENS[words[i + 1]]
                i += 2
            elif i + 1 < len(words) and words[i + 1] in _TENS:
                group += _TENS[words[i + 1]]
                if i + 2 < len(words) and words[i + 2] in _ONES:
                    group += _ONES[words[i + 2]]
                    i += 3
                else:
                    group += "0"
                    i += 2
            elif i + 1 < len(words) and words[i + 1] in _ONES:
                group += "0" + _ONES[words[i + 1]]
                i += 2
            else:
                group += "00"
                i += 1
            result.append(group)

        # Подростки (10-19)
        elif word in _TEENS:
            result.append(_TEENS[word])
            i += 1

        # Десятки (20-90)
        elif word in _TENS:
            group = _TENS[word]
            if i + 1 < len(words) and words[i + 1] in _ONES:
                group += _ONES[words[i + 1]]
                i += 2
            else:
                group += "0"
                i += 1
            result.append(group)

        # Единицы (0-9)
        elif word in _ONES:
            result.append(_ONES[word])
            i += 1

        else:
            # Оставляем как есть (цифры, разделители и т.д.)
            result.append(word)
            i += 1

    return " ".join(result)


def normalize_phone(raw: str) -> str | None:
    """Нормализует телефонный номер в формат +7XXXXXXXXXX.

    Возвращает None если номер невалиден.
    """
    if not raw or not raw.strip():
        return None

    # Шаг 1: преобразуем числительные в цифры
    converted = _words_to_digits(raw)

    # Шаг 2: оставляем только цифры и +
    digits_only = re.sub(r"[^\d+]", "", converted)

    # Убираем ведущий +
    if digits_only.startswith("+"):
        digits_only = digits_only[1:]

    # Шаг 3: нормализуем до 11 цифр с 7 в начале
    if len(digits_only) == 11:
        if digits_only[0] in ("7", "8"):
            normalized = "7" + digits_only[1:]
        else:
            logger.warning("Неожиданный код страны: %s", digits_only)
            return None
    elif len(digits_only) == 10:
        # Без кода страны — добавляем 7
        normalized = "7" + digits_only
    else:
        logger.warning(
            "Невалидная длина номера (%d цифр): %s -> %s",
            len(digits_only), raw, digits_only,
        )
        return None

    return f"+{normalized}"
