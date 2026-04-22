"""Тесты для apply_pronunciations — замены перед TTS/RUAccent."""
from backend.engines.tts_pronunciations import apply_pronunciations


class TestBasicReplacements:
    def test_abbreviation_upper(self):
        assert apply_pronunciations("в КЭСИ") == "в кэс+и"

    def test_abbreviation_mixed_case(self):
        assert apply_pronunciations("В Кэси") == "В кэс+и"

    def test_abbreviation_lower(self):
        assert apply_pronunciations("колледж кэси") == "колл+едж кэс+и"

    def test_no_replacement_inside_other_word(self):
        # "КЭСИ2024" не должен заменяться — это не отдельное слово
        result = apply_pronunciations("проект КЭСИ2024")
        assert "кэс+и2024" not in result.lower()
        assert "КЭСИ2024" in result or "кэси2024" in result.lower()

    def test_multi_word_key_wins_over_single(self):
        # "колледже КЭСИ" должен сматчиться ДО того как одиночный "КЭСИ"
        # обработается — проверяем что получаем правильную форму
        result = apply_pronunciations("в колледже КЭСИ")
        assert "колл+едже кэс+и" in result


class TestEdgeCases:
    def test_empty_string(self):
        assert apply_pronunciations("") == ""

    def test_no_target_words(self):
        assert apply_pronunciations("просто текст без терминов") == "просто текст без терминов"

    def test_multiple_occurrences(self):
        result = apply_pronunciations("КЭСИ лучше чем не-КЭСИ")
        # оба КЭСИ должны замениться
        assert result.count("кэс+и") == 2

    def test_preserves_punctuation(self):
        result = apply_pronunciations("В КЭСИ, после 11 класса.")
        assert "," in result and "." in result
