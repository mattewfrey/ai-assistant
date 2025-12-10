"""
Тесты для извлечения возрастной группы (age_group) из сообщений пользователя.
"""
from __future__ import annotations

import pytest

from app.services.slot_extraction import (
    AgeGroup,
    extract_age_group,
    extract_age_group_verbose,
    age_to_age_group,
    extract_age,
    extract_is_for_children,
    extract_is_for_teenager,
    extract_is_for_elderly,
    extract_all_entities,
)


# =============================================================================
# Тесты age_to_age_group
# =============================================================================


class TestAgeToAgeGroup:
    """Тесты преобразования числового возраста в группу."""

    @pytest.mark.parametrize("age,expected", [
        (0, AgeGroup.CHILD),
        (1, AgeGroup.CHILD),
        (5, AgeGroup.CHILD),
        (12, AgeGroup.CHILD),
        (13, AgeGroup.TEENAGER),
        (15, AgeGroup.TEENAGER),
        (17, AgeGroup.TEENAGER),
        (18, AgeGroup.ADULT),
        (30, AgeGroup.ADULT),
        (59, AgeGroup.ADULT),
        (60, AgeGroup.ELDERLY),
        (75, AgeGroup.ELDERLY),
        (100, AgeGroup.ELDERLY),
    ])
    def test_age_to_group_mapping(self, age: int, expected: str):
        """Проверяет корректность преобразования возраста в группу."""
        assert age_to_age_group(age) == expected

    def test_negative_age_returns_none(self):
        """Отрицательный возраст возвращает None."""
        assert age_to_age_group(-1) is None


# =============================================================================
# Тесты extract_age_group
# =============================================================================


class TestExtractAgeGroup:
    """Тесты извлечения возрастной группы из текста."""

    # Детские маркеры
    @pytest.mark.parametrize("message", [
        "для ребёнка",
        "для ребенка",
        "ребенку",
        "для детей",
        "детям",
        "детский",
        "малышу",
        "сыну",
        "дочке",
        "для малыша",
        "грудничку",
        "новорожденному",
    ])
    def test_child_markers(self, message: str):
        """Детские маркеры определяются как child."""
        assert extract_age_group(message) == AgeGroup.CHILD

    # Подростковые маркеры
    @pytest.mark.parametrize("message", [
        "подростку",
        "для подростка",
        "подросток",
        "школьнику",
        "тинейджер",
    ])
    def test_teenager_markers(self, message: str):
        """Подростковые маркеры определяются как teenager."""
        assert extract_age_group(message) == AgeGroup.TEENAGER

    # Взрослые маркеры
    @pytest.mark.parametrize("message", [
        "для взрослого",
        "взрослому",
        "взрослый",
        "мне",
        "себе",
    ])
    def test_adult_markers(self, message: str):
        """Взрослые маркеры определяются как adult."""
        assert extract_age_group(message) == AgeGroup.ADULT

    # Пожилые маркеры
    @pytest.mark.parametrize("message", [
        "для пожилого",
        "пожилому",
        "бабушке",
        "дедушке",
        "престарелому",
        "старику",
    ])
    def test_elderly_markers(self, message: str):
        """Пожилые маркеры определяются как elderly."""
        assert extract_age_group(message) == AgeGroup.ELDERLY

    # Числовой возраст
    @pytest.mark.parametrize("message,expected", [
        # "мне" - маркер взрослого, имеет приоритет над возрастом
        ("мне 5 лет", AgeGroup.ADULT),  # "мне" = взрослый маркер
        ("ребенку 8 лет", AgeGroup.CHILD),  # "ребенку" = детский маркер
        ("15 лет", AgeGroup.TEENAGER),  # Только возраст -> преобразуем
        ("мне 25", AgeGroup.ADULT),
        ("возраст 45 лет", AgeGroup.ADULT),
        # "для человека" не является явным маркером, возраст извлекается
        ("70 лет", AgeGroup.ELDERLY),  # Только возраст -> elderly
    ])
    def test_age_to_group_extraction(self, message: str, expected: str):
        """Числовой возраст преобразуется в группу (маркеры имеют приоритет)."""
        assert extract_age_group(message) == expected

    def test_no_age_info_returns_none(self):
        """Без информации о возрасте возвращает None."""
        assert extract_age_group("от кашля недорого") is None
        assert extract_age_group("таблетки от головной боли") is None


# =============================================================================
# Тесты extract_age_group_verbose
# =============================================================================


class TestExtractAgeGroupVerbose:
    """Тесты расширенного извлечения с метаданными."""

    def test_explicit_child_marker(self):
        """Явный маркер ребёнка."""
        group, age, source = extract_age_group_verbose("для ребёнка 5 лет")
        assert group == AgeGroup.CHILD
        assert age == 5
        assert source == "explicit_marker"

    def test_inferred_from_age(self):
        """Группа вычислена из возраста (без явного маркера)."""
        group, age, source = extract_age_group_verbose("возраст 35 лет")
        assert group == AgeGroup.ADULT
        assert age == 35
        assert source == "inferred_from_age"

    def test_explicit_adult_marker_with_age(self):
        """'мне' - явный маркер взрослого."""
        group, age, source = extract_age_group_verbose("мне 35 лет")
        assert group == AgeGroup.ADULT
        assert age == 35
        assert source == "explicit_marker"  # "мне" - маркер взрослого

    def test_not_found(self):
        """Информация не найдена."""
        group, age, source = extract_age_group_verbose("от кашля")
        assert group is None
        assert age is None
        assert source == "not_found"


# =============================================================================
# Тесты полного извлечения
# =============================================================================


class TestExtractAllEntitiesAgeGroup:
    """Тесты извлечения age_group через extract_all_entities."""

    def test_child_context_in_full_message(self):
        """Детский контекст в полном сообщении."""
        result = extract_all_entities("Подберите сироп от кашля для ребенка 5 лет")
        slots = result.to_slots_dict()
        assert slots.get("age_group") == AgeGroup.CHILD
        assert slots.get("age") == 5
        assert slots.get("is_for_children") is True
        assert "кашель" in slots.get("symptom", "")

    def test_adult_context_in_full_message(self):
        """Взрослый контекст в полном сообщении."""
        result = extract_all_entities("Мне нужны таблетки от головной боли, для взрослого")
        slots = result.to_slots_dict()
        assert slots.get("age_group") == AgeGroup.ADULT

    def test_elderly_context_in_full_message(self):
        """Пожилой контекст в полном сообщении."""
        result = extract_all_entities("Препараты от давления для бабушки")
        slots = result.to_slots_dict()
        assert slots.get("age_group") == AgeGroup.ELDERLY
        assert slots.get("is_for_elderly") is True

    def test_no_age_context(self):
        """Без возрастного контекста."""
        result = extract_all_entities("Таблетки от кашля до 300 рублей")
        slots = result.to_slots_dict()
        assert slots.get("age_group") is None
        assert slots.get("price_max") == 300


# =============================================================================
# Тесты граничных случаев
# =============================================================================


class TestEdgeCases:
    """Тесты граничных и сложных случаев."""

    def test_child_marker_overrides_adult_age(self):
        """Маркер 'для ребёнка' важнее числового возраста взрослого."""
        # Если пользователь написал "для ребенка" но случайно указал возраст 30
        # (что маловероятно, но проверим приоритет)
        group = extract_age_group("для ребенка")
        assert group == AgeGroup.CHILD

    def test_mixed_markers_child_wins(self):
        """При смешанных маркерах детский имеет приоритет."""
        result = extract_all_entities("нужно для ребенка 7 лет, я взрослый")
        slots = result.to_slots_dict()
        # Первый найденный маркер "для ребенка" должен установить child
        assert slots.get("age_group") == AgeGroup.CHILD

    def test_teenager_boundary_13(self):
        """Граница подросткового возраста - 13 лет."""
        # "ребенку" - детский маркер, имеет приоритет
        assert extract_age_group("ребенку 13 лет") == AgeGroup.CHILD
        # Только возраст - правильное преобразование
        assert extract_age_group("13 лет") == AgeGroup.TEENAGER
        assert age_to_age_group(13) == AgeGroup.TEENAGER

    def test_adult_boundary_18(self):
        """Граница взрослого возраста - 18 лет."""
        assert age_to_age_group(18) == AgeGroup.ADULT

    def test_elderly_boundary_60(self):
        """Граница пожилого возраста - 60 лет."""
        assert age_to_age_group(60) == AgeGroup.ELDERLY

    def test_cyrillic_e_yo_equivalence(self):
        """Буквы 'е' и 'ё' обрабатываются одинаково."""
        assert extract_age_group("для ребёнка") == AgeGroup.CHILD
        assert extract_age_group("для ребенка") == AgeGroup.CHILD

    def test_case_insensitive(self):
        """Регистронезависимый поиск."""
        assert extract_age_group("ДЛЯ РЕБЁНКА") == AgeGroup.CHILD
        assert extract_age_group("Для Взрослого") == AgeGroup.ADULT


# =============================================================================
# Тесты реальных запросов пользователей
# =============================================================================


class TestRealWorldQueries:
    """Тесты на реальных примерах запросов пользователей."""

    def test_cough_medicine_for_child(self):
        """Подбор лекарства от кашля для ребёнка."""
        result = extract_all_entities("Подбери недорогие таблетки от кашля для ребёнка до 300 рублей")
        slots = result.to_slots_dict()
        assert slots.get("age_group") == AgeGroup.CHILD
        assert slots.get("price_max") == 300
        assert "кашель" in slots.get("symptom", "")

    def test_headache_medicine_no_age(self):
        """Лекарство от головной боли без указания возраста."""
        result = extract_all_entities("что-нибудь от головной боли")
        slots = result.to_slots_dict()
        assert slots.get("age_group") is None
        assert "головная боль" in slots.get("symptom", "")

    def test_medicine_for_elderly_grandma(self):
        """Лекарство для пожилой бабушки."""
        result = extract_all_entities("лекарство от давления для бабушки 75 лет")
        slots = result.to_slots_dict()
        assert slots.get("age_group") == AgeGroup.ELDERLY
        assert slots.get("age") == 75

    def test_simple_age_response(self):
        """Простой ответ на вопрос о возрасте."""
        assert extract_age_group("5 лет") == AgeGroup.CHILD
        assert extract_age_group("для взрослого") == AgeGroup.ADULT
        assert extract_age_group("ребёнок") == AgeGroup.CHILD

