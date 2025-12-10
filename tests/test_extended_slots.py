"""
Тесты для расширенных слотов: беременность, хронические заболевания, аллергии,
длительность и выраженность симптомов.
"""
import pytest
from app.services.slot_extraction import (
    extract_pregnancy_status,
    extract_chronic_conditions,
    extract_allergies,
    extract_symptom_duration,
    extract_symptom_severity,
    extract_all_entities,
    PregnancyStatus,
)


class TestPregnancyStatusExtraction:
    """Тесты извлечения статуса беременности/лактации."""
    
    @pytest.mark.parametrize("message,expected", [
        # Беременность
        ("я беременна", PregnancyStatus.PREGNANT),
        ("при беременности можно?", PregnancyStatus.PREGNANT),
        ("беременным подойдет?", PregnancyStatus.PREGNANT),
        ("я в положении", PregnancyStatus.PREGNANT),
        ("жду ребенка", PregnancyStatus.PREGNANT),
        
        # Кормление грудью
        ("я кормящая мама", PregnancyStatus.BREASTFEEDING),
        ("при лактации", PregnancyStatus.BREASTFEEDING),
        ("кормлю грудью", PregnancyStatus.BREASTFEEDING),
        ("на гв", PregnancyStatus.BREASTFEEDING),
        ("кормящей маме", PregnancyStatus.BREASTFEEDING),
        
        # Отрицание
        ("не беременна", PregnancyStatus.NONE),
        ("ни то, ни другое", PregnancyStatus.NONE),
        ("не кормлю", PregnancyStatus.NONE),
        
        # Без указания
        ("болит голова", None),
        ("подбери лекарство", None),
    ])
    def test_pregnancy_status(self, message: str, expected: str | None):
        result = extract_pregnancy_status(message)
        assert result == expected


class TestChronicConditionsExtraction:
    """Тесты извлечения хронических заболеваний."""
    
    @pytest.mark.parametrize("message,has_chronic,conditions", [
        # Есть заболевания
        ("у меня диабет", True, ["diabetes"]),
        ("страдаю гипертонией", True, ["hypertension"]),
        ("астма с детства", True, ["asthma"]),
        ("есть проблемы с сердцем", True, ["heart_disease"]),
        ("диабет и гипертония", True, ["diabetes", "hypertension"]),
        ("проблемы с почками", True, ["kidney_disease"]),
        ("есть гастрит", True, ["stomach_ulcer"]),
        
        # Нет заболеваний
        ("нет хронических", False, []),
        ("ничем не болею", False, []),
        ("здоров полностью", False, []),
        ("противопоказаний нет", False, []),
        
        # Не указано
        ("болит голова", None, []),
    ])
    def test_chronic_conditions(self, message: str, has_chronic: bool | None, conditions: list):
        result_has, result_conditions = extract_chronic_conditions(message)
        assert result_has == has_chronic
        if conditions:
            for cond in conditions:
                assert cond in result_conditions


class TestAllergiesExtraction:
    """Тесты извлечения аллергий."""
    
    @pytest.mark.parametrize("message,has_allergy,allergens_exist", [
        # Есть аллергия
        ("аллергия на пенициллин", True, True),
        ("непереносимость аспирина", True, True),
        ("аллергик на антибиотики", True, False),  # Просто маркер, без конкретики
        ("реакция на ибупрофен", True, True),
        
        # Нет аллергии
        ("аллергии нет", False, False),
        ("не аллергик", False, False),
        ("нет аллергии", False, False),
        
        # Не указано
        ("болит живот", None, False),
    ])
    def test_allergies(self, message: str, has_allergy: bool | None, allergens_exist: bool):
        result_has, result_allergens = extract_allergies(message)
        assert result_has == has_allergy
        if allergens_exist:
            assert len(result_allergens) > 0
        else:
            assert len(result_allergens) == 0


class TestSymptomDurationExtraction:
    """Тесты извлечения длительности симптомов."""
    
    @pytest.mark.parametrize("message,expected", [
        # Сегодня
        ("заболел сегодня", "today"),
        ("с утра болит", "today"),
        ("только что началось", "today"),
        ("час назад появилось", "today"),
        
        # Несколько дней
        ("болит пару дней", "few_days"),
        ("началось вчера", "few_days"),
        ("уже 2 дня", "few_days"),
        ("три дня кашляю", "few_days"),
        
        # Неделя
        ("болит уже неделю", "week"),
        ("дней 5 уже", "week"),
        ("семь дней мучаюсь", "week"),
        
        # Давно
        ("давно беспокоит", "long"),
        ("уже больше недели", "long"),
        ("месяц страдаю", "long"),
        ("хронически болит", "long"),
        
        # Не указано
        ("болит голова", None),
    ])
    def test_symptom_duration(self, message: str, expected: str | None):
        result = extract_symptom_duration(message)
        assert result == expected


class TestSymptomSeverityExtraction:
    """Тесты извлечения выраженности симптомов."""
    
    @pytest.mark.parametrize("message,expected", [
        # Слабо
        ("немного болит", "mild"),
        ("слегка ноет", "mild"),
        ("терпимо", "mild"),
        ("чуть-чуть", "mild"),
        
        # Умеренно
        ("умеренная боль", "moderate"),
        ("ощутимо болит", "moderate"),
        ("заметно беспокоит", "moderate"),
        
        # Сильно
        ("очень сильно болит", "severe"),
        ("невыносимая боль", "severe"),
        ("ужасно болит", "severe"),
        ("острая боль", "severe"),
        ("нестерпимо", "severe"),
        
        # Не указано
        ("болит голова", None),
    ])
    def test_symptom_severity(self, message: str, expected: str | None):
        result = extract_symptom_severity(message)
        assert result == expected


class TestComplexExtractionWithNewSlots:
    """Тесты комплексного извлечения с новыми слотами."""
    
    def test_extraction_with_pregnancy_and_symptom(self):
        """Извлечение симптома и статуса беременности."""
        result = extract_all_entities("я беременна, болит голова")
        slots = result.to_slots_dict()
        
        assert slots.get("pregnancy_status") == PregnancyStatus.PREGNANT
        assert "головная боль" in slots.get("symptom", "").lower() or "голов" in slots.get("symptom", "").lower()
    
    def test_extraction_with_chronic_and_symptom(self):
        """Извлечение симптома и хронического заболевания."""
        result = extract_all_entities("у меня диабет, поднялась температура")
        slots = result.to_slots_dict()
        
        assert slots.get("has_chronic_conditions") is True
        assert "diabetes" in slots.get("chronic_conditions", [])
    
    def test_extraction_with_duration_and_severity(self):
        """Извлечение длительности и выраженности симптомов."""
        result = extract_all_entities("сильно болит голова уже неделю")
        slots = result.to_slots_dict()
        
        assert slots.get("symptom_severity") == "severe"
        assert slots.get("symptom_duration") == "week"
    
    def test_extraction_with_allergy(self):
        """Извлечение информации об аллергии."""
        result = extract_all_entities("болит горло, аллергия на пенициллин")
        slots = result.to_slots_dict()
        
        assert slots.get("has_allergies") is True
    
    def test_extraction_negative_conditions(self):
        """Извлечение отрицательных ответов."""
        result = extract_all_entities("нет хронических, аллергии нет, не беременна")
        slots = result.to_slots_dict()
        
        assert slots.get("has_chronic_conditions") is False
        assert slots.get("has_allergies") is False
        assert slots.get("pregnancy_status") == PregnancyStatus.NONE


class TestToSlotsDictWithNewFields:
    """Тесты конвертации ExtractionResult в словарь слотов."""
    
    def test_all_new_fields_in_slots_dict(self):
        """Проверка что все новые поля попадают в словарь."""
        result = extract_all_entities(
            "я беременна, болит голова сильно уже неделю, есть диабет, аллергия на аспирин"
        )
        slots = result.to_slots_dict()
        
        # Проверяем наличие новых полей
        assert "pregnancy_status" in slots
        assert "symptom_duration" in slots
        assert "symptom_severity" in slots
        assert "has_chronic_conditions" in slots
        assert "has_allergies" in slots

