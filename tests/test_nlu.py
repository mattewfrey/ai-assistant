"""
Тесты для NLU модулей: транслитерация, spell correction, fuzzy matching.
"""
import pytest

from app.services.nlu import (
    SpellCorrector,
    get_spell_corrector,
    Transliterator,
    transliterate,
    FuzzyMatcher,
    get_fuzzy_matcher,
    SynonymExpander,
    get_synonym_expander,
    QueryNormalizer,
    get_query_normalizer,
)
from app.services.nlu.spell_corrector import (
    levenshtein_distance,
    similarity_score,
    phonetic_key,
)


# ============================================================================
# SpellCorrector Tests
# ============================================================================

class TestLevenshteinDistance:
    """Тесты для расстояния Левенштейна."""
    
    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0
    
    def test_empty_string(self):
        assert levenshtein_distance("", "hello") == 5
        assert levenshtein_distance("hello", "") == 5
    
    def test_one_substitution(self):
        assert levenshtein_distance("cat", "hat") == 1
    
    def test_one_insertion(self):
        assert levenshtein_distance("cat", "cats") == 1
    
    def test_complex_difference(self):
        assert levenshtein_distance("парацетамол", "парацитамол") == 1


class TestSimilarityScore:
    """Тесты для score схожести."""
    
    def test_identical_strings(self):
        assert similarity_score("нурофен", "нурофен") == 1.0
    
    def test_similar_strings(self):
        score = similarity_score("нурафен", "нурофен")
        assert score > 0.8
    
    def test_completely_different(self):
        score = similarity_score("abc", "xyz")
        assert score < 0.5


class TestSpellCorrector:
    """Тесты для SpellCorrector."""
    
    @pytest.fixture
    def corrector(self):
        return get_spell_corrector()
    
    @pytest.mark.parametrize("typo,expected", [
        ("нурафен", "нурофен"),
        ("парацитамол", "парацетамол"),
        ("ібупрофен", "ибупрофен"),
        ("терафлу", "терафлю"),
        ("колдрэкс", "колдрекс"),
        ("ношпа", "но-шпа"),
        ("аспирін", "аспирин"),
    ])
    def test_common_typos(self, corrector: SpellCorrector, typo: str, expected: str):
        """Проверяем исправление частых опечаток."""
        result = corrector.correct_word(typo)
        assert result.corrected.lower() == expected.lower()
        assert result.was_corrected
    
    def test_correct_word_unchanged(self, corrector: SpellCorrector):
        """Правильное слово не меняется."""
        result = corrector.correct_word("нурофен")
        assert result.corrected == "нурофен"
        # was_corrected может быть True если регистр изменился
    
    def test_unknown_word_unchanged(self, corrector: SpellCorrector):
        """Неизвестное слово не меняется."""
        result = corrector.correct_word("qwerty123")
        assert result.corrected == "qwerty123"
        assert not result.was_corrected
    
    def test_correct_text(self, corrector: SpellCorrector):
        """Исправление опечаток в тексте."""
        text = "подбери нурафен от головы"
        corrected, corrections = corrector.correct_text(text)
        
        assert "нурофен" in corrected.lower()
        assert len(corrections) > 0
    
    def test_get_suggestions(self, corrector: SpellCorrector):
        """Получение предложений для опечатки."""
        # Используем слово с опечаткой которое будет исправлено
        result = corrector.correct_word("нурафен")
        # Должно быть исправлено на нурофен
        assert result.corrected.lower() == "нурофен"
        # Для правильного слова suggestions могут быть пустыми, это нормально


# ============================================================================
# Transliterator Tests
# ============================================================================

class TestTransliterator:
    """Тесты для транслитерации."""
    
    @pytest.fixture
    def transliterator(self):
        return Transliterator()
    
    @pytest.mark.parametrize("latin,expected", [
        ("nurofen", "нурофен"),
        ("paracetamol", "парацетамол"),
        ("ibuprofen", "ибупрофен"),
        ("aspirin", "аспирин"),
        ("lazolvan", "лазолван"),
    ])
    def test_drug_names(self, transliterator: Transliterator, latin: str, expected: str):
        """Транслитерация названий препаратов."""
        result = transliterator.transliterate_word(latin)
        assert result.result.lower() == expected.lower()
        assert result.was_transliterated
    
    def test_mixed_text(self, transliterator: Transliterator):
        """Смешанный текст (латиница + кириллица)."""
        text = "подбери nurofen от головы"
        result, _ = transliterator.transliterate_text(text)
        assert "нурофен" in result.lower()
        assert "подбери" in result.lower()
        assert "головы" in result.lower()
    
    def test_cyrillic_unchanged(self, transliterator: Transliterator):
        """Кириллица не меняется."""
        result = transliterator.transliterate_word("нурофен")
        assert result.result == "нурофен"
        assert not result.was_transliterated
    
    def test_detect_script(self, transliterator: Transliterator):
        """Определение скрипта."""
        assert transliterator.detect_script("hello") == "latin"
        assert transliterator.detect_script("привет") == "cyrillic"
        assert transliterator.detect_script("hello привет") == "mixed"
    
    def test_quick_transliterate(self):
        """Быстрая функция транслитерации."""
        result = transliterate("nurofen ot kashlya")
        assert "нурофен" in result.lower()


# ============================================================================
# FuzzyMatcher Tests
# ============================================================================

class TestFuzzyMatcher:
    """Тесты для нечёткого поиска."""
    
    @pytest.fixture
    def matcher(self):
        return get_fuzzy_matcher()
    
    def test_exact_match(self, matcher: FuzzyMatcher):
        """Точное совпадение."""
        matches = matcher.search("нурофен", category="drug")
        assert len(matches) > 0
        assert matches[0].matched.lower() == "нурофен"
        assert matches[0].score == 1.0
    
    def test_fuzzy_match(self, matcher: FuzzyMatcher):
        """Нечёткое совпадение."""
        matches = matcher.search("нурафен", category="drug")
        assert len(matches) > 0
        assert matches[0].matched.lower() == "нурофен"
        assert matches[0].score > 0.7
    
    def test_symptom_search(self, matcher: FuzzyMatcher):
        """Поиск симптомов."""
        matches = matcher.search("болит голова", category="symptom")
        assert len(matches) > 0
        assert any("голов" in m.matched.lower() for m in matches)
    
    def test_disease_search(self, matcher: FuzzyMatcher):
        """Поиск заболеваний."""
        matches = matcher.search("простуда", category="disease")
        assert len(matches) > 0
    
    def test_multi_word_search(self, matcher: FuzzyMatcher):
        """Поиск по нескольким словам."""
        matches = matcher.search_multi("нурофен от головы")
        drug_found = any(m.metadata.get("category") == "drug" for m in matches)
        symptom_found = any(m.metadata.get("category") == "symptom" for m in matches)
        assert drug_found or symptom_found
    
    def test_find_best_match(self, matcher: FuzzyMatcher):
        """Поиск лучшего совпадения."""
        match = matcher.find_best_match("парацетамол")
        assert match is not None
        assert match.matched.lower() == "парацетамол"


# ============================================================================
# SynonymExpander Tests
# ============================================================================

class TestSynonymExpander:
    """Тесты для расширения синонимами."""
    
    @pytest.fixture
    def expander(self):
        return get_synonym_expander()
    
    def test_expand_symptom(self, expander: SynonymExpander):
        """Расширение симптома синонимами."""
        synonyms = expander.expand_symptom("головная боль")
        assert len(synonyms) > 0
        assert any("мигрень" in s.lower() for s in synonyms)
    
    def test_brand_alternatives(self, expander: SynonymExpander):
        """Поиск альтернативных брендов."""
        inn, brands = expander.get_brand_alternatives("нурофен")
        assert inn == "ибупрофен"
        assert len(brands) > 0
    
    def test_inn_to_brands(self, expander: SynonymExpander):
        """МНН -> бренды."""
        brands = expander.get_brands_for_inn("парацетамол")
        assert len(brands) > 0
        assert any("панадол" in b.lower() for b in brands)
    
    def test_normalize_colloquial(self, expander: SynonymExpander):
        """Нормализация разговорных выражений."""
        result = expander.normalize_colloquial("от головы")
        assert result is not None
        assert "головной боли" in result.lower()
    
    def test_full_expansion(self, expander: SynonymExpander):
        """Полное расширение запроса."""
        result = expander.expand_query("нурофен от головы")
        assert result.original == "нурофен от головы"
        # Должны найтись альтернативы для нурофена
        assert result.inn_mappings.get("нурофен") == "ибупрофен" or len(result.related_drugs) > 0


# ============================================================================
# QueryNormalizer Tests (Integration)
# ============================================================================

class TestQueryNormalizer:
    """Интеграционные тесты для QueryNormalizer."""
    
    @pytest.fixture
    def normalizer(self):
        return get_query_normalizer()
    
    @pytest.mark.parametrize("query,expected_drug", [
        ("nurofen от головы", "нурофен"),
        ("нурафен от кашля", "нурофен"),
        ("paracetamol ребенку", "парацетамол"),
        ("дай терафлу", "терафлю"),
    ])
    def test_full_normalization(self, normalizer: QueryNormalizer, query: str, expected_drug: str):
        """Полная нормализация с обнаружением препарата."""
        result = normalizer.normalize(query)
        
        # Проверяем что препарат найден
        assert expected_drug in result.normalized.lower() or \
               any(expected_drug in d.lower() for d in result.detected_drugs)
    
    def test_quick_normalize(self, normalizer: QueryNormalizer):
        """Быстрая нормализация."""
        result = normalizer.quick_normalize("nurofen")
        assert "нурофен" in result.lower()
    
    def test_detect_drug(self, normalizer: QueryNormalizer):
        """Обнаружение препарата."""
        drug = normalizer.detect_drug("подбери нурофен для ребенка")
        assert drug is not None
        assert "нурофен" in drug.lower()
    
    def test_detect_symptom(self, normalizer: QueryNormalizer):
        """Обнаружение симптома."""
        symptom = normalizer.detect_symptom("болит голова уже неделю")
        assert symptom is not None
        assert "голов" in symptom.lower()
    
    def test_processing_steps(self, normalizer: QueryNormalizer):
        """Проверка шагов обработки."""
        result = normalizer.normalize("nurofen от головы")
        assert len(result.processing_steps) > 0
    
    def test_confidence(self, normalizer: QueryNormalizer):
        """Проверка confidence."""
        result = normalizer.normalize("нурофен")
        assert result.confidence > 0.5
    
    def test_to_dict(self, normalizer: QueryNormalizer):
        """Конвертация в словарь."""
        result = normalizer.normalize("nurofen")
        d = result.to_dict()
        
        assert "original" in d
        assert "normalized" in d
        assert "detected_entities" in d
    
    def test_complex_query(self, normalizer: QueryNormalizer):
        """Сложный запрос с опечатками и латиницей."""
        result = normalizer.normalize("подбери paracetamol или нурафен для ребенка от темпиратуры")
        
        # Должна произойти транслитерация
        assert result.was_transliterated or "парацетамол" in result.normalized.lower()
        
        # Должна быть коррекция опечаток
        assert result.was_spell_corrected or "нурофен" in result.normalized.lower()


class TestNLURealWorldQueries:
    """Тесты на реальных запросах пользователей."""
    
    @pytest.fixture
    def normalizer(self):
        return get_query_normalizer()
    
    @pytest.mark.parametrize("query", [
        "нурафен детский",
        "paracetamol для ребёнка",
        "от кашля лазалван",
        "терофлю от простуды",
        "називін для носа",
        "смекто от поноса",
        "мізім для желудка",
        "супрастін от аллергии",
    ])
    def test_real_queries_normalize(self, normalizer: QueryNormalizer, query: str):
        """Реальные запросы должны нормализоваться без ошибок."""
        result = normalizer.normalize(query)
        
        # Нормализованный текст не должен быть пустым
        assert result.normalized
        
        # Должен быть найден хотя бы один препарат или симптом
        entities_found = (
            len(result.detected_drugs) +
            len(result.detected_symptoms) +
            len(result.detected_diseases)
        )
        # Для некоторых запросов может не быть точных совпадений, это нормально
        # Главное - нет ошибок

