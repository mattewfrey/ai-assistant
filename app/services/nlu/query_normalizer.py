"""
QueryNormalizer - главный сервис нормализации запросов.

Объединяет все компоненты NLU:
1. Транслитерация (латиница → кириллица)
2. Исправление опечаток
3. Нечёткий поиск по базе терминов
4. Расширение синонимами

Пайплайн обработки:
input → transliterate → spell_correct → fuzzy_match → expand → output
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .transliterator import Transliterator, get_transliterator
from .spell_corrector import SpellCorrector, get_spell_corrector, CorrectionResult
from .fuzzy_matcher import FuzzyMatcher, FuzzyMatch, get_fuzzy_matcher
from .synonym_expander import SynonymExpander, get_synonym_expander, ExpansionResult

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Полный результат нормализации запроса."""
    
    # Входные данные
    original_query: str
    
    # Результат после каждого этапа
    transliterated: str
    spell_corrected: str
    normalized: str  # Финальный нормализованный текст
    
    # Детали обработки (булевы флаги идут перед полями с default)
    was_transliterated: bool
    was_spell_corrected: bool
    
    # Поля с default значениями
    transliteration_details: List[Dict[str, Any]] = field(default_factory=list)
    spell_corrections: List[CorrectionResult] = field(default_factory=list)
    fuzzy_matches: List[FuzzyMatch] = field(default_factory=list)
    
    # Расширения
    expansion: ExpansionResult | None = None
    
    # Найденные сущности
    detected_drugs: List[str] = field(default_factory=list)
    detected_symptoms: List[str] = field(default_factory=list)
    detected_diseases: List[str] = field(default_factory=list)
    detected_forms: List[str] = field(default_factory=list)
    
    # Метаданные
    confidence: float = 1.0
    processing_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для debug."""
        return {
            "original": self.original_query,
            "normalized": self.normalized,
            "was_transliterated": self.was_transliterated,
            "was_spell_corrected": self.was_spell_corrected,
            "spell_corrections": [
                {"from": c.original, "to": c.corrected, "confidence": c.confidence}
                for c in self.spell_corrections
            ],
            "detected_entities": {
                "drugs": self.detected_drugs,
                "symptoms": self.detected_symptoms,
                "diseases": self.detected_diseases,
                "forms": self.detected_forms,
            },
            "confidence": self.confidence,
            "processing_steps": self.processing_steps,
        }


class QueryNormalizer:
    """
    Главный сервис нормализации запросов.
    
    Применяет последовательно:
    1. Транслитерацию (nurofen → нурофен)
    2. Исправление опечаток (парацитамол → парацетамол)
    3. Нечёткий поиск сущностей
    4. Расширение синонимами (опционально)
    
    Пример:
        normalizer = QueryNormalizer()
        result = normalizer.normalize("nurafen ot golovy")
        # result.normalized = "нурофен от головной боли"
    """
    
    def __init__(
        self,
        transliterator: Transliterator | None = None,
        spell_corrector: SpellCorrector | None = None,
        fuzzy_matcher: FuzzyMatcher | None = None,
        synonym_expander: SynonymExpander | None = None,
        enable_expansion: bool = True,
        spell_confidence_threshold: float = 0.75,
        fuzzy_confidence_threshold: float = 0.6,
    ):
        self._transliterator = transliterator or get_transliterator()
        self._spell_corrector = spell_corrector or get_spell_corrector()
        self._fuzzy_matcher = fuzzy_matcher or get_fuzzy_matcher()
        self._synonym_expander = synonym_expander or get_synonym_expander()
        
        self._enable_expansion = enable_expansion
        self._spell_threshold = spell_confidence_threshold
        self._fuzzy_threshold = fuzzy_confidence_threshold
    
    def normalize(
        self,
        query: str,
        expand_synonyms: bool | None = None,
        detect_entities: bool = True,
    ) -> NormalizationResult:
        """
        Полная нормализация запроса.
        
        Args:
            query: Исходный запрос пользователя
            expand_synonyms: Расширять синонимами (None = использовать дефолт)
            detect_entities: Распознавать сущности (препараты, симптомы)
        
        Returns:
            NormalizationResult с полной информацией об обработке
        """
        if not query or not query.strip():
            return NormalizationResult(
                original_query=query,
                transliterated=query,
                spell_corrected=query,
                normalized=query,
                was_transliterated=False,
                was_spell_corrected=False,
            )
        
        processing_steps: List[str] = []
        current_text = query.strip()
        
        # ====== 1. ТРАНСЛИТЕРАЦИЯ ======
        transliterated, trans_details = self._transliterator.transliterate_text(current_text)
        was_transliterated = bool(trans_details)
        
        if was_transliterated:
            processing_steps.append(f"transliterate: {current_text} → {transliterated}")
            current_text = transliterated
        
        transliteration_details = [
            {"original": t.original, "result": t.result, "method": t.method}
            for t in trans_details
        ]
        
        # ====== 2. ИСПРАВЛЕНИЕ ОПЕЧАТОК ======
        spell_corrected, corrections = self._spell_corrector.correct_text(current_text)
        
        # Фильтруем по порогу уверенности
        valid_corrections = [
            c for c in corrections
            if c.confidence >= self._spell_threshold
        ]
        was_spell_corrected = bool(valid_corrections)
        
        if was_spell_corrected:
            processing_steps.append(f"spell_correct: {current_text} → {spell_corrected}")
            current_text = spell_corrected
        
        # ====== 3. НЕЧЁТКИЙ ПОИСК СУЩНОСТЕЙ ======
        fuzzy_matches: List[FuzzyMatch] = []
        detected_drugs: List[str] = []
        detected_symptoms: List[str] = []
        detected_diseases: List[str] = []
        detected_forms: List[str] = []
        
        if detect_entities:
            # Ищем препараты
            drug_matches = self._fuzzy_matcher.search_multi(current_text, category="drug")
            for match in drug_matches:
                if match.score >= self._fuzzy_threshold:
                    fuzzy_matches.append(match)
                    detected_drugs.append(match.matched)
            
            # Ищем симптомы
            symptom_matches = self._fuzzy_matcher.search_multi(current_text, category="symptom")
            for match in symptom_matches:
                if match.score >= self._fuzzy_threshold:
                    fuzzy_matches.append(match)
                    detected_symptoms.append(match.matched)
            
            # Ищем заболевания
            disease_matches = self._fuzzy_matcher.search_multi(current_text, category="disease")
            for match in disease_matches:
                if match.score >= self._fuzzy_threshold:
                    fuzzy_matches.append(match)
                    detected_diseases.append(match.matched)
            
            # Ищем формы выпуска
            form_matches = self._fuzzy_matcher.search_multi(current_text, category="dosage_form")
            for match in form_matches:
                if match.score >= self._fuzzy_threshold:
                    fuzzy_matches.append(match)
                    detected_forms.append(match.matched)
            
            if fuzzy_matches:
                entities_str = ", ".join([m.matched for m in fuzzy_matches[:5]])
                processing_steps.append(f"entities_detected: {entities_str}")
        
        # ====== 4. РАСШИРЕНИЕ СИНОНИМАМИ ======
        expansion: ExpansionResult | None = None
        should_expand = expand_synonyms if expand_synonyms is not None else self._enable_expansion
        
        if should_expand:
            expansion = self._synonym_expander.expand_query(current_text)
            
            if expansion.normalized_colloquial:
                processing_steps.append(
                    f"normalize_colloquial: {current_text} → {expansion.normalized_colloquial}"
                )
            
            if expansion.related_drugs:
                processing_steps.append(
                    f"related_drugs: {', '.join(expansion.related_drugs[:3])}"
                )
        
        # ====== ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ======
        # Вычисляем общую уверенность
        confidence = self._calculate_confidence(
            was_transliterated,
            valid_corrections,
            fuzzy_matches,
        )
        
        return NormalizationResult(
            original_query=query,
            transliterated=transliterated,
            spell_corrected=spell_corrected,
            normalized=current_text,
            was_transliterated=was_transliterated,
            transliteration_details=transliteration_details,
            was_spell_corrected=was_spell_corrected,
            spell_corrections=valid_corrections,
            fuzzy_matches=fuzzy_matches,
            expansion=expansion,
            detected_drugs=detected_drugs,
            detected_symptoms=detected_symptoms,
            detected_diseases=detected_diseases,
            detected_forms=detected_forms,
            confidence=confidence,
            processing_steps=processing_steps,
        )
    
    def _calculate_confidence(
        self,
        was_transliterated: bool,
        corrections: List[CorrectionResult],
        matches: List[FuzzyMatch],
    ) -> float:
        """Вычисляет общую уверенность в нормализации."""
        confidence = 1.0
        
        # Транслитерация немного снижает уверенность
        if was_transliterated:
            confidence *= 0.95
        
        # Коррекции снижают пропорционально их уверенности
        for correction in corrections:
            confidence *= correction.confidence
        
        # Fuzzy matches могут повысить уверенность
        if matches:
            avg_match_score = sum(m.score for m in matches) / len(matches)
            confidence = min(confidence, avg_match_score)
        
        return round(confidence, 3)
    
    def quick_normalize(self, query: str) -> str:
        """
        Быстрая нормализация — только транслитерация и spell correction.
        Возвращает только нормализованный текст.
        """
        result = self.normalize(query, expand_synonyms=False, detect_entities=False)
        return result.normalized
    
    def detect_drug(self, query: str) -> str | None:
        """
        Пытается найти название препарата в запросе.
        
        Returns:
            Нормализованное название или None
        """
        result = self.normalize(query)
        if result.detected_drugs:
            return result.detected_drugs[0]
        return None
    
    def detect_symptom(self, query: str) -> str | None:
        """
        Пытается найти симптом в запросе.
        
        Returns:
            Нормализованный симптом или None
        """
        result = self.normalize(query)
        if result.detected_symptoms:
            return result.detected_symptoms[0]
        return None
    
    def get_drug_alternatives(self, drug_name: str) -> List[str]:
        """
        Возвращает альтернативные названия препарата (другие бренды того же МНН).
        """
        _, alternatives = self._synonym_expander.get_brand_alternatives(drug_name)
        return alternatives


# Singleton instance
_query_normalizer: QueryNormalizer | None = None


def get_query_normalizer() -> QueryNormalizer:
    """Возвращает singleton экземпляр QueryNormalizer."""
    global _query_normalizer
    if _query_normalizer is None:
        _query_normalizer = QueryNormalizer()
    return _query_normalizer

