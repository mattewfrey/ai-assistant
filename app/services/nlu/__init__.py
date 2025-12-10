"""
NLU Enhancement Module - улучшенная обработка естественного языка.

Компоненты:
- SpellCorrector: исправление опечаток в названиях препаратов
- Transliterator: конвертация латиницы в кириллицу
- FuzzyMatcher: нечёткий поиск по триггерам и названиям
- SynonymExpander: расширение запроса синонимами
- QueryNormalizer: объединяющий сервис нормализации
"""

from .spell_corrector import SpellCorrector, get_spell_corrector
from .transliterator import Transliterator, transliterate
from .fuzzy_matcher import FuzzyMatcher, get_fuzzy_matcher
from .synonym_expander import SynonymExpander, get_synonym_expander
from .query_normalizer import QueryNormalizer, get_query_normalizer

__all__ = [
    "SpellCorrector",
    "get_spell_corrector",
    "Transliterator", 
    "transliterate",
    "FuzzyMatcher",
    "get_fuzzy_matcher",
    "SynonymExpander",
    "get_synonym_expander",
    "QueryNormalizer",
    "get_query_normalizer",
]

