"""
FuzzyMatcher - нечёткий поиск по триггерам, препаратам и категориям.

Использует:
- N-грамм индексирование для быстрого поиска кандидатов
- Расстояние Левенштейна для ранжирования
- Токенизация для составных запросов
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .spell_corrector import levenshtein_distance, similarity_score


@dataclass
class FuzzyMatch:
    """Результат нечёткого совпадения."""
    query: str
    matched: str
    score: float
    match_type: str  # "exact", "fuzzy", "partial"
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_ngrams(text: str, n: int = 3) -> Set[str]:
    """Генерирует n-граммы из текста."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    """
    Вычисляет схожесть строк на основе n-грамм (Jaccard similarity).
    """
    if not s1 or not s2:
        return 0.0
    
    ngrams1 = generate_ngrams(s1, n)
    ngrams2 = generate_ngrams(s2, n)
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def tokenize(text: str) -> List[str]:
    """
    Токенизирует текст на слова.
    """
    # Убираем знаки препинания, разбиваем по пробелам
    text = re.sub(r'[^\w\s-]', ' ', text.lower())
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2]


class FuzzyMatcher:
    """
    Нечёткий поисковик по индексу термов.
    
    Особенности:
    - N-грамм индекс для быстрого нахождения кандидатов
    - Комбинированный скоринг (n-грамм + Левенштейн)
    - Поддержка составных запросов
    - Категории и метаданные
    """
    
    def __init__(
        self,
        ngram_size: int = 3,
        min_score: float = 0.5,
        max_results: int = 10,
    ):
        self._ngram_size = ngram_size
        self._min_score = min_score
        self._max_results = max_results
        
        # Основной индекс: term -> metadata
        self._index: Dict[str, Dict[str, Any]] = {}
        
        # N-грамм индекс: ngram -> set of terms
        self._ngram_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Индекс по категориям
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_term(
        self,
        term: str,
        category: str | None = None,
        metadata: Dict[str, Any] | None = None,
        aliases: List[str] | None = None,
    ) -> None:
        """
        Добавляет терм в индекс.
        
        Args:
            term: Основной терм (будет возвращаться при совпадении)
            category: Категория терма (для группировки)
            metadata: Дополнительные данные
            aliases: Альтернативные написания
        """
        term_lower = term.lower()
        
        # Сохраняем в основной индекс
        self._index[term_lower] = {
            "original": term,
            "category": category,
            "metadata": metadata or {},
            "aliases": [a.lower() for a in (aliases or [])],
        }
        
        # Индексируем n-граммы
        for ngram in generate_ngrams(term_lower, self._ngram_size):
            self._ngram_index[ngram].add(term_lower)
        
        # Индексируем алиасы
        for alias in (aliases or []):
            alias_lower = alias.lower()
            for ngram in generate_ngrams(alias_lower, self._ngram_size):
                self._ngram_index[ngram].add(term_lower)
        
        # Индексируем по категории
        if category:
            self._category_index[category.lower()].add(term_lower)
    
    def add_terms_bulk(self, terms: List[Dict[str, Any]]) -> None:
        """
        Массовое добавление термов.
        
        Args:
            terms: Список словарей с ключами: term, category, metadata, aliases
        """
        for item in terms:
            self.add_term(
                term=item["term"],
                category=item.get("category"),
                metadata=item.get("metadata"),
                aliases=item.get("aliases"),
            )
    
    def _get_candidates(self, query: str) -> Set[str]:
        """
        Получает кандидатов для проверки на основе n-грамм.
        """
        query_ngrams = generate_ngrams(query.lower(), self._ngram_size)
        candidates: Set[str] = set()
        
        for ngram in query_ngrams:
            if ngram in self._ngram_index:
                candidates.update(self._ngram_index[ngram])
        
        return candidates
    
    def _calculate_score(self, query: str, term: str, aliases: List[str]) -> Tuple[float, str]:
        """
        Вычисляет комбинированный скор и определяет тип совпадения.
        
        Returns:
            (score, match_type)
        """
        query_lower = query.lower()
        term_lower = term.lower()
        
        # 1. Точное совпадение
        if query_lower == term_lower:
            return 1.0, "exact"
        
        # Проверяем алиасы
        for alias in aliases:
            if query_lower == alias:
                return 0.98, "exact"
        
        # 2. Частичное вхождение
        if query_lower in term_lower or term_lower in query_lower:
            ratio = min(len(query_lower), len(term_lower)) / max(len(query_lower), len(term_lower))
            return 0.7 + (ratio * 0.25), "partial"
        
        # 3. Нечёткое совпадение
        # Комбинируем n-грамм и Левенштейн
        ngram_score = ngram_similarity(query_lower, term_lower, self._ngram_size)
        lev_score = similarity_score(query_lower, term_lower)
        
        # Проверяем алиасы и берём лучший скор
        best_alias_score = 0.0
        for alias in aliases:
            alias_ngram = ngram_similarity(query_lower, alias, self._ngram_size)
            alias_lev = similarity_score(query_lower, alias)
            alias_score = (alias_ngram * 0.4 + alias_lev * 0.6)
            best_alias_score = max(best_alias_score, alias_score)
        
        # Комбинированный скор: 40% n-грамм + 60% Левенштейн
        combined_score = max(
            ngram_score * 0.4 + lev_score * 0.6,
            best_alias_score,
        )
        
        return combined_score, "fuzzy"
    
    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> List[FuzzyMatch]:
        """
        Ищет совпадения для запроса.
        
        Args:
            query: Поисковый запрос
            category: Ограничить поиск категорией
            limit: Максимум результатов
        
        Returns:
            Список совпадений, отсортированный по score
        """
        limit = limit or self._max_results
        query_lower = query.lower().strip()
        
        if not query_lower:
            return []
        
        # Получаем кандидатов
        candidates = self._get_candidates(query_lower)
        
        # Если задана категория — фильтруем
        if category:
            category_terms = self._category_index.get(category.lower(), set())
            candidates = candidates & category_terms
        
        # Вычисляем скоры
        matches: List[FuzzyMatch] = []
        
        for term_lower in candidates:
            term_data = self._index.get(term_lower)
            if not term_data:
                continue
            
            score, match_type = self._calculate_score(
                query_lower,
                term_data["original"],
                term_data.get("aliases", []),
            )
            
            if score >= self._min_score:
                matches.append(FuzzyMatch(
                    query=query,
                    matched=term_data["original"],
                    score=score,
                    match_type=match_type,
                    metadata={
                        "category": term_data.get("category"),
                        **term_data.get("metadata", {}),
                    },
                ))
        
        # Сортируем по убыванию скора
        matches.sort(key=lambda m: m.score, reverse=True)
        
        return matches[:limit]
    
    def search_multi(
        self,
        query: str,
        category: str | None = None,
    ) -> List[FuzzyMatch]:
        """
        Ищет совпадения для составного запроса (несколько слов).
        
        Каждое слово проверяется отдельно.
        """
        tokens = tokenize(query)
        all_matches: List[FuzzyMatch] = []
        seen_terms: Set[str] = set()
        
        for token in tokens:
            matches = self.search(token, category=category)
            for match in matches:
                if match.matched.lower() not in seen_terms:
                    all_matches.append(match)
                    seen_terms.add(match.matched.lower())
        
        # Сортируем по убыванию скора
        all_matches.sort(key=lambda m: m.score, reverse=True)
        
        return all_matches[:self._max_results]
    
    def find_best_match(
        self,
        query: str,
        category: str | None = None,
    ) -> FuzzyMatch | None:
        """
        Возвращает лучшее совпадение или None.
        """
        matches = self.search(query, category=category, limit=1)
        return matches[0] if matches else None
    
    def get_by_category(self, category: str) -> List[str]:
        """Возвращает все термы в категории."""
        terms = self._category_index.get(category.lower(), set())
        return [self._index[t]["original"] for t in terms if t in self._index]
    
    @property
    def size(self) -> int:
        """Количество термов в индексе."""
        return len(self._index)


# Singleton instance с предзагруженными данными
_fuzzy_matcher: FuzzyMatcher | None = None


def get_fuzzy_matcher() -> FuzzyMatcher:
    """
    Возвращает singleton экземпляр FuzzyMatcher,
    предзагруженный с основными препаратами и триггерами.
    """
    global _fuzzy_matcher
    if _fuzzy_matcher is None:
        _fuzzy_matcher = FuzzyMatcher()
        _load_default_terms(_fuzzy_matcher)
    return _fuzzy_matcher


def _load_default_terms(matcher: FuzzyMatcher) -> None:
    """Загружает стандартный набор термов."""
    
    # Препараты (из spell_corrector)
    from .spell_corrector import DRUG_DICTIONARY
    
    for drug, variations in DRUG_DICTIONARY.items():
        matcher.add_term(
            term=drug,
            category="drug",
            aliases=variations,
            metadata={"type": "medication"},
        )
    
    # Симптомы
    symptoms = [
        ("головная боль", ["болит голова", "голова болит", "мигрень", "головная"]),
        ("боль в горле", ["горло болит", "болит горло", "першит", "больно глотать"]),
        ("кашель", ["кашляю", "кашля", "покашливание", "сухой кашель", "мокрый кашель"]),
        ("насморк", ["сопли", "заложен нос", "течёт из носа", "ринит"]),
        ("температура", ["жар", "лихорадка", "озноб", "высокая температура"]),
        ("боль в животе", ["живот болит", "болит живот", "колики", "спазмы"]),
        ("тошнота", ["тошнит", "мутит", "подташнивает"]),
        ("диарея", ["понос", "расстройство желудка", "жидкий стул"]),
        ("запор", ["не могу сходить", "проблемы со стулом"]),
        ("изжога", ["жжение в желудке", "кислота", "отрыжка"]),
        ("аллергия", ["аллергик", "аллергическая реакция", "зуд", "сыпь", "чешется"]),
        ("бессонница", ["не могу уснуть", "плохой сон", "не сплю"]),
        ("давление", ["высокое давление", "низкое давление", "гипертония", "гипотония"]),
        ("усталость", ["устал", "утомление", "слабость", "нет сил"]),
        ("стресс", ["нервы", "тревога", "беспокойство", "волнуюсь"]),
    ]
    
    for symptom, aliases in symptoms:
        matcher.add_term(
            term=symptom,
            category="symptom",
            aliases=aliases,
            metadata={"type": "symptom"},
        )
    
    # Заболевания
    diseases = [
        ("ОРВИ", ["орз", "простуда", "простыл", "простудился"]),
        ("грипп", ["гриппую", "грип"]),
        ("ангина", ["тонзиллит", "гнойная ангина"]),
        ("бронхит", ["бронхи", "воспаление бронхов"]),
        ("гастрит", ["воспаление желудка", "проблемы с желудком"]),
        ("цистит", ["воспаление мочевого", "больно писать"]),
        ("геморрой", ["геморой", "геморои"]),
        ("гипертония", ["высокое давление", "повышенное давление"]),
        ("диабет", ["сахарный диабет", "диабетик"]),
        ("астма", ["бронхиальная астма", "астматик"]),
    ]
    
    for disease, aliases in diseases:
        matcher.add_term(
            term=disease,
            category="disease",
            aliases=aliases,
            metadata={"type": "disease"},
        )
    
    # Формы выпуска
    forms = [
        ("таблетки", ["таблетка", "табл", "пилюли"]),
        ("капсулы", ["капсула", "капс"]),
        ("сироп", ["сиропа", "микстура"]),
        ("капли", ["капля", "капать"]),
        ("спрей", ["аэрозоль", "пшикалка"]),
        ("мазь", ["крем", "гель", "бальзам"]),
        ("свечи", ["суппозитории", "свеча"]),
        ("порошок", ["саше", "порошки"]),
        ("раствор", ["ампулы", "инъекции", "уколы"]),
    ]
    
    for form, aliases in forms:
        matcher.add_term(
            term=form,
            category="dosage_form",
            aliases=aliases,
            metadata={"type": "form"},
        )

