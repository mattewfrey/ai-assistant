"""
SynonymExpander - расширение запроса синонимами и связанными терминами.

Помогает:
- Находить связанные препараты (аналоги)
- Расширять симптомы связанными понятиями
- Добавлять торговые названия к МНН
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

# ============================================================================
# Словари синонимов
# ============================================================================

# Синонимы симптомов: симптом -> связанные термины
SYMPTOM_SYNONYMS: Dict[str, List[str]] = {
    "головная боль": ["мигрень", "болит голова", "голова раскалывается", "цефалгия"],
    "боль в горле": ["першение", "больно глотать", "фарингит", "ангина"],
    "кашель": ["покашливание", "бронхит", "трахеит", "отхаркивание"],
    "насморк": ["ринит", "заложенность носа", "сопли", "течёт из носа"],
    "температура": ["жар", "лихорадка", "гипертермия", "озноб"],
    "тошнота": ["рвота", "укачивание", "токсикоз"],
    "диарея": ["понос", "расстройство кишечника", "жидкий стул"],
    "запор": ["констипация", "проблемы со стулом", "затруднённая дефекация"],
    "изжога": ["гастрит", "рефлюкс", "кислотность"],
    "аллергия": ["крапивница", "зуд", "отёк", "гистамин"],
    "бессонница": ["инсомния", "нарушение сна", "не могу уснуть"],
    "усталость": ["астения", "слабость", "утомляемость", "упадок сил"],
    "боль в суставах": ["артрит", "артроз", "ревматизм", "ломота"],
    "боль в спине": ["остеохондроз", "радикулит", "люмбаго", "поясница болит"],
}

# МНН -> Торговые названия
INN_TO_BRANDS: Dict[str, List[str]] = {
    "парацетамол": ["панадол", "эффералган", "цефекон", "калпол"],
    "ибупрофен": ["нурофен", "миг", "ибуклин", "некст"],
    "ацетилсалициловая кислота": ["аспирин", "упсарин", "тромбо асс"],
    "метамизол натрия": ["анальгин", "баралгин", "темпалгин"],
    "дротаверин": ["но-шпа", "спазмол", "спазмонет"],
    "лоратадин": ["кларитин", "лорагексал", "кларидол"],
    "цетиризин": ["зиртек", "цетрин", "зодак"],
    "омепразол": ["омез", "лосек", "ультоп"],
    "амброксол": ["лазолван", "амбробене", "халиксол"],
    "ацетилцистеин": ["ацц", "флуимуцил", "викс актив"],
    "фенилэфрин": ["виброцил", "назол", "адрианол"],
    "ксилометазолин": ["отривин", "ксимелин", "галазолин", "снуп"],
    "оксиметазолин": ["називин", "назол", "африн"],
}

# Бренд -> МНН (обратный словарь)
BRAND_TO_INN: Dict[str, str] = {}
for inn, brands in INN_TO_BRANDS.items():
    for brand in brands:
        BRAND_TO_INN[brand.lower()] = inn

# Категории препаратов -> типичные запросы
CATEGORY_EXPANSIONS: Dict[str, List[str]] = {
    "жаропонижающие": ["от температуры", "снизить жар", "сбить температуру"],
    "обезболивающие": ["от боли", "болеутоляющие", "анальгетики"],
    "от кашля": ["противокашлевые", "муколитики", "отхаркивающие"],
    "от насморка": ["сосудосуживающие", "назальные", "для носа"],
    "антигистаминные": ["от аллергии", "противоаллергические"],
    "для желудка": ["антациды", "от изжоги", "гастропротекторы"],
    "витамины": ["поливитамины", "бады", "добавки"],
    "антибиотики": ["противомикробные", "антибактериальные"],
    "противовирусные": ["от вирусов", "иммуномодуляторы", "от гриппа"],
}

# Разговорные выражения -> нормализованные термины
COLLOQUIAL_TO_FORMAL: Dict[str, str] = {
    "от головы": "от головной боли",
    "от горла": "от боли в горле",
    "от живота": "от боли в животе",
    "от сердца": "сердечные",
    "от нервов": "успокоительные",
    "от давления": "антигипертензивные",
    "от простуды": "противопростудные",
    "от гриппа": "противовирусные",
    "для сна": "снотворные",
    "для памяти": "ноотропы",
    "для иммунитета": "иммуномодуляторы",
    "для печени": "гепатопротекторы",
    "для суставов": "хондропротекторы",
    "от прыщей": "против акне",
    "от грибка": "противогрибковые",
}


@dataclass
class ExpansionResult:
    """Результат расширения запроса."""
    original: str
    expanded_terms: List[str]
    related_drugs: List[str]
    inn_mappings: Dict[str, str]  # бренд -> МНН или МНН -> бренды
    normalized_colloquial: str | None
    
    def all_terms(self) -> List[str]:
        """Возвращает все термины (оригинал + расширения)."""
        terms = [self.original]
        terms.extend(self.expanded_terms)
        terms.extend(self.related_drugs)
        if self.normalized_colloquial:
            terms.append(self.normalized_colloquial)
        return list(set(terms))


class SynonymExpander:
    """
    Сервис расширения запросов синонимами.
    
    Возможности:
    - Расширение симптомов связанными терминами
    - Маппинг МНН <-> торговые названия
    - Нормализация разговорных выражений
    - Категориальное расширение
    """
    
    def __init__(
        self,
        symptom_synonyms: Dict[str, List[str]] | None = None,
        inn_to_brands: Dict[str, List[str]] | None = None,
        colloquial_map: Dict[str, str] | None = None,
    ):
        self._symptom_synonyms = symptom_synonyms or SYMPTOM_SYNONYMS
        self._inn_to_brands = inn_to_brands or INN_TO_BRANDS
        self._colloquial_map = colloquial_map or COLLOQUIAL_TO_FORMAL
        
        # Строим обратный индекс для симптомов
        self._reverse_symptom_index: Dict[str, str] = {}
        for main_symptom, synonyms in self._symptom_synonyms.items():
            self._reverse_symptom_index[main_symptom.lower()] = main_symptom
            for syn in synonyms:
                self._reverse_symptom_index[syn.lower()] = main_symptom
        
        # Строим обратный индекс для МНН
        self._brand_to_inn: Dict[str, str] = {}
        for inn, brands in self._inn_to_brands.items():
            for brand in brands:
                self._brand_to_inn[brand.lower()] = inn
    
    def expand_symptom(self, symptom: str) -> List[str]:
        """
        Расширяет симптом связанными терминами.
        """
        symptom_lower = symptom.lower()
        
        # Ищем основной симптом
        main_symptom = self._reverse_symptom_index.get(symptom_lower)
        if main_symptom:
            return self._symptom_synonyms.get(main_symptom, [])
        
        # Частичное совпадение
        for key, synonyms in self._symptom_synonyms.items():
            if symptom_lower in key.lower() or key.lower() in symptom_lower:
                return synonyms
        
        return []
    
    def get_brand_alternatives(self, drug_name: str) -> Tuple[str | None, List[str]]:
        """
        Находит альтернативы для препарата.
        
        Returns:
            (МНН, [список торговых названий])
        """
        drug_lower = drug_name.lower()
        
        # Если это бренд — находим МНН и другие бренды
        if drug_lower in self._brand_to_inn:
            inn = self._brand_to_inn[drug_lower]
            brands = self._inn_to_brands.get(inn, [])
            # Убираем текущий бренд из списка
            alternatives = [b for b in brands if b.lower() != drug_lower]
            return inn, alternatives
        
        # Если это МНН — возвращаем бренды
        if drug_lower in [inn.lower() for inn in self._inn_to_brands.keys()]:
            for inn, brands in self._inn_to_brands.items():
                if inn.lower() == drug_lower:
                    return inn, brands
        
        return None, []
    
    def normalize_colloquial(self, text: str) -> str | None:
        """
        Нормализует разговорное выражение.
        """
        text_lower = text.lower()
        
        for colloquial, formal in self._colloquial_map.items():
            if colloquial in text_lower:
                return formal
        
        return None
    
    def expand_query(self, query: str) -> ExpansionResult:
        """
        Полное расширение запроса.
        """
        query_lower = query.lower()
        expanded_terms: List[str] = []
        related_drugs: List[str] = []
        inn_mappings: Dict[str, str] = {}
        
        # 1. Расширяем симптомы
        symptom_expansions = self.expand_symptom(query)
        expanded_terms.extend(symptom_expansions)
        
        # 2. Ищем препараты и их альтернативы
        words = query_lower.split()
        for word in words:
            inn, brands = self.get_brand_alternatives(word)
            if inn:
                inn_mappings[word] = inn
                related_drugs.extend(brands)
        
        # 3. Нормализуем разговорные выражения
        normalized = self.normalize_colloquial(query)
        
        return ExpansionResult(
            original=query,
            expanded_terms=list(set(expanded_terms)),
            related_drugs=list(set(related_drugs)),
            inn_mappings=inn_mappings,
            normalized_colloquial=normalized,
        )
    
    def get_inn_for_brand(self, brand: str) -> str | None:
        """Возвращает МНН для торгового названия."""
        return self._brand_to_inn.get(brand.lower())
    
    def get_brands_for_inn(self, inn: str) -> List[str]:
        """Возвращает торговые названия для МНН."""
        return self._inn_to_brands.get(inn.lower(), [])


# Singleton instance
_synonym_expander: SynonymExpander | None = None


def get_synonym_expander() -> SynonymExpander:
    """Возвращает singleton экземпляр SynonymExpander."""
    global _synonym_expander
    if _synonym_expander is None:
        _synonym_expander = SynonymExpander()
    return _synonym_expander

