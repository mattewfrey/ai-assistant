"""
SpellCorrector - исправление опечаток в названиях препаратов.

Использует:
- Словарь популярных препаратов
- Расстояние Левенштейна для fuzzy matching
- Фонетическое сходство для русского языка
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

# ============================================================================
# Словарь препаратов с вариациями написания
# ============================================================================

# Формат: "каноническое_название": ["вариации", "опечатки", "сокращения"]
DRUG_DICTIONARY: Dict[str, List[str]] = {
    # Анальгетики и жаропонижающие
    "нурофен": ["нурафен", "нурафэн", "нурофэн", "нуровен", "нурофін"],
    "парацетамол": ["парацитамол", "парацетомол", "поросетамол", "парацетамал", "парацетомал"],
    "ибупрофен": ["ибупрафен", "ібупрофен", "ибупрофін", "ибуфен", "ибупрафін"],
    "анальгин": ["аналгин", "анальгін", "аналгін", "ональгин"],
    "аспирин": ["аспирін", "оспирин", "аспирен", "оспірін"],
    "цитрамон": ["цитромон", "цітрамон", "цитрамін", "ситрамон"],
    "но-шпа": ["ношпа", "но шпа", "нош-па", "ношпа", "но-шпу", "ношпу"],
    "кеторол": ["кетарол", "кеторал", "кеторолак", "кетонал"],
    "найз": ["найс", "наиз", "наіз", "найз гель"],
    "пенталгин": ["пенталгін", "пентальгин", "пенталген", "пентальгін"],
    
    # Противовирусные и от простуды
    "терафлю": ["терафлу", "терафлю", "тирафлю", "тэрафлю", "терофлю"],
    "колдрекс": ["колдрекс", "калдрекс", "колдрэкс", "колдрікс"],
    "антигриппин": ["антигрипин", "антігріппін", "антигрипін", "анти-гриппин"],
    "арбидол": ["арбідол", "орбидол", "арбидал", "орбідол"],
    "ингавирин": ["інгавірин", "ингаверин", "інгавірін", "ингавирен"],
    "кагоцел": ["когоцел", "кагацел", "кагоцєл", "когацел"],
    "оциллококцинум": ["оцилококцинум", "оцилокукцинум", "осцилококцинум", "оцилококцінум"],
    "анаферон": ["онаферон", "анаферан", "анофєрон"],
    "эргоферон": ["ергоферон", "эргоферан", "єргоферон"],
    
    # От кашля
    "лазолван": ["лозолван", "лазалван", "лозалван", "лазолвон"],
    "амброксол": ["амбраксол", "амброксал", "амброгексал"],
    "бромгексин": ["бромгексін", "брамгексин", "бромгіксін"],
    "мукалтин": ["мукалтін", "мукальтин", "мукалтен"],
    "синекод": ["сінекод", "сінікод", "синикод"],
    "коделак": ["каделак", "кодєлак", "коделок"],
    "геделикс": ["гедєлікс", "геделікс", "гєдєлікс"],
    "доктор мом": ["доктор мам", "др мом", "дрмом"],
    "аскорил": ["оскорил", "аскоріл", "оскоріл"],
    "ренгалин": ["рєнгалін", "ренголін", "рингалин"],
    
    # От боли в горле
    "стрепсилс": ["стрепсілс", "стрепсилз", "стрепсилс", "стрэпсилс"],
    "граммидин": ["грамидин", "граммідін", "грамідін"],
    "лизобакт": ["лізобакт", "лизобак", "лізобак"],
    "фарингосепт": ["форингосепт", "фарінгосепт", "форінгосепт"],
    "гексорал": ["гексарал", "гєксорал", "гєксарал"],
    "тантум верде": ["тантумверде", "тантум-верде", "тантум", "тонтум верде"],
    "ингалипт": ["інгаліпт", "ингаліпт", "інголіпт"],
    "каметон": ["каметан", "каметон", "комєтон"],
    
    # От насморка
    "називин": ["нозівін", "називін", "називен"],
    "отривин": ["отрівін", "отривен", "отрівен"],
    "ксимелин": ["ксімелін", "ксимелен", "ксімілін"],
    "снуп": ["снуп", "снап", "snup"],
    "виброцил": ["вібрацил", "вибрацил", "віброцил"],
    "аквамарис": ["аква марис", "аква-марис", "аквамаріс", "акваморис"],
    "ринофлуимуцил": ["ріноффлуімуцил", "ринофлуимуцел", "рінофлуімуцил"],
    "полидекса": ["полідекса", "полидекс", "полідєкса"],
    "изофра": ["ізофра", "изофро", "ізофро"],
    
    # Антигистаминные
    "супрастин": ["супрастін", "супростин", "супрастен"],
    "цетрин": ["цетрін", "цітрін", "цетрен"],
    "зиртек": ["зіртек", "зыртек", "зіртєк"],
    "кларитин": ["кларітін", "клоритин", "кларітен"],
    "лоратадин": ["лоратодин", "лоратадін", "лоратодін"],
    "зодак": ["зодок", "зодак", "задак"],
    "эриус": ["еріус", "эріус", "єріус"],
    "тавегил": ["товегил", "тавєгіл", "товєгіл"],
    
    # ЖКТ
    "мезим": ["мізім", "мезім", "мезем"],
    "фестал": ["фестол", "фєстал", "фістал"],
    "панкреатин": ["панкреотин", "панкреатін", "понкреатин"],
    "омепразол": ["омепрозол", "омепразал", "омепрозал"],
    "де-нол": ["денол", "де нол", "дінол"],
    "фосфалюгель": ["фосфолюгель", "фасфалюгель", "фосфалюгєль"],
    "альмагель": ["алмагель", "альмогель", "олмагель"],
    "смекта": ["смєкта", "смекто", "смєкто"],
    "энтеросгель": ["ентеросгель", "энтеросгєль", "інтеросгель"],
    "линекс": ["лінекс", "линекс", "лінєкс"],
    "хилак форте": ["хилак-форте", "хілак форте", "хилакфорте"],
    "эспумизан": ["еспумізан", "эспумізан", "испумизан"],
    "имодиум": ["імодіум", "имодіум", "імодіум"],
    "лоперамид": ["лоперамід", "лопєрамід", "лопирамид"],
    "мотилиум": ["мотіліум", "мотиліум", "мотілум"],
    
    # Сердечно-сосудистые
    "валидол": ["валідол", "волидол", "валідал"],
    "корвалол": ["корволол", "карвалол", "корвалал"],
    "валокордин": ["валокардин", "волокордин", "валокардін"],
    "нитроглицерин": ["нітрогліцерин", "нитроглицирин", "нітроглицєрін"],
    "каптоприл": ["коптоприл", "каптапріл", "коптопріл"],
    "эналаприл": ["еналаприл", "эналопріл", "єналаприл"],
    "амлодипин": ["амлодіпін", "омлодипин", "амлодіпен"],
    "конкор": ["конкар", "конкор", "канкор"],
    "кардиомагнил": ["кардіомагніл", "кардиомагнел", "кордиомагнил"],
    
    # Витамины и БАДы
    "компливит": ["комплівіт", "компливет", "комплівєт"],
    "витрум": ["вітрум", "витром", "вітром"],
    "супрадин": ["супрадін", "супродин", "супродін"],
    "центрум": ["цєнтрум", "центром", "цєнтром"],
    "алфавит": ["алфовит", "альфавіт", "олфавит"],
    "аевит": ["оєвіт", "аевіт", "оєвит"],
    "ревит": ["рєвіт", "ревіт", "рівіт"],
    
    # Антибиотики (часто ищут)
    "амоксициллин": ["амоксицилин", "амоксіцилін", "омоксициллин"],
    "азитромицин": ["азітроміцин", "озитромицин", "азитромицен"],
    "цефтриаксон": ["цефтріаксон", "цєфтріаксон", "цефтреаксон"],
    "флемоксин": ["флємоксін", "флимоксин", "флємоксин"],
    "сумамед": ["сумомед", "сумамєд", "сумомєд"],
    "аугментин": ["аугментін", "огментин", "аугмєнтін"],
    
    # Другие популярные
    "мирамистин": ["міромистін", "мирамістін", "мірамистин"],
    "хлоргексидин": ["хлоргексідін", "хларгексидин", "хларгєксідін"],
    "перекись водорода": ["перекис водорода", "пірикис", "перекис"],
    "йод": ["іод", "йодом", "іодом"],
    "зелёнка": ["зелёнка", "зеленка", "бриллиантовый зеленый"],
    "бинт": ["бінт", "бент"],
    "вата": ["вато", "вату"],
    "пластырь": ["пластир", "пластырь", "лейкопластырь"],
}

# Обратный индекс: вариация -> каноническое название
_REVERSE_INDEX: Dict[str, str] = {}
for canonical, variations in DRUG_DICTIONARY.items():
    _REVERSE_INDEX[canonical.lower()] = canonical
    for var in variations:
        _REVERSE_INDEX[var.lower()] = canonical


@dataclass
class CorrectionResult:
    """Результат коррекции."""
    original: str
    corrected: str
    confidence: float
    was_corrected: bool
    suggestions: List[Tuple[str, float]] = None  # [(слово, score), ...]
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


def levenshtein_distance(s1: str, s2: str) -> int:
    """Вычисляет расстояние Левенштейна между двумя строками."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Стоимость вставки, удаления, замены
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def similarity_score(s1: str, s2: str) -> float:
    """
    Вычисляет схожесть строк (0.0 - 1.0).
    1.0 = идентичные, 0.0 = совершенно разные.
    """
    if not s1 or not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1.lower(), s2.lower())
    return 1.0 - (distance / max_len)


# Фонетические группы для русского языка
PHONETIC_GROUPS = {
    # Гласные
    frozenset("аоу"): "а",
    frozenset("еёэи"): "е",
    frozenset("ыи"): "и",
    # Согласные
    frozenset("бп"): "п",
    frozenset("вф"): "ф", 
    frozenset("гк"): "к",
    frozenset("дт"): "т",
    frozenset("жш"): "ш",
    frozenset("зс"): "с",
    # Мягкий/твёрдый знак часто опускают
    frozenset("ьъ"): "",
}


def phonetic_key(word: str) -> str:
    """
    Создаёт фонетический ключ для слова (упрощённый русский soundex).
    Помогает находить слова, звучащие похоже.
    """
    word = word.lower().strip()
    result = []
    prev_char = ""
    
    for char in word:
        # Заменяем на базовый звук
        replaced = char
        for group, replacement in PHONETIC_GROUPS.items():
            if char in group:
                replaced = replacement
                break
        
        # Убираем дубликаты
        if replaced and replaced != prev_char:
            result.append(replaced)
            prev_char = replaced
    
    return "".join(result)


class SpellCorrector:
    """
    Сервис исправления опечаток в названиях препаратов.
    
    Использует комбинацию методов:
    1. Точное совпадение с вариациями
    2. Расстояние Левенштейна
    3. Фонетическое сходство
    """
    
    def __init__(
        self,
        drug_dictionary: Dict[str, List[str]] | None = None,
        min_similarity: float = 0.7,
        max_distance: int = 3,
    ):
        self._dictionary = drug_dictionary or DRUG_DICTIONARY
        self._min_similarity = min_similarity
        self._max_distance = max_distance
        
        # Строим индексы
        self._reverse_index: Dict[str, str] = {}
        self._phonetic_index: Dict[str, Set[str]] = {}
        self._all_canonical: Set[str] = set()
        
        self._build_indices()
    
    def _build_indices(self) -> None:
        """Строит индексы для быстрого поиска."""
        for canonical, variations in self._dictionary.items():
            canonical_lower = canonical.lower()
            self._all_canonical.add(canonical_lower)
            self._reverse_index[canonical_lower] = canonical
            
            # Фонетический индекс
            pk = phonetic_key(canonical)
            if pk not in self._phonetic_index:
                self._phonetic_index[pk] = set()
            self._phonetic_index[pk].add(canonical_lower)
            
            for var in variations:
                var_lower = var.lower()
                self._reverse_index[var_lower] = canonical
                
                pk = phonetic_key(var)
                if pk not in self._phonetic_index:
                    self._phonetic_index[pk] = set()
                self._phonetic_index[pk].add(canonical_lower)
    
    def correct_word(self, word: str) -> CorrectionResult:
        """
        Исправляет одно слово (предположительно название препарата).
        """
        word_lower = word.lower().strip()
        
        # 1. Точное совпадение
        if word_lower in self._reverse_index:
            canonical = self._reverse_index[word_lower]
            return CorrectionResult(
                original=word,
                corrected=canonical,
                confidence=1.0,
                was_corrected=(word_lower != canonical.lower()),
            )
        
        # 2. Поиск по фонетическому ключу
        pk = phonetic_key(word_lower)
        candidates: List[Tuple[str, float]] = []
        
        if pk in self._phonetic_index:
            for candidate in self._phonetic_index[pk]:
                score = similarity_score(word_lower, candidate)
                if score >= self._min_similarity:
                    candidates.append((self._reverse_index[candidate], score))
        
        # 3. Если фонетика не помогла — полный перебор с Левенштейном
        if not candidates:
            for canonical in self._all_canonical:
                distance = levenshtein_distance(word_lower, canonical)
                if distance <= self._max_distance:
                    score = similarity_score(word_lower, canonical)
                    if score >= self._min_similarity:
                        candidates.append((self._reverse_index[canonical], score))
        
        # Сортируем по убыванию score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            best_match, best_score = candidates[0]
            return CorrectionResult(
                original=word,
                corrected=best_match,
                confidence=best_score,
                was_corrected=True,
                suggestions=candidates[:5],
            )
        
        # Не нашли подходящего — возвращаем как есть
        return CorrectionResult(
            original=word,
            corrected=word,
            confidence=0.0,
            was_corrected=False,
        )
    
    def correct_text(self, text: str) -> Tuple[str, List[CorrectionResult]]:
        """
        Исправляет опечатки в тексте.
        
        Returns:
            (исправленный_текст, список_коррекций)
        """
        # Разбиваем на слова, сохраняя позиции
        words = re.findall(r'[\w-]+', text, re.UNICODE)
        corrections: List[CorrectionResult] = []
        result_text = text
        
        for word in words:
            if len(word) < 3:  # Слишком короткие слова не проверяем
                continue
            
            correction = self.correct_word(word)
            if correction.was_corrected and correction.confidence >= self._min_similarity:
                corrections.append(correction)
                # Заменяем в тексте (case-insensitive)
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                result_text = pattern.sub(correction.corrected, result_text, count=1)
        
        return result_text, corrections
    
    def get_suggestions(self, word: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Возвращает список предложений для слова.
        """
        correction = self.correct_word(word)
        return correction.suggestions[:limit]
    
    def add_drug(self, canonical: str, variations: List[str]) -> None:
        """Добавляет новый препарат в словарь."""
        self._dictionary[canonical] = variations
        self._build_indices()  # Перестраиваем индексы


# Singleton instance
_spell_corrector: SpellCorrector | None = None


def get_spell_corrector() -> SpellCorrector:
    """Возвращает singleton экземпляр SpellCorrector."""
    global _spell_corrector
    if _spell_corrector is None:
        _spell_corrector = SpellCorrector()
    return _spell_corrector

