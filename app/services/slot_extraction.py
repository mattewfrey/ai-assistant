"""
Продвинутый модуль извлечения сущностей (слотов) из русского текста.

Поддерживает:
- Возраст (с учетом разных формулировок)
- Цены и ценовые диапазоны  
- Симптомы и болезни
- Формы выпуска лекарств
- Детский/взрослый контекст
- Фильтры (без сахара, без лактозы, etc.)
- Нормализацию текста для русской морфологии
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ============================================================================
# Предобработка текста
# ============================================================================

def normalize_text(text: str) -> str:
    """Нормализует текст: приводит к нижнему регистру, убирает лишние пробелы, заменяет ё на е."""
    result = text.lower().strip()
    result = result.replace("ё", "е")
    # Убираем повторяющиеся пробелы
    result = re.sub(r"\s+", " ", result)
    # Убираем знаки препинания в конце для лучшего матчинга
    result = result.rstrip(".,!?;:")
    return result


# ============================================================================
# Паттерны для извлечения возраста
# ============================================================================

# Расширенные паттерны для возраста
AGE_PATTERNS = [
    # "мне 30 лет", "ему 5 лет", "ей 42 года"
    re.compile(r"(?:мне|ему|ей|нам|им|себе|человеку|пациенту)\s+(?P<age>\d{1,3})\s*(?:лет|года?|годик)", re.IGNORECASE),
    # "ребенку 5 лет", "малышу 3 года", "сыну 10 лет", "дочке 7 лет"
    re.compile(r"(?:ребенк[уа]|малыш[уа]|сын[уа]|дочк[еи]|дочер[иь]|внук[уа]|внучк[еи]|бабушк[еи]|дедушк[еи]|мам[еы]|пап[еы])\s+(?P<age>\d{1,3})\s*(?:лет|года?|годик)?", re.IGNORECASE),
    # "возраст 30", "возраст: 30 лет"
    re.compile(r"возраст[:\s]+(?P<age>\d{1,3})\s*(?:лет|года?|годик)?", re.IGNORECASE),
    # "для ребенка 5 лет", "для взрослого 30 лет"
    re.compile(r"для\s+(?:ребенка|ребёнка|малыша|детей|взрослого)\s+(?P<age>\d{1,3})\s*(?:лет|года?)?", re.IGNORECASE),
    # "5 лет" в контексте ответа на вопрос о возрасте (короткий ответ)
    re.compile(r"^(?P<age>\d{1,2})\s*(?:лет|года?|годик)?\s*$", re.IGNORECASE),
    # "нам 5", "мне 30" (без "лет")
    re.compile(r"(?:мне|нам|ему|ей)\s+(?P<age>\d{1,2})(?:\s|$|,)", re.IGNORECASE),
    # "ребенок 5 лет", "взрослый 30 лет"
    re.compile(r"(?:ребенок|ребёнок|малыш|взрослый|подросток)\s+(?P<age>\d{1,3})\s*(?:лет|года?)?", re.IGNORECASE),
]

# Паттерн для простого числа - используется как fallback
SIMPLE_AGE_PATTERN = re.compile(r"\b(?P<age>\d{1,2})\b")


# ============================================================================
# Паттерны для цены и ценовых диапазонов
# ============================================================================

PRICE_MAX_PATTERNS = [
    # "до 500 рублей", "до 500р", "до 500₽"
    re.compile(r"(?:до|максимум|не\s+дороже|не\s+больше|в\s+пределах)\s*(?P<price>\d{2,6})\s*(?:р(?:уб(?:лей)?)?|₽)?", re.IGNORECASE),
    # "бюджет 500", "бюджет до 500"
    re.compile(r"бюджет\s*(?:до\s*)?(?P<price>\d{2,6})\s*(?:р(?:уб(?:лей)?)?|₽)?", re.IGNORECASE),
    # "500р максимум", "500 рублей максимум"
    re.compile(r"(?P<price>\d{2,6})\s*(?:р(?:уб(?:лей)?)?|₽)?\s*максимум", re.IGNORECASE),
    # "подешевле", "недорого" - специальные маркеры для фильтрации (задаем дефолт)
]

PRICE_MIN_PATTERNS = [
    # "от 500 рублей", "от 500р"
    re.compile(r"(?:от|минимум|не\s+дешевле|не\s+меньше)\s*(?P<price>\d{2,6})\s*(?:р(?:уб(?:лей)?)?|₽)?", re.IGNORECASE),
]

PRICE_RANGE_PATTERN = re.compile(
    r"(?:от\s*)?(?P<min>\d{2,6})\s*(?:[-–—до]\s*|до\s+)(?P<max>\d{2,6})\s*(?:р(?:уб(?:лей)?)?|₽)?",
    re.IGNORECASE
)

# Маркеры бюджетного поиска
BUDGET_MARKERS = ["подешевле", "недорого", "дешевый", "бюджетный", "эконом"]
DEFAULT_BUDGET_PRICE = 500


# ============================================================================
# Симптомы - расширенный словарь с основами слов
# ============================================================================

# Словарь симптомов: основа слова -> нормализованное название
SYMPTOM_STEMS: Dict[str, str] = {
    # Боль
    "голов": "головная боль",
    "головн": "головная боль",
    "мигрен": "мигрень",
    # Горло
    "горл": "боль в горле",
    "ангин": "ангина",
    "глотат": "боль при глотании",
    # Кашель
    "кашл": "кашель",
    "кашел": "кашель",
    "сух": "сухой кашель",  # в контексте кашля
    "мокрот": "влажный кашель",
    # Насморк
    "насморк": "насморк",
    "ринит": "ринит",
    "сопл": "насморк",
    "нос зало": "заложенность носа",
    "заложен": "заложенность носа",
    # Температура
    "температур": "температура",
    "жар": "жар",
    "лихорад": "лихорадка",
    "озноб": "озноб",
    # Живот
    "живот": "боль в животе",
    "желудок": "боль в желудке",
    "желудоч": "желудочные проблемы",
    "тошнот": "тошнота",
    "рвот": "рвота",
    "изжог": "изжога",
    "диаре": "диарея",
    "понос": "диарея",
    "запор": "запор",
    "вздут": "вздутие",
    "метеоризм": "метеоризм",
    # Слабость
    "слабост": "слабость",
    "устал": "усталость",
    "утомля": "утомляемость",
    # Суставы и мышцы
    "суста": "боль в суставах",
    "мышц": "мышечная боль",
    "спин": "боль в спине",
    "поясниц": "боль в пояснице",
    "шея": "боль в шее",
    "шей": "боль в шее",
    "ломот": "ломота",
    # Аллергия
    "аллерг": "аллергия",
    "зуд": "зуд",
    "чеш": "зуд",
    "чихан": "чихание",
    "сыпь": "сыпь",
    "крапивниц": "крапивница",
    "отек": "отёк",
    # Глаза
    "глаз": "проблемы с глазами",
    "конъюнктив": "конъюнктивит",
    "слезотеч": "слезотечение",
    "красн": "покраснение",  # глаз, кожи
    # Ухо
    "ух": "боль в ухе",
    "отит": "отит",
    # Сон
    "бессонниц": "бессонница",
    "не сп": "проблемы со сном",
    "сон": "проблемы со сном",
    # Давление
    "давлен": "давление",
    "гипертон": "гипертония",
    "гипотон": "гипотония",
    # Сердце
    "серд": "проблемы с сердцем",
    "аритми": "аритмия",
    "тахикард": "тахикардия",
    # Дыхание
    "дыш": "затрудненное дыхание",
    "одышк": "одышка",
    "задых": "одышка",
    # Кожа
    "кож": "проблемы с кожей",
    "дерматит": "дерматит",
    "экзем": "экзема",
    "прыщ": "прыщи",
    "акне": "акне",
    # Нервы
    "нерв": "нервозность",
    "тревог": "тревога",
    "стресс": "стресс",
    "раздраж": "раздражительность",
    # Простуда общая
    "простуд": "простуда",
    "орви": "ОРВИ",
    "орз": "ОРЗ",
    "гриб": "грибок",  # грибок, не грипп
    "грипп": "грипп",
    "грип": "грипп",
}

# Заболевания - отдельный словарь
DISEASE_STEMS: Dict[str, str] = {
    "орви": "ОРВИ",
    "орз": "ОРЗ",
    "грипп": "грипп",
    "грип": "грипп",
    "ангин": "ангина",
    "гастрит": "гастрит",
    "язв": "язва",
    "панкреатит": "панкреатит",
    "холецистит": "холецистит",
    "диабет": "диабет",
    "гипертон": "гипертония",
    "астм": "астма",
    "бронхит": "бронхит",
    "пневмон": "пневмония",
    "цистит": "цистит",
    "артрит": "артрит",
    "артроз": "артроз",
    "остеохондроз": "остеохондроз",
    "геморро": "геморрой",
    "варикоз": "варикоз",
    "аллерг": "аллергия",
    "дерматит": "дерматит",
    "экзем": "экзема",
    "псориаз": "псориаз",
    "герпес": "герпес",
    "молочниц": "молочница",
    "кандидоз": "кандидоз",
    "грибок": "грибок",
    "микоз": "микоз",
    "подагр": "подагра",
    "ревматизм": "ревматизм",
    "мигрен": "мигрень",
    "невралг": "невралгия",
    "радикулит": "радикулит",
    "простатит": "простатит",
    "запор": "запор",
    "геморрой": "геморрой",
    "синусит": "синусит",
    "гайморит": "гайморит",
    "фарингит": "фарингит",
    "ларингит": "ларингит",
    "тонзиллит": "тонзиллит",
    "отит": "отит",
    "конъюнктивит": "конъюнктивит",
    "стоматит": "стоматит",
}


# ============================================================================
# Формы выпуска - расширенный словарь
# ============================================================================

DEFAULT_DOSAGE_FORM_KEYWORDS: Dict[str, List[str]] = {
    "tablets": ["таблет", "табл.", "табл", "пилюл"],
    "capsules": ["капсул"],
    "syrup": ["сироп"],
    "spray": ["спрей"],
    "drops": ["капли", "капел"],
    "powder": ["порошок", "порошк", "саше"],
    "cream": ["крем"],
    "ointment": ["мазь", "мази"],
    "gel": ["гель", "геля"],
    "solution": ["раствор"],
    "suspension": ["суспензи"],
    "injection": ["ампул", "уколы", "укол", "инъекци"],
    "suppository": ["свечи", "свеч", "суппозитор"],
    "patch": ["пластыр"],
    "lozenges": ["пастилк", "леденц"],
    "granules": ["гранул"],
    "emulsion": ["эмульси"],
}


# ============================================================================
# Фильтры и специальные маркеры
# ============================================================================

# Маркеры для детских препаратов
CHILDREN_MARKERS = [
    "для ребенка", "для ребёнка", "ребенку", "ребёнку",
    "ребенок", "ребёнок",  # Одиночные слова
    "для детей", "детям", "детский", "детское", "детская",
    "малышу", "малыша", "для малыша", "для малышей",
    "малыш",  # Одиночное слово
    "сыну", "дочке", "дочери", "внуку", "внучке",
    "грудничку", "грудничок", "новорожденному", "младенцу",
]

# Маркеры для подростков
TEENAGER_MARKERS = [
    "подростку", "подростка", "для подростка", "подросток",
    "тинейджер", "школьнику", "школьника", "для школьника",
]

# Маркеры для взрослых
ADULT_MARKERS = [
    "для взрослого", "взрослому", "взрослый",
    "мне", "себе",
]

# Маркеры для пожилых
ELDERLY_MARKERS = [
    "пожилому", "пожилого", "для пожилого", "пожилой",
    "бабушке", "бабушки", "для бабушки",
    "дедушке", "дедушки", "для дедушки",
    "престарелому", "престарелого",
    "старику", "старика", "старушке",
]

# Возрастные группы (enum-like)
class AgeGroup:
    CHILD = "child"       # 0-12 лет
    TEENAGER = "teenager" # 13-17 лет
    ADULT = "adult"       # 18-59 лет
    ELDERLY = "elderly"   # 60+ лет

# Диапазоны возраста для каждой группы
AGE_GROUP_RANGES = {
    AgeGroup.CHILD: (0, 12),
    AgeGroup.TEENAGER: (13, 17),
    AgeGroup.ADULT: (18, 59),
    AgeGroup.ELDERLY: (60, 120),
}

# Специальные фильтры
SPECIAL_FILTERS: Dict[str, List[str]] = {
    "sugar_free": ["без сахара", "бессахарный", "несладкий", "sugar-free", "sugar free"],
    "lactose_free": ["без лактозы", "безлактозный", "lactose-free", "lactose free"],
    "gluten_free": ["без глютена", "безглютеновый", "gluten-free", "gluten free"],
    "alcohol_free": ["без спирта", "безалкогольный", "без алкоголя"],
    "dye_free": ["без красителей", "без красител"],
    "natural": ["натуральный", "природный", "растительный", "на травах", "травяной"],
    "homeopathic": ["гомеопатический", "гомеопатия"],
    "generic": ["дженерик", "аналог", "замена"],
}

# Маркеры беременности/лактации
PREGNANCY_MARKERS = [
    "беременн", "при беременности", "беременным",
    "в положении", "жду ребенка", "жду ребёнка",
]

BREASTFEEDING_MARKERS = [
    "кормящ", "при лактации", "кормлю грудью", "грудное вскармливание",
    "гв", "на гв", "кормящей маме",
]

# Статусы беременности для slot
class PregnancyStatus:
    PREGNANT = "pregnant"
    BREASTFEEDING = "breastfeeding"
    NONE = "none"

# Маркеры хронических заболеваний
CHRONIC_CONDITIONS_MARKERS: Dict[str, List[str]] = {
    "diabetes": ["диабет", "диабетик", "сахарный диабет", "инсулин"],
    "hypertension": ["гипертон", "давление высокое", "гипертензи", "повышенное давление"],
    "asthma": ["астм", "бронхиальная астма"],
    "heart_disease": [
        "сердечн", "сердц", "аритми", "стенокарди", "инфаркт",
        "проблемы с сердцем", "больное сердце",
    ],
    "kidney_disease": [
        "почки", "почечн", "нефрит", "пиелонефрит",
        "проблемы с почками", "больные почки",
    ],
    "liver_disease": ["печен", "гепатит", "цирроз", "проблемы с печенью"],
    "stomach_ulcer": ["язва желудка", "язвенн", "гастрит", "проблемы с жкт", "проблемы с желудком"],
    "glaucoma": ["глауком"],
    "thyroid": ["щитовид", "тиреоидит", "гипотиреоз", "гипертиреоз"],
}

# Маркеры аллергий на лекарства
ALLERGY_MARKERS = [
    "аллергия на", "аллергик", "непереносимость",
    "реакция на", "не переношу", "нельзя принимать",
]

# Маркеры отсутствия аллергий/противопоказаний
NO_ALLERGY_MARKERS = [
    "аллергии нет", "нет аллергии", "не аллергик",
    "без аллергии", "аллергий нет",
]

NO_CHRONIC_MARKERS = [
    "нет хронических", "хронических нет", "ничем не болею",
    "здоров", "противопоказаний нет", "нет противопоказаний",
]

NO_PREGNANCY_MARKERS = [
    "не беременна", "не кормлю", "не беременность",
    "ни то, ни другое", "ни то ни другое",
]

# Длительность симптомов
SYMPTOM_DURATION_MARKERS: Dict[str, List[str]] = {
    "today": ["сегодня", "только что", "час назад", "с утра", "недавно"],
    "few_days": ["пару дней", "несколько дней", "2 дня", "3 дня", "два дня", "три дня", "вчера"],
    "week": ["неделю", "неделя", "7 дней", "семь дней", "дней 5", "пять дней"],
    "long": ["давно", "долго", "больше недели", "две недели", "месяц", "хронически"],
}

# Выраженность симптомов (порядок важен - сначала проверяем severe, потом mild)
SYMPTOM_SEVERITY_MARKERS: Dict[str, List[str]] = {
    # severe проверяется первым, чтобы "нестерпимо" не совпало с "терпимо"
    "severe": [
        "сильно", "очень сильн", "невыносим", "ужасно", "адская", "адский",
        "резко", "острая", "острый", "нестерпим", "невозможн", "дикая", "дикий",
    ],
    "moderate": ["умеренн", "средн", "ощутим", "заметн"],
    "mild": ["слабо", "немного", "чуть-чуть", "слегка", "несильно", "терпимо"],
}

# Маркеры OTC (без рецепта)
OTC_MARKERS = [
    "без рецепта", "безрецептурн", "otc", "свободная продажа",
]

# Маркеры рецептурных
PRESCRIPTION_MARKERS = [
    "рецептурн", "по рецепту", "rx", "с рецептом",
]


# ============================================================================
# Паттерны для симптомов в контексте
# ============================================================================

SYMPTOM_CONTEXT_PATTERNS = [
    # "от головы", "от кашля", "от боли"
    re.compile(r"от\s+(?P<symptom>[а-яё]+(?:\s+[а-яё]+)?)", re.IGNORECASE),
    # "при головной боли", "при кашле"
    re.compile(r"при\s+(?P<symptom>[а-яё]+(?:\s+[а-яё]+)?(?:\s+боли)?)", re.IGNORECASE),
    # "против кашля", "против насморка"
    re.compile(r"против\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
    # "если болит голова", "когда болит горло"
    re.compile(r"(?:если|когда)\s+болит\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
    # "болит голова", "болит живот"
    re.compile(r"болит\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
    # "боль в горле", "боль в животе"
    re.compile(r"боль\s+в\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
    # "лекарство от простуды"
    re.compile(r"лекарство\s+от\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
    # "средство от кашля"
    re.compile(r"средство\s+от\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
    # "препарат от температуры"
    re.compile(r"препарат\s+от\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
    # "что-нибудь от головы"
    re.compile(r"что-(?:нибудь|то)\s+от\s+(?P<symptom>[а-яё]+(?:\s+[а-яё]+)?)", re.IGNORECASE),
    # "помогает от кашля"
    re.compile(r"помогает\s+от\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
    # "у меня кашель", "у ребенка температура"
    re.compile(r"у\s+(?:меня|него|нее|ребенка|ребёнка|малыша)\s+(?P<symptom>[а-яё]+(?:\s+[а-яё]+)?)", re.IGNORECASE),
    # "мучает кашель", "беспокоит насморк"
    re.compile(r"(?:мучает|беспокоит|замучил[а]?)\s+(?P<symptom>[а-яё]+)", re.IGNORECASE),
]

# Legacy pattern для обратной совместимости
SYMPTOM_PATTERN = re.compile(
    r"(?:от|для|при)\s+(?P<symptom>[а-яa-z\s]+)", re.IGNORECASE
)

# Для обратной совместимости
AGE_PATTERN = SIMPLE_AGE_PATTERN
PRICE_PATTERN = PRICE_MAX_PATTERNS[0]


# ============================================================================
# Dataclass для результата извлечения
# ============================================================================

@dataclass
class ExtractionResult:
    """Результат извлечения сущностей из текста."""
    
    # Возраст
    age: Optional[int] = None
    age_group: Optional[str] = None  # child, teenager, adult, elderly
    is_for_children: bool = False
    is_for_adults: bool = False
    is_for_elderly: bool = False
    is_for_teenager: bool = False
    
    # Цена
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    price_filter_disabled: bool = False  # Пользователь сказал "неважно"/"любая"
    
    # Симптомы и болезни
    symptom: Optional[str] = None
    symptoms: List[str] = field(default_factory=list)
    disease: Optional[str] = None
    diseases: List[str] = field(default_factory=list)
    symptom_duration: Optional[str] = None  # today, few_days, week, long
    symptom_severity: Optional[str] = None  # mild, moderate, severe
    
    # Форма выпуска
    dosage_form: Optional[str] = None
    dosage_forms: List[str] = field(default_factory=list)
    
    # Специальные фильтры
    special_filters: Dict[str, bool] = field(default_factory=dict)
    
    # Беременность/лактация
    is_pregnant: bool = False
    is_breastfeeding: bool = False
    pregnancy_status: Optional[str] = None  # pregnant, breastfeeding, none
    
    # Хронические заболевания и противопоказания
    chronic_conditions: List[str] = field(default_factory=list)
    has_chronic_conditions: Optional[bool] = None  # True/False/None=не спрашивали
    
    # Аллергии
    has_allergies: Optional[bool] = None  # True/False/None=не спрашивали
    allergies: List[str] = field(default_factory=list)
    
    # Текущие лекарства
    current_medications: List[str] = field(default_factory=list)
    
    # Рецептурность
    is_otc: Optional[bool] = None  # True = без рецепта, False = рецептурный, None = не указано
    
    # Метаданные извлечения
    matched_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_slots_dict(self) -> Dict[str, Any]:
        """Конвертирует результат в словарь слотов."""
        slots: Dict[str, Any] = {}
        
        # Возраст
        if self.age is not None:
            slots["age"] = self.age
        if self.age_group is not None:
            slots["age_group"] = self.age_group
        if self.is_for_children:
            slots["is_for_children"] = True
        if self.is_for_adults:
            slots["is_for_adults"] = True
        if self.is_for_elderly:
            slots["is_for_elderly"] = True
        if self.is_for_teenager:
            slots["is_for_teenager"] = True
            
        # Цена
        if self.price_min is not None:
            slots["price_min"] = self.price_min
        if self.price_max is not None:
            slots["price_max"] = self.price_max
        if self.price_filter_disabled:
            slots["price_filter_disabled"] = True
            
        # Симптомы и болезни
        if self.symptom:
            slots["symptom"] = self.symptom
        if self.symptoms:
            slots["symptoms"] = self.symptoms
        if self.disease:
            slots["disease"] = self.disease
        if self.diseases:
            slots["diseases"] = self.diseases
        if self.symptom_duration:
            slots["symptom_duration"] = self.symptom_duration
        if self.symptom_severity:
            slots["symptom_severity"] = self.symptom_severity
            
        # Форма выпуска
        if self.dosage_form:
            slots["dosage_form"] = self.dosage_form
        if self.dosage_forms:
            slots["dosage_forms"] = self.dosage_forms
            
        # Специальные фильтры
        for filter_name, value in self.special_filters.items():
            if value:
                slots[filter_name] = True
                
        # Беременность/лактация
        if self.is_pregnant:
            slots["is_pregnant"] = True
        if self.is_breastfeeding:
            slots["is_breastfeeding"] = True
        if self.pregnancy_status:
            slots["pregnancy_status"] = self.pregnancy_status
            
        # Хронические заболевания
        if self.chronic_conditions:
            slots["chronic_conditions"] = self.chronic_conditions
        if self.has_chronic_conditions is not None:
            slots["has_chronic_conditions"] = self.has_chronic_conditions
            
        # Аллергии
        if self.has_allergies is not None:
            slots["has_allergies"] = self.has_allergies
        if self.allergies:
            slots["allergies"] = self.allergies
            
        # Текущие лекарства
        if self.current_medications:
            slots["current_medications"] = self.current_medications
            
        # Рецептурность
        if self.is_otc is not None:
            slots["is_otc"] = self.is_otc
            
        return slots


# ============================================================================
# Основные функции извлечения
# ============================================================================

def extract_age(message: str) -> Optional[int]:
    """
    Извлекает возраст из сообщения.
    
    Поддерживает форматы:
    - "мне 30 лет", "ему 5 лет"
    - "ребенку 5 лет", "малышу 3 года"
    - "возраст 30"
    - "30" (короткий ответ)
    - "нам 5"
    
    Returns:
        int если возраст найден (1-110), None в противном случае
    """
    normalized = normalize_text(message)
    
    # Пробуем расширенные паттерны по порядку приоритета
    for pattern in AGE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            try:
                age = int(match.group("age"))
                if 1 <= age <= 110:
                    return age
            except (ValueError, IndexError):
                continue
    
    # Fallback на простой паттерн для коротких ответов
    if len(normalized.split()) <= 3:
        match = SIMPLE_AGE_PATTERN.search(normalized)
        if match:
            try:
                age = int(match.group("age"))
                if 1 <= age <= 99:
                    return age
            except ValueError:
                pass
    
    return None


def extract_price(message: str) -> Optional[int]:
    """
    Извлекает максимальную цену из сообщения (для обратной совместимости).
    
    Returns:
        int если цена найдена, None в противном случае
    """
    return extract_price_max(message)


def extract_price_max(message: str) -> Optional[int]:
    """
    Извлекает верхнюю границу цены.
    
    Поддерживает:
    - "до 500 рублей", "до 500р", "до 500₽"
    - "максимум 1000", "не дороже 500"
    - "бюджет 500"
    - "подешевле" → дефолтный бюджет
    """
    normalized = normalize_text(message)
    
    # Сначала проверяем диапазон
    range_match = PRICE_RANGE_PATTERN.search(normalized)
    if range_match:
        try:
            return int(range_match.group("max"))
        except (ValueError, IndexError):
            pass
    
    # Проверяем паттерны максимальной цены
    for pattern in PRICE_MAX_PATTERNS:
        match = pattern.search(normalized)
        if match:
            try:
                price = int(match.group("price"))
                if price > 0:
                    return price
            except (ValueError, IndexError):
                continue
    
    # Проверяем бюджетные маркеры
    if any(marker in normalized for marker in BUDGET_MARKERS):
        return DEFAULT_BUDGET_PRICE
    
    return None


def extract_price_min(message: str) -> Optional[int]:
    """
    Извлекает нижнюю границу цены.
    
    Поддерживает:
    - "от 500 рублей"
    - "минимум 300"
    - "не дешевле 1000"
    """
    normalized = normalize_text(message)
    
    # Сначала проверяем диапазон
    range_match = PRICE_RANGE_PATTERN.search(normalized)
    if range_match:
        try:
            return int(range_match.group("min"))
        except (ValueError, IndexError):
            pass
    
    # Проверяем паттерны минимальной цены
    for pattern in PRICE_MIN_PATTERNS:
        match = pattern.search(normalized)
        if match:
            try:
                price = int(match.group("price"))
                if price > 0:
                    return price
            except (ValueError, IndexError):
                continue
    
    return None


def extract_price_range(message: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Извлекает ценовой диапазон.
    
    Returns:
        (price_min, price_max) - кортеж с границами
    """
    return (extract_price_min(message), extract_price_max(message))


def extract_symptom(message: str) -> Optional[str]:
    """
    Извлекает симптом из сообщения.
    
    Returns:
        Нормализованное название симптома или None
    """
    normalized = normalize_text(message)
    
    # Проверяем контекстные паттерны
    for pattern in SYMPTOM_CONTEXT_PATTERNS:
        match = pattern.search(normalized)
        if match:
            raw_symptom = match.group("symptom")
            normalized_symptom = normalize_symptom(raw_symptom)
            if normalized_symptom:
                return normalized_symptom
    
    # Проверяем прямое вхождение симптомов
    symptom = find_symptom_by_stem(normalized)
    if symptom:
        return symptom
    
    return None


def extract_symptoms(message: str) -> List[str]:
    """
    Извлекает все симптомы из сообщения.
    
    Returns:
        Список нормализованных симптомов
    """
    normalized = normalize_text(message)
    symptoms: Set[str] = set()
    
    # Ищем через паттерны
    for pattern in SYMPTOM_CONTEXT_PATTERNS:
        for match in pattern.finditer(normalized):
            raw_symptom = match.group("symptom")
            normalized_symptom = normalize_symptom(raw_symptom)
            if normalized_symptom:
                symptoms.add(normalized_symptom)
    
    # Ищем прямые вхождения
    for stem, symptom_name in SYMPTOM_STEMS.items():
        if stem in normalized:
            symptoms.add(symptom_name)
    
    return list(symptoms)


def normalize_symptom(raw: str) -> Optional[str]:
    """
    Нормализует симптом, приводя к стандартному названию.
    
    ВАЖНО: Возвращает симптом ТОЛЬКО если он найден в словаре SYMPTOM_STEMS.
    Это предотвращает ложные срабатывания на не-медицинские тексты.
    """
    if not raw:
        return None
    
    cleaned = clean_symptom(raw.lower().strip())
    if not cleaned:
        return None
    
    # Ищем в словаре симптомов - возвращаем ТОЛЬКО если нашли
    for stem, symptom_name in SYMPTOM_STEMS.items():
        if stem in cleaned:
            return symptom_name
    
    # Если не нашли в словаре медицинских терминов - это НЕ симптом
    return None


def find_symptom_by_stem(text: str) -> Optional[str]:
    """
    Находит симптом по основе слова в тексте.
    """
    for stem, symptom_name in SYMPTOM_STEMS.items():
        if stem in text:
            return symptom_name
    return None


def extract_disease(message: str) -> Optional[str]:
    """
    Извлекает заболевание из сообщения.
    """
    normalized = normalize_text(message)
    
    for stem, disease_name in DISEASE_STEMS.items():
        if stem in normalized:
            return disease_name
    
    return None


def extract_diseases(message: str) -> List[str]:
    """
    Извлекает все заболевания из сообщения.
    """
    normalized = normalize_text(message)
    diseases: Set[str] = set()
    
    for stem, disease_name in DISEASE_STEMS.items():
        if stem in normalized:
            diseases.add(disease_name)
    
    return list(diseases)


def extract_dosage_form(
    message: str,
    keywords: Dict[str, List[str]] | None = None,
) -> Optional[str]:
    """
    Извлекает форму выпуска из сообщения.
    
    Args:
        message: Текст сообщения
        keywords: Опциональный словарь ключевых слов
    
    Returns:
        Название формы (tablets, syrup, etc.) или None
    """
    if keywords is None:
        keywords = DEFAULT_DOSAGE_FORM_KEYWORDS
    
    normalized = normalize_text(message)
    
    for form, form_keywords in keywords.items():
        if any(keyword in normalized for keyword in form_keywords):
            return form
    
    return None


def extract_dosage_forms(
    message: str,
    keywords: Dict[str, List[str]] | None = None,
) -> List[str]:
    """
    Извлекает все упомянутые формы выпуска.
    """
    if keywords is None:
        keywords = DEFAULT_DOSAGE_FORM_KEYWORDS
    
    normalized = normalize_text(message)
    forms: List[str] = []
    
    for form, form_keywords in keywords.items():
        if any(keyword in normalized for keyword in form_keywords):
            forms.append(form)
    
    return forms


def extract_is_for_children(message: str) -> bool:
    """
    Определяет, запрашивается ли детский препарат.
    """
    normalized = normalize_text(message)
    return any(marker in normalized for marker in CHILDREN_MARKERS)


def extract_is_for_adults(message: str) -> bool:
    """
    Определяет, запрашивается ли взрослый препарат.
    """
    normalized = normalize_text(message)
    return any(marker in normalized for marker in ADULT_MARKERS)


def extract_is_for_teenager(message: str) -> bool:
    """
    Определяет, запрашивается ли препарат для подростка.
    """
    normalized = normalize_text(message)
    return any(marker in normalized for marker in TEENAGER_MARKERS)


def extract_is_for_elderly(message: str) -> bool:
    """
    Определяет, запрашивается ли препарат для пожилого человека.
    """
    normalized = normalize_text(message)
    return any(marker in normalized for marker in ELDERLY_MARKERS)


def age_to_age_group(age: int) -> Optional[str]:
    """
    Преобразует числовой возраст в возрастную группу.
    
    Args:
        age: Возраст в годах
        
    Returns:
        Возрастная группа (child, teenager, adult, elderly) или None
    """
    if age < 0:
        return None
    if age <= 12:
        return AgeGroup.CHILD
    if age <= 17:
        return AgeGroup.TEENAGER
    if age <= 59:
        return AgeGroup.ADULT
    return AgeGroup.ELDERLY


def extract_age_group(message: str) -> Optional[str]:
    """
    Извлекает возрастную группу из сообщения.
    
    Приоритеты:
    1. Явное указание возрастной группы (ребёнок, взрослый и т.д.)
    2. Числовой возраст -> преобразование в группу
    
    Returns:
        Одно из значений: child, teenager, adult, elderly или None
    """
    normalized = normalize_text(message)
    
    # Сначала проверяем явные маркеры возрастных групп
    if any(marker in normalized for marker in CHILDREN_MARKERS):
        return AgeGroup.CHILD
    
    if any(marker in normalized for marker in TEENAGER_MARKERS):
        return AgeGroup.TEENAGER
    
    if any(marker in normalized for marker in ELDERLY_MARKERS):
        return AgeGroup.ELDERLY
    
    if any(marker in normalized for marker in ADULT_MARKERS):
        return AgeGroup.ADULT
    
    # Пытаемся извлечь возраст и преобразовать в группу
    age = extract_age(message)
    if age is not None:
        return age_to_age_group(age)
    
    return None


def extract_age_group_verbose(message: str) -> Tuple[Optional[str], Optional[int], str]:
    """
    Расширенное извлечение возрастной группы с метаданными.
    
    Returns:
        (age_group, age, source) - группа, возраст (если есть), источник определения
    """
    normalized = normalize_text(message)
    
    # Явные маркеры
    if any(marker in normalized for marker in CHILDREN_MARKERS):
        age = extract_age(message)
        return (AgeGroup.CHILD, age, "explicit_marker")
    
    if any(marker in normalized for marker in TEENAGER_MARKERS):
        age = extract_age(message)
        return (AgeGroup.TEENAGER, age, "explicit_marker")
    
    if any(marker in normalized for marker in ELDERLY_MARKERS):
        age = extract_age(message)
        return (AgeGroup.ELDERLY, age, "explicit_marker")
    
    if any(marker in normalized for marker in ADULT_MARKERS):
        age = extract_age(message)
        return (AgeGroup.ADULT, age, "explicit_marker")
    
    # Числовой возраст
    age = extract_age(message)
    if age is not None:
        return (age_to_age_group(age), age, "inferred_from_age")
    
    return (None, None, "not_found")


def extract_special_filters(message: str) -> Dict[str, bool]:
    """
    Извлекает специальные фильтры (без сахара, без лактозы и т.д.)
    """
    normalized = normalize_text(message)
    filters: Dict[str, bool] = {}
    
    for filter_name, markers in SPECIAL_FILTERS.items():
        if any(marker in normalized for marker in markers):
            filters[filter_name] = True
    
    return filters


def extract_pregnancy_context(message: str) -> Tuple[bool, bool]:
    """
    Извлекает контекст беременности/лактации.
    
    Returns:
        (is_pregnant, is_breastfeeding)
    """
    normalized = normalize_text(message)
    is_pregnant = any(marker in normalized for marker in ["беременн", "в положении"])
    is_breastfeeding = any(marker in normalized for marker in ["кормящ", "лактации", "грудное вскармливание"])
    
    return is_pregnant, is_breastfeeding


def extract_otc_preference(message: str) -> Optional[bool]:
    """
    Определяет предпочтение по рецептурности.
    
    Returns:
        True = без рецепта, False = рецептурный, None = не указано
    """
    normalized = normalize_text(message)
    
    if any(marker in normalized for marker in OTC_MARKERS):
        return True
    if any(marker in normalized for marker in PRESCRIPTION_MARKERS):
        return False
    
    return None


def extract_pregnancy_status(message: str) -> Optional[str]:
    """
    Извлекает статус беременности/лактации.
    
    Returns:
        'pregnant', 'breastfeeding', 'none' или None если не указано
    """
    normalized = normalize_text(message)
    
    # Проверяем отрицание
    if any(marker in normalized for marker in NO_PREGNANCY_MARKERS):
        return PregnancyStatus.NONE
    
    # Проверяем беременность
    if any(marker in normalized for marker in PREGNANCY_MARKERS):
        return PregnancyStatus.PREGNANT
    
    # Проверяем кормление грудью
    if any(marker in normalized for marker in BREASTFEEDING_MARKERS):
        return PregnancyStatus.BREASTFEEDING
    
    return None


def extract_chronic_conditions(message: str) -> Tuple[Optional[bool], List[str]]:
    """
    Извлекает хронические заболевания из сообщения.
    
    Returns:
        (has_conditions, list_of_conditions)
        has_conditions: True если есть, False если явно указано что нет, None если не указано
    """
    normalized = normalize_text(message)
    conditions: List[str] = []
    
    # Проверяем отрицание
    if any(marker in normalized for marker in NO_CHRONIC_MARKERS):
        return (False, [])
    
    # Ищем конкретные заболевания
    for condition, markers in CHRONIC_CONDITIONS_MARKERS.items():
        if any(marker in normalized for marker in markers):
            conditions.append(condition)
    
    if conditions:
        return (True, conditions)
    
    return (None, [])


def extract_allergies(message: str) -> Tuple[Optional[bool], List[str]]:
    """
    Извлекает информацию об аллергиях.
    
    Returns:
        (has_allergies, list_of_allergens)
    """
    normalized = normalize_text(message)
    allergens: List[str] = []
    
    # Проверяем отрицание
    if any(marker in normalized for marker in NO_ALLERGY_MARKERS):
        return (False, [])
    
    # Проверяем наличие аллергии
    has_allergy_marker = any(marker in normalized for marker in ALLERGY_MARKERS)
    
    if has_allergy_marker:
        # Пытаемся извлечь на что именно аллергия
        # Паттерн: "аллергия на <что-то>"
        import re
        patterns = [
            r"аллергия на ([а-яё\s]+)",
            r"непереносимость ([а-яё\s]+)",
            r"реакция на ([а-яё\s]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                allergen = match.group(1).strip()
                if allergen and len(allergen) > 2:
                    allergens.append(allergen)
        
        return (True, allergens)
    
    return (None, [])


def extract_symptom_duration(message: str) -> Optional[str]:
    """
    Извлекает длительность симптомов.
    
    Returns:
        'today', 'few_days', 'week', 'long' или None
    """
    normalized = normalize_text(message)
    
    for duration, markers in SYMPTOM_DURATION_MARKERS.items():
        if any(marker in normalized for marker in markers):
            return duration
    
    return None


def extract_symptom_severity(message: str) -> Optional[str]:
    """
    Извлекает выраженность симптомов.
    
    Returns:
        'mild', 'moderate', 'severe' или None
    """
    normalized = normalize_text(message)
    
    # Проверяем в порядке приоритета: severe -> moderate -> mild
    # Это важно, чтобы "нестерпимо" не совпало с "терпимо"
    priority_order = ["severe", "moderate", "mild"]
    
    for severity in priority_order:
        markers = SYMPTOM_SEVERITY_MARKERS.get(severity, [])
        if any(marker in normalized for marker in markers):
            return severity
    
    return None


def clean_symptom(raw: str) -> str:
    """
    Очищает извлеченный симптом от стоп-слов.
    """
    text = raw.strip()
    stop_words = [" до ", " за ", " по ", " выше ", " ниже ", " по цене ", " рублей", " руб", " лет", " года"]
    for stop in stop_words:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip(" ,.")


def check_price_indifference(message: str) -> bool:
    """
    Проверяет, указал ли пользователь безразличие к цене.
    """
    normalized = normalize_text(message)
    indifference_markers = [
        "неважно", "не важно", "любая", "любой", "все равно", "всё равно",
        "без разницы", "не имеет значения", "любая цена", "на любую сумму",
        "сколько угодно", "без ограничений", "пропустить", "пропущу"
    ]
    return any(marker in normalized for marker in indifference_markers)


# ============================================================================
# Комплексное извлечение
# ============================================================================

def extract_all_slots(
    message: str,
    dosage_form_keywords: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    """
    Извлекает все доступные слоты из сообщения.
    
    Это основная функция для интеграции с router'ом и slot_manager'ом.
    
    Returns:
        Словарь только с non-None значениями
    """
    result = extract_all_entities(message, dosage_form_keywords)
    return result.to_slots_dict()


def extract_all_entities(
    message: str,
    dosage_form_keywords: Dict[str, List[str]] | None = None,
) -> ExtractionResult:
    """
    Полное извлечение всех сущностей с метаданными.
    
    Returns:
        ExtractionResult со всеми найденными сущностями
    """
    result = ExtractionResult()
    matched_patterns: List[str] = []
    
    # Возраст
    age = extract_age(message)
    if age is not None:
        result.age = age
        matched_patterns.append("age")
    
    # Возрастная группа (из явных маркеров или из возраста)
    age_group = extract_age_group(message)
    if age_group is not None:
        result.age_group = age_group
        matched_patterns.append("age_group")
    
    # Детский/взрослый/подросток/пожилой контекст
    result.is_for_children = extract_is_for_children(message)
    if result.is_for_children:
        matched_patterns.append("children_marker")
        # Если нет age_group, но есть детский маркер - устанавливаем
        if result.age_group is None:
            result.age_group = AgeGroup.CHILD
    
    result.is_for_teenager = extract_is_for_teenager(message)
    if result.is_for_teenager:
        matched_patterns.append("teenager_marker")
        if result.age_group is None:
            result.age_group = AgeGroup.TEENAGER
    
    result.is_for_adults = extract_is_for_adults(message)
    if result.is_for_adults:
        matched_patterns.append("adult_marker")
        if result.age_group is None:
            result.age_group = AgeGroup.ADULT
    
    result.is_for_elderly = extract_is_for_elderly(message)
    if result.is_for_elderly:
        matched_patterns.append("elderly_marker")
        if result.age_group is None:
            result.age_group = AgeGroup.ELDERLY
    
    # Цена
    price_min = extract_price_min(message)
    price_max = extract_price_max(message)
    
    if price_min is not None:
        result.price_min = price_min
        matched_patterns.append("price_min")
    if price_max is not None:
        result.price_max = price_max
        matched_patterns.append("price_max")
    
    # Безразличие к цене
    if check_price_indifference(message):
        result.price_filter_disabled = True
        matched_patterns.append("price_indifference")
    
    # Симптомы
    symptoms = extract_symptoms(message)
    if symptoms:
        result.symptom = symptoms[0]
        result.symptoms = symptoms
        matched_patterns.append("symptom")
    
    # Заболевания
    diseases = extract_diseases(message)
    if diseases:
        result.disease = diseases[0]
        result.diseases = diseases
        matched_patterns.append("disease")
    
    # Длительность и выраженность симптомов
    symptom_duration = extract_symptom_duration(message)
    if symptom_duration:
        result.symptom_duration = symptom_duration
        matched_patterns.append("symptom_duration")
    
    symptom_severity = extract_symptom_severity(message)
    if symptom_severity:
        result.symptom_severity = symptom_severity
        matched_patterns.append("symptom_severity")
    
    # Форма выпуска
    forms = extract_dosage_forms(message, dosage_form_keywords)
    if forms:
        result.dosage_form = forms[0]
        result.dosage_forms = forms
        matched_patterns.append("dosage_form")
    
    # Специальные фильтры
    result.special_filters = extract_special_filters(message)
    if result.special_filters:
        matched_patterns.append("special_filters")
    
    # Беременность/лактация
    is_pregnant, is_breastfeeding = extract_pregnancy_context(message)
    result.is_pregnant = is_pregnant
    result.is_breastfeeding = is_breastfeeding
    if is_pregnant:
        result.pregnancy_status = PregnancyStatus.PREGNANT
        matched_patterns.append("pregnancy")
    if is_breastfeeding:
        result.pregnancy_status = PregnancyStatus.BREASTFEEDING
        matched_patterns.append("breastfeeding")
    
    # Статус беременности (включая отрицание)
    pregnancy_status = extract_pregnancy_status(message)
    if pregnancy_status:
        result.pregnancy_status = pregnancy_status
        matched_patterns.append("pregnancy_status")
    
    # Хронические заболевания
    has_chronic, chronic_list = extract_chronic_conditions(message)
    if has_chronic is not None:
        result.has_chronic_conditions = has_chronic
        result.chronic_conditions = chronic_list
        matched_patterns.append("chronic_conditions")
    
    # Аллергии
    has_allergy, allergy_list = extract_allergies(message)
    if has_allergy is not None:
        result.has_allergies = has_allergy
        result.allergies = allergy_list
        matched_patterns.append("allergies")
    
    # OTC
    result.is_otc = extract_otc_preference(message)
    if result.is_otc is not None:
        matched_patterns.append("otc_preference")
    
    # Метаданные
    result.matched_patterns = matched_patterns
    result.confidence = min(1.0, len(matched_patterns) * 0.15)  # Примитивная оценка
    
    return result


# ============================================================================
# Экспорт
# ============================================================================

__all__ = [
    # Паттерны (для обратной совместимости)
    "AGE_PATTERN",
    "PRICE_PATTERN",
    "SYMPTOM_PATTERN",
    "DEFAULT_DOSAGE_FORM_KEYWORDS",
    # Словари
    "SYMPTOM_STEMS",
    "DISEASE_STEMS",
    "SPECIAL_FILTERS",
    "CHILDREN_MARKERS",
    "TEENAGER_MARKERS",
    "ADULT_MARKERS",
    "ELDERLY_MARKERS",
    "CHRONIC_CONDITIONS_MARKERS",
    "SYMPTOM_DURATION_MARKERS",
    "SYMPTOM_SEVERITY_MARKERS",
    # Классы и константы
    "AgeGroup",
    "AGE_GROUP_RANGES",
    "PregnancyStatus",
    # Основные функции извлечения
    "extract_age",
    "extract_age_group",
    "extract_age_group_verbose",
    "age_to_age_group",
    "extract_price",
    "extract_price_max",
    "extract_price_min",
    "extract_price_range",
    "extract_symptom",
    "extract_symptoms",
    "extract_disease",
    "extract_diseases",
    "extract_symptom_duration",
    "extract_symptom_severity",
    "extract_dosage_form",
    "extract_dosage_forms",
    "extract_is_for_children",
    "extract_is_for_teenager",
    "extract_is_for_adults",
    "extract_is_for_elderly",
    "extract_special_filters",
    "extract_pregnancy_context",
    "extract_pregnancy_status",
    "extract_chronic_conditions",
    "extract_allergies",
    "extract_otc_preference",
    # Утилиты
    "normalize_text",
    "normalize_symptom",
    "clean_symptom",
    "check_price_indifference",
    # Комплексное извлечение
    "extract_all_slots",
    "extract_all_entities",
    "ExtractionResult",
]
