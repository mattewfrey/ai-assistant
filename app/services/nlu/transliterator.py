"""
Transliterator - конвертация латиницы в кириллицу и наоборот.

Обрабатывает случаи когда пользователь вводит:
- "nurofen" → "нурофен"
- "paracetamol" → "парацетамол"
- Смешанный ввод "nурофен" → "нурофен"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# ============================================================================
# Таблицы транслитерации
# ============================================================================

# Латиница → Кириллица (фонетическая)
LATIN_TO_CYRILLIC: Dict[str, str] = {
    # Двухбуквенные сочетания (проверяем первыми)
    "sh": "ш",
    "ch": "ч",
    "zh": "ж",
    "ts": "ц",
    "sch": "щ",
    "shch": "щ",
    "ya": "я",
    "yu": "ю",
    "yo": "ё",
    "ye": "е",
    "iy": "ий",
    "ey": "ей",
    "ay": "ай",
    "oy": "ой",
    "uy": "уй",
    "kh": "х",
    "ph": "ф",
    "th": "т",  # В медицинских терминах часто
    
    # Однобуквенные
    "a": "а",
    "b": "б",
    "c": "ц",  # или "к" в зависимости от контекста
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "и",
    "z": "з",
}

# Кириллица → Латиница (для обратной транслитерации)
CYRILLIC_TO_LATIN: Dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

# Специальные правила для медицинских терминов
# Слова которые НЕ нужно транслитерировать (команды, сокращения)
SKIP_TRANSLITERATION: Set[str] = {
    "cart", "ok", "hi", "hello", "bye", "yes", "no",
    "show", "add", "remove", "clear", "order", "help",
}

MEDICAL_TRANSLITERATION: Dict[str, str] = {
    # Латинские названия → русские
    "nurofen": "нурофен",
    "paracetamol": "парацетамол",
    "ibuprofen": "ибупрофен",
    "aspirin": "аспирин",
    "analgin": "анальгин",
    "citramone": "цитрамон",
    "theraflu": "терафлю",
    "coldrex": "колдрекс",
    "arbidol": "арбидол",
    "lazolvan": "лазолван",
    "ambroxol": "амброксол",
    "strepsils": "стрепсилс",
    "nazivin": "називин",
    "otrivin": "отривин",
    "suprastin": "супрастин",
    "loratadin": "лоратадин",
    "cetirizine": "цетиризин",
    "mezim": "мезим",
    "festal": "фестал",
    "omeprazole": "омепразол",
    "smecta": "смекта",
    "linex": "линекс",
    "validol": "валидол",
    "corvalol": "корвалол",
    "miramistin": "мирамистин",
    "chlorhexidine": "хлоргексидин",
    "amoxicillin": "амоксициллин",
    "azithromycin": "азитромицин",
    # Добавляем варианты без окончаний
    "nurofn": "нурофен",
    "parasetamol": "парацетамол",
    "ibuprofn": "ибупрофен",
}


@dataclass
class TransliterationResult:
    """Результат транслитерации."""
    original: str
    result: str
    was_transliterated: bool
    source_script: str  # "latin", "cyrillic", "mixed"
    method: str  # "dictionary", "rules", "none"


class Transliterator:
    """
    Сервис транслитерации текста.
    
    Особенности:
    - Приоритет словаря медицинских терминов
    - Поддержка смешанного ввода (nурофен)
    - Сохранение регистра
    """
    
    def __init__(
        self,
        medical_dict: Dict[str, str] | None = None,
        latin_to_cyrillic: Dict[str, str] | None = None,
    ):
        self._medical_dict = medical_dict or MEDICAL_TRANSLITERATION
        self._lat_to_cyr = latin_to_cyrillic or LATIN_TO_CYRILLIC
        
        # Сортируем ключи по длине (длинные первыми)
        self._sorted_lat_keys = sorted(
            self._lat_to_cyr.keys(),
            key=len,
            reverse=True,
        )
    
    def detect_script(self, text: str) -> str:
        """
        Определяет преобладающий скрипт текста.
        
        Returns:
            "latin", "cyrillic", или "mixed"
        """
        latin_count = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        cyrillic_count = sum(1 for c in text if c.isalpha() and ord(c) >= 0x0400 and ord(c) <= 0x04FF)
        
        if latin_count == 0 and cyrillic_count == 0:
            return "mixed"  # Нет букв вообще
        
        total = latin_count + cyrillic_count
        if latin_count / total > 0.8:
            return "latin"
        elif cyrillic_count / total > 0.8:
            return "cyrillic"
        else:
            return "mixed"
    
    def transliterate_word(self, word: str) -> TransliterationResult:
        """
        Транслитерирует одно слово.
        """
        word_lower = word.lower()
        script = self.detect_script(word)
        
        # Если уже кириллица — ничего не делаем
        if script == "cyrillic":
            return TransliterationResult(
                original=word,
                result=word,
                was_transliterated=False,
                source_script=script,
                method="none",
            )
        
        # Пропускаем короткие команды и служебные слова
        if word_lower in SKIP_TRANSLITERATION:
            return TransliterationResult(
                original=word,
                result=word,
                was_transliterated=False,
                source_script=script,
                method="skipped",
            )
        
        # 1. Проверяем словарь медицинских терминов
        if word_lower in self._medical_dict:
            result = self._medical_dict[word_lower]
            # Восстанавливаем регистр
            if word[0].isupper():
                result = result.capitalize()
            return TransliterationResult(
                original=word,
                result=result,
                was_transliterated=True,
                source_script=script,
                method="dictionary",
            )
        
        # 2. Применяем правила транслитерации
        if script in ("latin", "mixed"):
            result = self._apply_rules(word_lower)
            # Восстанавливаем регистр
            if word[0].isupper():
                result = result.capitalize()
            return TransliterationResult(
                original=word,
                result=result,
                was_transliterated=(result != word),
                source_script=script,
                method="rules",
            )
        
        return TransliterationResult(
            original=word,
            result=word,
            was_transliterated=False,
            source_script=script,
            method="none",
        )
    
    def _apply_rules(self, text: str) -> str:
        """Применяет правила транслитерации посимвольно."""
        result = []
        i = 0
        text_lower = text.lower()
        
        while i < len(text_lower):
            matched = False
            
            # Пробуем найти самое длинное совпадение
            for key in self._sorted_lat_keys:
                if text_lower[i:].startswith(key):
                    result.append(self._lat_to_cyr[key])
                    i += len(key)
                    matched = True
                    break
            
            if not matched:
                # Символ не латинский — оставляем как есть
                result.append(text[i])
                i += 1
        
        return "".join(result)
    
    def transliterate_text(self, text: str) -> Tuple[str, List[TransliterationResult]]:
        """
        Транслитерирует текст, обрабатывая каждое слово.
        
        Returns:
            (обработанный_текст, список_результатов)
        """
        # Разбиваем на токены, сохраняя разделители
        tokens = re.split(r'(\s+|[^\w]+)', text, flags=re.UNICODE)
        results: List[TransliterationResult] = []
        output_tokens: List[str] = []
        
        for token in tokens:
            if re.match(r'^\w+$', token, re.UNICODE):
                result = self.transliterate_word(token)
                if result.was_transliterated:
                    results.append(result)
                output_tokens.append(result.result)
            else:
                output_tokens.append(token)
        
        return "".join(output_tokens), results
    
    def to_cyrillic(self, text: str) -> str:
        """Упрощённый метод — просто конвертировать в кириллицу."""
        result, _ = self.transliterate_text(text)
        return result
    
    def to_latin(self, text: str) -> str:
        """Конвертирует кириллицу в латиницу."""
        result = []
        for char in text.lower():
            if char in CYRILLIC_TO_LATIN:
                result.append(CYRILLIC_TO_LATIN[char])
            else:
                result.append(char)
        return "".join(result)


# Singleton instance
_transliterator: Transliterator | None = None


def get_transliterator() -> Transliterator:
    """Возвращает singleton экземпляр Transliterator."""
    global _transliterator
    if _transliterator is None:
        _transliterator = Transliterator()
    return _transliterator


def transliterate(text: str) -> str:
    """Быстрая функция транслитерации латиницы в кириллицу."""
    return get_transliterator().to_cyrillic(text)

