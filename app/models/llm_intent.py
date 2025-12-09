"""
LLMIntentResult - Pydantic модель для structured output от LLM.

Эта модель определяет контракт между LLM и системой:
- LLM ВСЕГДА возвращает данные по этой структуре
- Используется механизм with_structured_output в LangChain
- Гарантирует предсказуемость и контроль над выходом LLM

Роль LangChain в проекте:
==========================
LangChain выступает как "умный мозг" ассистента со следующими задачами:

1. РАСПОЗНАВАНИЕ ИНТЕНТА - выбор одного интента из фиксированного списка IntentType
   - LLM получает сообщение пользователя и контекст
   - Возвращает intent с confidence (0-1)
   - При низком confidence Router может использовать fallback

2. ИЗВЛЕЧЕНИЕ СЛОТОВ - структурированное извлечение параметров:
   - age (возраст), symptom (симптом), disease (заболевание)
   - price_min/price_max (ценовые ограничения)
   - dosage_form (форма выпуска), is_for_children (детский контекст)
   - filters (без сахара, без лактозы и т.д.)

3. ГЕНЕРАЦИЯ REPLY - формирование человеко-понятного текста ответа:
   - На русском языке
   - Кратко и вежливо
   - С медицинскими дисклеймерами где нужно

4. ДИЗАМБИГУАЦИЯ - выбор между кандидатами Router'а:
   - Когда Router не уверен, он передаёт список кандидатов
   - LLM выбирает наиболее подходящий интент
   - Возвращает reasoning для отладки

Pipeline вызова LangChain:
=========================
1. Router пытается определить интент детерминированно
2. Если Router уверен (confidence >= 0.85) → используем его результат
3. Если Router не уверен, но есть кандидаты → LLM дизамбигуирует
4. Если Router ничего не нашёл → LLM полностью определяет intent + slots
5. SlotManager заполняет недостающие слоты
6. Orchestrator собирает финальный ответ с вызовами платформы
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..intents import IntentType


class SlotType(StrEnum):
    """Типы слотов, которые может извлечь LLM."""
    
    # Основные слоты
    SYMPTOM = "symptom"
    DISEASE = "disease"
    AGE = "age"
    
    # Ценовые слоты
    PRICE_MIN = "price_min"
    PRICE_MAX = "price_max"
    
    # Форма выпуска
    DOSAGE_FORM = "dosage_form"
    
    # Товар/заказ
    PRODUCT_NAME = "product_name"
    PRODUCT_ID = "product_id"
    ORDER_ID = "order_id"
    
    # Локация
    METRO = "metro"
    PHARMACY_ID = "pharmacy_id"
    
    # Булевые флаги
    IS_FOR_CHILDREN = "is_for_children"
    SUGAR_FREE = "sugar_free"
    LACTOSE_FREE = "lactose_free"
    IS_PREGNANT = "is_pregnant"
    
    # Прочее
    PROMO_CODE = "promo_code"
    DELIVERY_TYPE = "delivery_type"


class ExtractedSlots(BaseModel):
    """
    Извлечённые слоты из сообщения пользователя.
    
    Все поля опциональны - LLM заполняет только те, что нашёл в сообщении.
    """
    
    model_config = ConfigDict(extra="allow")
    
    # Симптомы и заболевания
    symptom: Optional[str] = Field(
        default=None,
        description="Симптом: головная боль, кашель, насморк, температура, боль в горле, etc."
    )
    symptoms: Optional[List[str]] = Field(
        default=None,
        description="Список симптомов, если пользователь упомянул несколько"
    )
    disease: Optional[str] = Field(
        default=None,
        description="Заболевание/диагноз: ОРВИ, грипп, гастрит, аллергия, etc."
    )
    
    # Возраст
    age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Возраст в годах (0-120)"
    )
    
    # Ценовые ограничения
    price_min: Optional[float] = Field(
        default=None,
        ge=0,
        description="Минимальная цена в рублях"
    )
    price_max: Optional[float] = Field(
        default=None,
        ge=0,
        description="Максимальная цена в рублях"
    )
    
    # Форма выпуска
    dosage_form: Optional[str] = Field(
        default=None,
        description="Форма выпуска: tablets, syrup, spray, drops, ointment, gel, capsules, powder"
    )
    
    # Товар
    product_name: Optional[str] = Field(
        default=None,
        description="Название препарата или бренд"
    )
    product_id: Optional[str] = Field(
        default=None,
        description="ID товара если известен"
    )
    
    # Заказ
    order_id: Optional[str] = Field(
        default=None,
        description="Номер заказа"
    )
    
    # Локация
    metro: Optional[str] = Field(
        default=None,
        description="Название станции метро"
    )
    pharmacy_id: Optional[str] = Field(
        default=None,
        description="ID аптеки если известен"
    )
    
    # Булевые флаги
    is_for_children: Optional[bool] = Field(
        default=None,
        description="True если ищут детский препарат"
    )
    sugar_free: Optional[bool] = Field(
        default=None,
        description="True если нужен препарат без сахара"
    )
    lactose_free: Optional[bool] = Field(
        default=None,
        description="True если нужен препарат без лактозы"
    )
    is_pregnant: Optional[bool] = Field(
        default=None,
        description="True если для беременной/кормящей"
    )
    
    # Прочее
    promo_code: Optional[str] = Field(
        default=None,
        description="Промокод"
    )
    delivery_type: Optional[str] = Field(
        default=None,
        description="Способ доставки: pickup или delivery"
    )
    
    # Дополнительные фильтры
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Дополнительные фильтры, не попавшие в основные поля"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь, исключая None значения."""
        result: Dict[str, Any] = {}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result
    
    def merge_with(self, other: "ExtractedSlots") -> "ExtractedSlots":
        """Мержит с другим ExtractedSlots, приоритет у self."""
        other_dict = other.to_dict()
        self_dict = self.to_dict()
        merged = {**other_dict, **self_dict}
        return ExtractedSlots(**merged)


class LLMIntentResult(BaseModel):
    """
    Результат классификации интента от LLM.
    
    Это основная модель для structured output. LLM ОБЯЗАН возвращать
    данные строго по этой структуре.
    
    Пример:
    {
        "intent": "FIND_BY_SYMPTOM",
        "confidence": 0.92,
        "slots": {
            "symptom": "головная боль",
            "age": 35
        },
        "reply": "Подберу препараты от головной боли для взрослого.",
        "reasoning": "Пользователь явно указал симптом 'болит голова' и возраст."
    }
    """
    
    model_config = ConfigDict(extra="forbid")
    
    intent: IntentType = Field(
        description="Выбранный интент из списка IntentType. Выбирай ОДИН наиболее подходящий."
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Уверенность в выборе интента от 0.0 до 1.0. "
                   "1.0 = абсолютно уверен, 0.5 = 50/50, <0.3 = неуверен."
    )
    
    slots: ExtractedSlots = Field(
        default_factory=ExtractedSlots,
        description="Извлечённые слоты/параметры из сообщения пользователя."
    )
    
    reply: str = Field(
        min_length=1,
        max_length=1000,
        description="Текст ответа пользователю на русском языке. "
                   "Кратко, вежливо, информативно. Без медицинских назначений."
    )
    
    reasoning: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Краткое пояснение выбора интента (для отладки, не показывается пользователю)."
    )
    
    # Дополнительные метаданные
    needs_clarification: bool = Field(
        default=False,
        description="True если требуется уточнение от пользователя."
    )
    
    missing_required_slots: Optional[List[str]] = Field(
        default=None,
        description="Список недостающих обязательных слотов."
    )
    
    alternative_intents: Optional[List[str]] = Field(
        default=None,
        description="Альтернативные интенты с меньшей уверенностью."
    )
    
    def to_parameters_dict(self) -> Dict[str, Any]:
        """Конвертирует slots в словарь для actions.parameters."""
        return self.slots.to_dict()
    
    def has_required_slots_for(self, intent: IntentType) -> bool:
        """Проверяет, есть ли обязательные слоты для интента."""
        required_slots_map: Dict[IntentType, List[str]] = {
            IntentType.FIND_BY_SYMPTOM: ["symptom"],
            IntentType.FIND_BY_DISEASE: ["disease"],
            IntentType.ADD_TO_CART: ["product_id"],
            IntentType.CANCEL_ORDER: ["order_id"],
            IntentType.SHOW_PHARMACIES_BY_METRO: ["metro"],
        }
        
        required = required_slots_map.get(intent, [])
        slots_dict = self.slots.to_dict()
        
        for slot in required:
            if slot not in slots_dict or slots_dict[slot] is None:
                return False
        return True


class LLMDisambiguationResult(BaseModel):
    """
    Результат дизамбигуации между кандидатами Router'а.
    
    Используется когда Router не уверен и нужна помощь LLM.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    selected_intent: IntentType = Field(
        description="Выбранный интент из списка кандидатов."
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Уверенность в выборе."
    )
    
    reasoning: str = Field(
        description="Объяснение почему выбран именно этот интент."
    )
    
    additional_slots: Optional[ExtractedSlots] = Field(
        default=None,
        description="Дополнительные слоты, извлечённые при анализе."
    )


class LLMSlotExtractionResult(BaseModel):
    """
    Результат извлечения слотов без классификации интента.
    
    Используется когда интент уже известен (от Router'а), но нужно
    извлечь дополнительные слоты с помощью LLM.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    slots: ExtractedSlots = Field(
        description="Извлечённые слоты."
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Общая уверенность в извлечении."
    )
    
    slot_confidences: Optional[Dict[str, float]] = Field(
        default=None,
        description="Уверенность по каждому извлечённому слоту."
    )


# =============================================================================
# Утилиты для конвертации
# =============================================================================

def slots_to_parameters(slots: ExtractedSlots | Dict[str, Any]) -> Dict[str, Any]:
    """Конвертирует слоты в параметры для AssistantAction."""
    if isinstance(slots, ExtractedSlots):
        return slots.to_dict()
    return {k: v for k, v in slots.items() if v is not None}


def merge_router_and_llm_slots(
    router_slots: Dict[str, Any],
    llm_slots: ExtractedSlots | Dict[str, Any],
) -> Dict[str, Any]:
    """
    Мержит слоты от Router'а и LLM.
    
    Приоритет:
    1. LLM слоты (более точное извлечение)
    2. Router слоты (как fallback)
    """
    if isinstance(llm_slots, ExtractedSlots):
        llm_dict = llm_slots.to_dict()
    else:
        llm_dict = {k: v for k, v in llm_slots.items() if v is not None}
    
    # Router слоты как база, LLM перезаписывает
    merged = {**router_slots}
    merged.update(llm_dict)
    
    return merged

