"""Service for checking drug interactions.

This service provides drug interaction checking functionality.
Currently uses a built-in database of common interactions.
Ready for integration with external APIs like DrugBank.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import Settings
from ..models.drug_interaction import (
    DrugInfo,
    DrugInteraction,
    DrugInteractionCheckRequest,
    DrugInteractionCheckResponse,
    InteractionSeverity,
)

logger = logging.getLogger(__name__)


# Common drug interactions database (simplified)
# In production, this would be replaced by DrugBank API
INTERACTION_DATABASE: List[Dict[str, Any]] = [
    # Anticoagulants
    {
        "drug_a": "варфарин",
        "drug_b": "ацетилсалициловая кислота",
        "severity": InteractionSeverity.MAJOR,
        "description": "Повышенный риск кровотечения при совместном применении",
        "mechanism": "Оба препарата влияют на свертываемость крови",
        "recommendation": "Избегайте совместного применения. Проконсультируйтесь с врачом.",
    },
    {
        "drug_a": "варфарин",
        "drug_b": "ибупрофен",
        "severity": InteractionSeverity.MAJOR,
        "description": "НПВС усиливают антикоагулянтный эффект варфарина",
        "mechanism": "Ибупрофен вытесняет варфарин из связи с белками плазмы",
        "recommendation": "Избегайте совместного применения или требуется мониторинг МНО.",
    },
    {
        "drug_a": "ацетилсалициловая кислота",
        "drug_b": "ибупрофен",
        "severity": InteractionSeverity.MAJOR,
        "description": "Повышенный риск желудочно-кишечных кровотечений",
        "mechanism": "Оба препарата раздражают слизистую желудка",
        "recommendation": "Избегайте совместного применения. При необходимости — с гастропротекцией.",
    },

    # Statins and certain drugs
    {
        "drug_a": "симвастатин",
        "drug_b": "амиодарон",
        "severity": InteractionSeverity.MAJOR,
        "description": "Повышенный риск миопатии и рабдомиолиза",
        "mechanism": "Амиодарон ингибирует метаболизм симвастатина",
        "recommendation": "Доза симвастатина не должна превышать 20 мг/сут.",
    },
    {
        "drug_a": "аторвастатин",
        "drug_b": "кларитромицин",
        "severity": InteractionSeverity.MODERATE,
        "description": "Повышение концентрации статина в крови",
        "mechanism": "Кларитромицин ингибирует CYP3A4",
        "recommendation": "Рассмотрите временное прекращение приема статина.",
    },

    # Common OTC interactions
    {
        "drug_a": "парацетамол",
        "drug_b": "алкоголь",
        "severity": InteractionSeverity.MAJOR,
        "description": "Повышенный риск повреждения печени",
        "mechanism": "Алкоголь усиливает гепатотоксичность парацетамола",
        "recommendation": "Избегайте употребления алкоголя при приеме парацетамола.",
    },
    {
        "drug_a": "ибупрофен",
        "drug_b": "напроксен",
        "severity": InteractionSeverity.MODERATE,
        "description": "Повышенный риск побочных эффектов НПВС",
        "mechanism": "Оба препарата — НПВС с аналогичным механизмом действия",
        "recommendation": "Не принимайте два НПВС одновременно.",
    },

    # Antibiotics
    {
        "drug_a": "метронидазол",
        "drug_b": "алкоголь",
        "severity": InteractionSeverity.CONTRAINDICATED,
        "description": "Дисульфирамоподобная реакция (тошнота, рвота, головная боль)",
        "mechanism": "Метронидазол ингибирует альдегиддегидрогеназу",
        "recommendation": "Категорически запрещено употреблять алкоголь во время и 3 дня после лечения.",
    },
    {
        "drug_a": "ципрофлоксацин",
        "drug_b": "антациды",
        "severity": InteractionSeverity.MODERATE,
        "description": "Снижение всасывания ципрофлоксацина",
        "mechanism": "Катионы металлов в антацидах связывают антибиотик",
        "recommendation": "Принимайте ципрофлоксацин за 2 часа до или через 6 часов после антацидов.",
    },

    # Blood pressure medications
    {
        "drug_a": "эналаприл",
        "drug_b": "калий",
        "severity": InteractionSeverity.MODERATE,
        "description": "Риск гиперкалиемии",
        "mechanism": "Ингибиторы АПФ уменьшают выведение калия",
        "recommendation": "Контролируйте уровень калия в крови.",
    },
    {
        "drug_a": "лозартан",
        "drug_b": "ибупрофен",
        "severity": InteractionSeverity.MODERATE,
        "description": "Снижение антигипертензивного эффекта",
        "mechanism": "НПВС блокируют синтез простагландинов в почках",
        "recommendation": "Мониторинг АД. Рассмотрите альтернативный анальгетик.",
    },

    # Diabetes medications
    {
        "drug_a": "метформин",
        "drug_b": "алкоголь",
        "severity": InteractionSeverity.MAJOR,
        "description": "Повышенный риск лактоацидоза",
        "mechanism": "Алкоголь усиливает эффект метформина на метаболизм лактата",
        "recommendation": "Ограничьте употребление алкоголя.",
    },
    {
        "drug_a": "инсулин",
        "drug_b": "алкоголь",
        "severity": InteractionSeverity.MAJOR,
        "description": "Риск тяжелой гипогликемии",
        "mechanism": "Алкоголь подавляет глюконеогенез в печени",
        "recommendation": "Избегайте алкоголя или употребляйте с едой. Контролируйте глюкозу.",
    },

    # Thyroid medications
    {
        "drug_a": "левотироксин",
        "drug_b": "кальций",
        "severity": InteractionSeverity.MODERATE,
        "description": "Снижение всасывания левотироксина",
        "mechanism": "Кальций связывает левотироксин в ЖКТ",
        "recommendation": "Принимайте препараты с интервалом минимум 4 часа.",
    },
    {
        "drug_a": "левотироксин",
        "drug_b": "железо",
        "severity": InteractionSeverity.MODERATE,
        "description": "Снижение всасывания левотироксина",
        "mechanism": "Железо образует нерастворимые комплексы с левотироксином",
        "recommendation": "Принимайте препараты с интервалом минимум 4 часа.",
    },

    # Psychiatric medications
    {
        "drug_a": "сертралин",
        "drug_b": "трамадол",
        "severity": InteractionSeverity.MAJOR,
        "description": "Риск серотонинового синдрома",
        "mechanism": "Оба препарата повышают уровень серотонина",
        "recommendation": "Избегайте комбинации. Симптомы: возбуждение, гипертермия, тремор.",
    },
    {
        "drug_a": "флуоксетин",
        "drug_b": "мао ингибиторы",
        "severity": InteractionSeverity.CONTRAINDICATED,
        "description": "Тяжелый серотониновый синдром",
        "mechanism": "Критическое повышение уровня серотонина",
        "recommendation": "Абсолютно противопоказано. Требуется период вымывания.",
    },
]

# Mapping of trade names to active ingredients (simplified)
TRADE_NAME_TO_INGREDIENT: Dict[str, str] = {
    "нурофен": "ибупрофен",
    "нурофен 400мг": "ибупрофен",
    "миг 400": "ибупрофен",
    "ибуклин": "ибупрофен",
    "панадол": "парацетамол",
    "эффералган": "парацетамол",
    "тайленол": "парацетамол",
    "аспирин": "ацетилсалициловая кислота",
    "кардиомагнил": "ацетилсалициловая кислота",
    "варфарин": "варфарин",
    "тромбоасс": "ацетилсалициловая кислота",
    "зиртек": "цетиризин",
    "кларитин": "лоратадин",
    "эутирокс": "левотироксин",
    "l-тироксин": "левотироксин",
    "золофт": "сертралин",
    "ципролет": "ципрофлоксацин",
    "ципринол": "ципрофлоксацин",
    "трихопол": "метронидазол",
    "флагил": "метронидазол",
    "липримар": "аторвастатин",
    "торвакард": "аторвастатин",
    "зокор": "симвастатин",
    "кордарон": "амиодарон",
    "клацид": "кларитромицин",
    "эналаприл": "эналаприл",
    "ренитек": "эналаприл",
    "лозап": "лозартан",
    "козаар": "лозартан",
    "сиофор": "метформин",
    "глюкофаж": "метформин",
}


class DrugInteractionService:
    """Service for checking drug interactions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._interaction_db = INTERACTION_DATABASE
        self._trade_name_map = TRADE_NAME_TO_INGREDIENT

    async def check_interactions(
        self,
        *,
        request: DrugInteractionCheckRequest,
        trace_id: Optional[str] = None,
    ) -> DrugInteractionCheckResponse:
        """Check for drug interactions between the product and other drugs."""

        interactions: List[DrugInteraction] = []
        product_ingredient = self._normalize_drug_name(
            request.active_ingredient or request.product_name or ""
        )

        for other_drug in request.other_drugs:
            other_ingredient = self._normalize_drug_name(
                other_drug.active_ingredient or other_drug.name
            )

            if not product_ingredient or not other_ingredient:
                continue

            # Check for interaction
            interaction = self._find_interaction(product_ingredient, other_ingredient)
            if interaction:
                interactions.append(DrugInteraction(
                    drug_a=request.product_name or product_ingredient,
                    drug_b=other_drug.name,
                    severity=interaction["severity"],
                    description=interaction["description"],
                    mechanism=interaction.get("mechanism"),
                    recommendation=interaction["recommendation"],
                    source="internal_db",
                ))

        has_major = any(i.severity in (InteractionSeverity.MAJOR, InteractionSeverity.CONTRAINDICATED) for i in interactions)
        has_contraindication = any(i.severity == InteractionSeverity.CONTRAINDICATED for i in interactions)

        logger.info(
            "drug_interaction.check product_id=%s checked=%d found=%d major=%s",
            request.product_id,
            len(request.other_drugs),
            len(interactions),
            has_major,
        )

        return DrugInteractionCheckResponse(
            product_id=request.product_id,
            product_name=request.product_name,
            interactions=interactions,
            has_major_interaction=has_major,
            has_contraindication=has_contraindication,
            checked_drugs_count=len(request.other_drugs),
            meta={
                "source": "internal_db",
                "trace_id": trace_id,
            },
        )

    async def check_single_interaction(
        self,
        *,
        drug_a: str,
        drug_b: str,
    ) -> Optional[DrugInteraction]:
        """Check interaction between two drugs by name."""
        ingredient_a = self._normalize_drug_name(drug_a)
        ingredient_b = self._normalize_drug_name(drug_b)

        if not ingredient_a or not ingredient_b:
            return None

        interaction = self._find_interaction(ingredient_a, ingredient_b)
        if interaction:
            return DrugInteraction(
                drug_a=drug_a,
                drug_b=drug_b,
                severity=interaction["severity"],
                description=interaction["description"],
                mechanism=interaction.get("mechanism"),
                recommendation=interaction["recommendation"],
                source="internal_db",
            )
        return None

    def get_common_interactions_for_drug(self, drug_name: str) -> List[DrugInteraction]:
        """Get list of common interactions for a drug."""
        ingredient = self._normalize_drug_name(drug_name)
        if not ingredient:
            return []

        interactions = []
        for entry in self._interaction_db:
            drug_a = entry["drug_a"].lower()
            drug_b = entry["drug_b"].lower()

            if ingredient in drug_a or ingredient in drug_b:
                other = drug_b if ingredient in drug_a else drug_a
                interactions.append(DrugInteraction(
                    drug_a=drug_name,
                    drug_b=other.capitalize(),
                    severity=entry["severity"],
                    description=entry["description"],
                    mechanism=entry.get("mechanism"),
                    recommendation=entry["recommendation"],
                    source="internal_db",
                ))

        return interactions

    def _normalize_drug_name(self, name: str) -> str:
        """Normalize drug name to active ingredient."""
        if not name:
            return ""

        normalized = name.strip().lower()
        # Remove common suffixes
        normalized = re.sub(r'\s*\d+\s*(мг|г|мл|таб|капс).*$', '', normalized)
        normalized = normalized.strip()

        # Check if it's a trade name
        if normalized in self._trade_name_map:
            return self._trade_name_map[normalized]

        return normalized

    def _find_interaction(self, ingredient_a: str, ingredient_b: str) -> Optional[Dict[str, Any]]:
        """Find interaction between two ingredients in the database."""
        for entry in self._interaction_db:
            db_drug_a = entry["drug_a"].lower()
            db_drug_b = entry["drug_b"].lower()

            # Check both directions
            if (ingredient_a in db_drug_a and ingredient_b in db_drug_b) or \
               (ingredient_a in db_drug_b and ingredient_b in db_drug_a):
                return entry

        return None


# Singleton instance
_drug_interaction_service: Optional[DrugInteractionService] = None


def get_drug_interaction_service(settings: Settings) -> DrugInteractionService:
    """Get or create drug interaction service instance."""
    global _drug_interaction_service
    if _drug_interaction_service is None:
        _drug_interaction_service = DrugInteractionService(settings)
    return _drug_interaction_service
