from __future__ import annotations

import re

from ..models import AssistantMeta
from .metrics import get_metrics_service


class SafetyFilter:
    """Minimal guardrail layer that enforces medical hygiene in replies."""

    SAFE_REPLY = (
        "По вопросам дозировок, схем лечения и комбинирования препаратов обязательно "
        "проконсультируйтесь с врачом. Я могу помочь подобрать безрецептурные средства "
        "для симптоматического облегчения."
    )

    DEFAULT_DISCLAIMER = (
        "Информация, предоставляемая ассистентом, не заменяет консультацию врача или "
        "фармацевта. Перед началом, изменением или прекращением лечения обязательно "
        "обратитесь к специалисту. Ассистент помогает с подбором товаров из ассортимента "
        "аптеки и не ставит диагнозы, не назначает препараты и не отменяет назначения врача."
    )

    _DISCLAIMER_MIN_LENGTH = 120

    # TODO: вынести конфигурацию опасных паттернов и стратегий фильтрации в настройки.
    _DANGEROUS_PATTERNS = [
        re.compile(r"принимайт[ея]\s+по\s+\d+.*раз[аи]?\s+в\s+день", re.IGNORECASE | re.DOTALL),
        re.compile(r"курс\s+\d+\s*(?:дн|недел|месяц)", re.IGNORECASE),
        re.compile(r"в\s+течение\s+\d+\s*(?:дн|недел|месяц)", re.IGNORECASE),
        re.compile(r"отмен[яе]йте?\s+(?:при[её]м|назначени[ея])", re.IGNORECASE),
        re.compile(r"(?:совмещайт|комбинируйте)\s+.*(антибиот|антикоагулянт|алкогол)", re.IGNORECASE),
    ]

    @classmethod
    def sanitize_reply(cls, text: str) -> str:
        """Replace unsafe payloads with a standard warning."""

        if not text or not text.strip():
            get_metrics_service().record_safety_filter_trigger()
            return cls.SAFE_REPLY
        clean_text = text.strip()
        for pattern in cls._DANGEROUS_PATTERNS:
            if pattern.search(clean_text):
                get_metrics_service().record_safety_filter_trigger()
                return cls.SAFE_REPLY
        return clean_text

    @classmethod
    def ensure_disclaimer(cls, meta: AssistantMeta) -> AssistantMeta:
        """Enforce a minimum-length disclaimer in replies."""

        disclaimer = (meta.legal_disclaimer or "").strip()
        if len(disclaimer) < cls._DISCLAIMER_MIN_LENGTH:
            meta.legal_disclaimer = cls.DEFAULT_DISCLAIMER
        return meta

    @classmethod
    def is_safe(cls, text: str) -> bool:
        """Check if text passes safety filter without modifying it."""
        if not text or not text.strip():
            return False
        clean_text = text.strip()
        for pattern in cls._DANGEROUS_PATTERNS:
            if pattern.search(clean_text):
                return False
        return True


__all__ = ["SafetyFilter"]
