from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langsmith import traceable
import yaml

from ..intents import ActionChannel, IntentType
from ..models import ChatRequest, UserProfile
from .dialog_state import DialogState

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "router_config.yaml"
DEFAULT_PROMPT = "Уточните, пожалуйста."


@dataclass
class SlotDefinition:
    name: str
    prompt: str


@dataclass
class RouterRule:
    intent: IntentType
    channel: ActionChannel
    triggers: List[str] = field(default_factory=list)
    slots_required: List[str] = field(default_factory=list)
    slot_questions: Dict[str, str] = field(default_factory=dict)


@dataclass
class RouterConfig:
    rules: List[RouterRule]
    default_slot_questions: Dict[str, str]
    brand_keywords: List[str]
    symptom_keywords: List[str]
    dosage_forms: Dict[str, List[str]]


@dataclass
class RouterResult:
    matched: bool
    intent: Optional[IntentType] = None
    channel: Optional[ActionChannel] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[SlotDefinition] = field(default_factory=list)
    slot_questions: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    router_matched: bool = False

    @property
    def extracted_slots(self) -> Dict[str, Any]:
        return self.slots


class RouterService:
    """Config-driven router that tries to map queries without LLM."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or CONFIG_PATH
        self._config = self._load_config(self._config_path)
        self._rules = self._config.rules
        self._brand_keywords = self._config.brand_keywords
        self._symptom_keywords = self._config.symptom_keywords
        self._dosage_form_keywords = self._config.dosage_forms
        self._symptom_regex = re.compile(r"(?:от|для)\s+(?P<symptom>[а-яa-z\s]+)", re.IGNORECASE)
        self._price_regex = re.compile(r"(?:до|максимум|не дороже)\s*(?P<price>\d{2,5})", re.IGNORECASE)
        self._age_regex = re.compile(r"\b(?P<age>\d{1,2})\b")

    @traceable(run_type="chain", name="router_match")
    def match(
        self,
        *,
        request: ChatRequest,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
        debug_builder: Any | None = None,
        trace_id: str | None = None,
    ) -> RouterResult:
        message = (request.message or "").strip()
        if not message:
            return RouterResult(matched=False, router_matched=False)
        normalized = message.lower()

        rule_match = self._match_rules(message, normalized)
        if rule_match:
            rule = rule_match
            result = self._build_rule_result(
                rule=rule,
                message=message,
                normalized_message=normalized,
                user_profile=user_profile,
                dialog_state=dialog_state,
            )
            if debug_builder:
                debug_builder.set_router_matched(True).add_intent(result.intent.value if result.intent else None)
            logger.info(
                "trace_id=%s user_id=%s intent=%s Router matched rule slots=%s",
                trace_id or "-",
                getattr(request, "user_id", None) or "-",
                getattr(result.intent, "value", result.intent) if result.intent else "-",
                list(result.slots.keys()),
            )
            return result

        product_result = self._detect_product_query(message, normalized, user_profile, dialog_state)
        if product_result:
            if debug_builder:
                debug_builder.set_router_matched(True).add_intent(
                    product_result.intent.value if product_result.intent else None
                )
            logger.info(
                "trace_id=%s user_id=%s intent=%s Router product shortcut slots=%s",
                trace_id or "-",
                getattr(request, "user_id", None) or "-",
                getattr(product_result.intent, "value", product_result.intent) if product_result.intent else "-",
                list(product_result.slots.keys()),
            )
            return product_result

        return RouterResult(matched=False, router_matched=False)

    def _match_rules(self, message: str, normalized_message: str) -> RouterRule | None:
        for rule in self._rules:
            if self._triggers_hit(rule, normalized_message, message):
                return rule
        return None

    def _triggers_hit(self, rule: RouterRule, normalized_message: str, message: str) -> bool:
        if rule.triggers and any(trigger in normalized_message for trigger in rule.triggers):
            return True
        if rule.intent == IntentType.FIND_BY_SYMPTOM:
            return self._matches_symptom_keywords(normalized_message) or bool(self._symptom_regex.search(message))
        return False

    def _build_rule_result(
        self,
        *,
        rule: RouterRule,
        message: str,
        normalized_message: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> RouterResult:
        slots = self._prefill_slots(rule.intent, message, normalized_message, user_profile, dialog_state)
        missing_slots = self._missing_slots(rule, slots)
        return RouterResult(
            matched=True,
            intent=rule.intent,
            channel=rule.channel,
            slots=slots,
            missing_slots=missing_slots,
            slot_questions=self._questions_map(rule),
            confidence=0.95 if not missing_slots else 0.8,
            router_matched=True,
        )

    def _detect_product_query(
        self,
        message: str,
        normalized_message: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> RouterResult | None:
        if len(message) > 96:
            return None
        brand_hit = any(brand in normalized_message for brand in self._brand_keywords)
        looks_like_product = brand_hit or self._looks_like_short_query(message)
        if not looks_like_product:
            return None
        slots = {
            "name": message.strip(),
        }
        slots.update(self._prefill_slots(IntentType.FIND_PRODUCT_BY_NAME, message, normalized_message, user_profile, dialog_state))
        return RouterResult(
            matched=True,
            intent=IntentType.FIND_PRODUCT_BY_NAME,
            channel=ActionChannel.DATA,
            slots=slots,
            slot_questions={},
            confidence=0.75 if brand_hit else 0.65,
            router_matched=True,
        )

    def _prefill_slots(
        self,
        intent: IntentType | None,
        message: str,
        normalized_message: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        if intent == IntentType.FIND_BY_SYMPTOM:
            symptom = self._extract_symptom_phrase(message, normalized_message)
            if symptom:
                slots.setdefault("symptom", symptom)
        age = self._extract_age(message)
        if age is not None:
            slots.setdefault("age", age)
        price = self._extract_price(message)
        if price is not None:
            slots.setdefault("price_max", price)
        dosage_form = self._extract_dosage_form(normalized_message)
        if dosage_form:
            slots.setdefault("dosage_form", dosage_form)

        if user_profile:
            prefs = user_profile.preferences
            slots.setdefault("age", getattr(prefs, "age", None))
            slots.setdefault("price_max", getattr(prefs, "default_max_price", None))
            preferred_forms = getattr(prefs, "preferred_dosage_forms", None)
            if preferred_forms:
                slots.setdefault("preferred_dosage_forms", preferred_forms)
                slots.setdefault("dosage_form", preferred_forms[0])

        if dialog_state and dialog_state.slots:
            slots = {**dialog_state.slots, **{k: v for k, v in slots.items() if v is not None}}

        return {k: v for k, v in slots.items() if v is not None}

    def _missing_slots(self, rule: RouterRule, slots: Dict[str, Any]) -> List[SlotDefinition]:
        missing: List[SlotDefinition] = []
        for slot_name in rule.slots_required:
            if slot_name not in slots or slots[slot_name] in (None, "", []):
                prompt = self._questions_map(rule).get(slot_name, DEFAULT_PROMPT)
                missing.append(SlotDefinition(name=slot_name, prompt=prompt))
        return missing

    def _questions_map(self, rule: RouterRule) -> Dict[str, str]:
        result = dict(self._config.default_slot_questions)
        result.update(rule.slot_questions or {})
        return result

    def _extract_age(self, message: str) -> Optional[int]:
        match = self._age_regex.search(message)
        if not match:
            return None
        try:
            return int(match.group("age"))
        except ValueError:
            return None

    def _extract_price(self, message: str) -> Optional[int]:
        match = self._price_regex.search(message)
        if not match:
            return None
        try:
            return int(match.group("price"))
        except ValueError:
            return None

    def _extract_symptom_phrase(self, message: str, normalized_message: str) -> Optional[str]:
        match = self._symptom_regex.search(message)
        if match:
            return match.group("symptom").strip()
        for keyword in self._symptom_keywords:
            if keyword in normalized_message:
                return keyword
        return None

    def _extract_dosage_form(self, normalized_message: str) -> Optional[str]:
        for form, keywords in self._dosage_form_keywords.items():
            if any(keyword in normalized_message for keyword in keywords):
                return form
        return None

    def _matches_symptom_keywords(self, normalized_message: str) -> bool:
        return any(keyword in normalized_message for keyword in self._symptom_keywords)

    def _looks_like_short_query(self, message: str) -> bool:
        tokens = message.split()
        if not (1 <= len(tokens) <= 6):
            return False
        has_digit = any(char.isdigit() for char in message)
        has_upper = any(char.isupper() for char in message)
        return has_digit or has_upper

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _load_config(config_path: Path) -> RouterConfig:
        if not config_path.exists():
            raise FileNotFoundError(f"Router config not found at {config_path}")
        with config_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        defaults = (data.get("defaults") or {}).get("slot_questions") or {}
        intents = data.get("intents") or {}
        rules: List[RouterRule] = []
        for intent_name, config in intents.items():
            try:
                intent = IntentType(intent_name)
            except ValueError:
                continue
            channel_value = config.get("channel", "other")
            try:
                channel = ActionChannel(channel_value)
            except ValueError:
                channel = ActionChannel.OTHER
            triggers = [str(trigger).lower() for trigger in (config.get("triggers") or [])]
            slots_required = [str(slot) for slot in config.get("slots_required") or []]
            slot_questions = {str(k): str(v) for k, v in (config.get("slot_questions") or {}).items()}
            rules.append(
                RouterRule(
                    intent=intent,
                    channel=channel,
                    triggers=triggers,
                    slots_required=slots_required,
                    slot_questions=slot_questions,
                )
            )
        entities = data.get("entities") or {}
        brand_keywords = [str(value).lower() for value in entities.get("product_brands") or []]
        symptom_keywords = [str(value).lower() for value in entities.get("symptom_keywords") or []]
        dosage_forms_raw = entities.get("dosage_forms") or {}
        dosage_forms: Dict[str, List[str]] = {}
        for form, keywords in dosage_forms_raw.items():
            dosage_forms[str(form)] = [str(keyword).lower() for keyword in keywords or []]
        return RouterConfig(
            rules=rules,
            default_slot_questions={str(k): str(v) for k, v in defaults.items()},
            brand_keywords=brand_keywords,
            symptom_keywords=symptom_keywords,
            dosage_forms=dosage_forms,
        )


_router_service: RouterService | None = None


def get_router_service() -> RouterService:
    global _router_service
    if _router_service is None:
        _router_service = RouterService()
    return _router_service
