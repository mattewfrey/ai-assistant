from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langsmith import traceable
import yaml

from ..intents import ActionChannel, ActionType, IntentType
from ..models.assistant import AssistantAction, AssistantMeta, AssistantResponse, Reply
from .dialog_state_store import DialogStateStore, get_dialog_state_store
from .debug_meta import DebugMetaBuilder
from .router import CONFIG_PATH, RouterResult
from .user_profile_store import UserProfile, UserProfileStore, get_user_profile_store


@dataclass
class SlotHandlingResult:
    handled: bool
    assistant_response: Optional[AssistantResponse] = None
    slot_filling_used: bool = False


class SlotManager:
    """Manages slot filling without relying on the LLM."""

    def __init__(
        self,
        dialog_state_store: DialogStateStore | None = None,
        user_profile_store: UserProfileStore | None = None,
        config_path: Path | None = None,
    ) -> None:
        self._dialog_state_store = dialog_state_store or get_dialog_state_store()
        self._user_profile_store = user_profile_store or get_user_profile_store()
        self._config_path = config_path or CONFIG_PATH
        config = self._load_config(self._config_path)
        self._intent_slot_questions = self._build_slot_questions(config)
        self._dosage_form_keywords = self._load_dosage_forms(config)
        self._default_slot_questions = (config.get("defaults") or {}).get("slot_questions") or {}
        self._age_regex = re.compile(r"\b(?P<age>\d{1,2})\b")
        self._price_regex = re.compile(r"(?:до|максимум)\s*(?P<price>\d{2,5})", re.IGNORECASE)

    @traceable(run_type="chain", name="slot_manager_followup")
    def try_handle_followup(
        self,
        *,
        request_message: str,
        conversation_id: str,
        user_profile: UserProfile | None,
        debug_builder: DebugMetaBuilder | None = None,
        trace_id: str | None = None,
    ) -> SlotHandlingResult:
        state = self._dialog_state_store.get_state(conversation_id)
        if not state or not state.pending_slots:
            return SlotHandlingResult(handled=False)

        extracted = self._extract_slots(request_message, user_profile)
        merged_slots = {**state.slots, **{k: v for k, v in extracted.items() if v is not None}}
        merged_slots = self._apply_profile_defaults(merged_slots, user_profile)

        remaining = [slot for slot in state.pending_slots if not merged_slots.get(slot)]
        if remaining:
            next_slot = remaining[0]
            prompt = state.slot_questions.get(next_slot) or self._question_for_slot(state.current_intent, next_slot)
            self._dialog_state_store.upsert_state(
                conversation_id,
                slots=merged_slots,
                pending_slots=remaining,
                slot_questions=state.slot_questions,
                last_prompt=prompt,
            )
            response = AssistantResponse(
                reply=Reply(text=prompt),
                actions=[],
                meta=AssistantMeta(
                    top_intent=getattr(state.current_intent, "value", None),
                    debug={"slot_filling_used": True, "slot_prompt_pending": True},
                ),
            )
            response.meta.debug = (response.meta.debug or {}) | {"slot_filling_used": True}
            if debug_builder:
                debug_builder.set_slot_filling_used(True).set_pending_slots(True).add_intent(
                    getattr(state.current_intent, "value", None)
                )
            logger.info(
                "trace_id=%s user_id=%s intent=%s slot_followup pending=%s",
                trace_id or "-",
                getattr(user_profile, "user_id", None) or "-",
                getattr(state.current_intent, "value", state.current_intent) if state.current_intent else "-",
                state.pending_slots,
            )
            return SlotHandlingResult(
                handled=True,
                assistant_response=response,
                slot_filling_used=True,
            )

        self._dialog_state_store.clear_state(conversation_id)
        action_response = self._build_action_response(
            intent=state.current_intent,
            channel=state.channel,
            parameters=merged_slots,
            slot_filling_used=True,
            slot_prompt_pending=False,
        )
        return SlotHandlingResult(handled=True, assistant_response=action_response, slot_filling_used=True)

    @traceable(run_type="chain", name="slot_manager_handle_router_result")
    def handle_router_result(
        self,
        *,
        router_result: RouterResult,
        conversation_id: str,
        user_profile: UserProfile | None = None,
        debug_builder: DebugMetaBuilder | None = None,
        trace_id: str | None = None,
    ) -> AssistantResponse:
        slots = self._apply_profile_defaults(dict(router_result.slots), user_profile)
        slot_questions = router_result.slot_questions or self._intent_slot_questions.get(router_result.intent, {})
        pending_slots = [slot.name for slot in router_result.missing_slots if not slots.get(slot.name)]
        if pending_slots:
            prompt = next(
                (slot.prompt for slot in router_result.missing_slots if slot.name == pending_slots[0]),
                self._question_for_slot(router_result.intent, pending_slots[0]),
            )
            self._dialog_state_store.upsert_state(
                conversation_id,
                current_intent=router_result.intent,
                channel=router_result.channel,
                slots=slots,
                pending_slots=pending_slots,
                slot_questions=slot_questions,
                last_prompt=prompt,
            )
            if debug_builder:
                debug_builder.set_slot_filling_used(True).set_pending_slots(True).add_intent(
                    getattr(router_result.intent, "value", None)
                )
            return AssistantResponse(
                reply=Reply(text=prompt),
                actions=[],
                meta=AssistantMeta(
                    top_intent=getattr(router_result.intent, "value", None),
                    confidence=router_result.confidence,
                    debug={"slot_filling_used": True},
                ),
            )
        self._dialog_state_store.clear_state(conversation_id)
        if debug_builder:
            debug_builder.set_slot_filling_used(False).set_pending_slots(False).add_intent(
                getattr(router_result.intent, "value", None)
            )
        logger.info(
            "trace_id=%s user_id=%s intent=%s slot_handle slots=%s pending=%s",
            trace_id or "-",
            getattr(user_profile, "user_id", None) or "-",
            getattr(router_result.intent, "value", router_result.intent) if router_result.intent else "-",
            list(slots.keys()),
            pending_slots,
        )
        return self._build_action_response(
            intent=router_result.intent,
            channel=router_result.channel,
            parameters=slots,
            slot_filling_used=False,
        )

    def _build_action_response(
        self,
        intent: IntentType | None,
        channel: ActionChannel | None,
        parameters: Dict[str, Any],
        slot_filling_used: bool,
        slot_prompt_pending: bool = False,
    ) -> AssistantResponse:
        action = AssistantAction(
            type=ActionType.CALL_PLATFORM_API,
            intent=intent,
            channel=channel,
            parameters=parameters,
        )
        meta = AssistantMeta(
            top_intent=getattr(intent, "value", None),
            confidence=0.95,
            debug={"slot_filling_used": slot_filling_used, "slot_prompt_pending": slot_prompt_pending},
        )
        return AssistantResponse(reply=Reply(text="Понял запрос, выполняю."), actions=[action], meta=meta)

    def _extract_slots(self, message: str, user_profile: UserProfile | None) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        age = self._extract_age(message)
        if age is not None:
            slots["age"] = age
            if user_profile:
                self._user_profile_store.update_preferences(user_profile.user_id, age=age)
        price = self._extract_price(message)
        if price is not None and price > 0:
            slots["price_max"] = price
            if user_profile:
                self._user_profile_store.update_preferences(user_profile.user_id, default_max_price=price)
        form = self._extract_form(message)
        if form is not None:
            slots["dosage_form"] = form
            if user_profile:
                existing = user_profile.preferences.preferred_dosage_forms or []
                updated_forms = list(existing)
                if form not in updated_forms:
                    updated_forms.append(form)
                self._user_profile_store.update_preferences(
                    user_profile.user_id,
                    preferred_dosage_forms=updated_forms,
                )
        return slots

    def _apply_profile_defaults(self, slots: Dict[str, Any], user_profile: UserProfile | None) -> Dict[str, Any]:
        if not user_profile:
            return slots
        prefs = user_profile.preferences
        if not slots.get("age") and getattr(prefs, "age", None):
            slots["age"] = prefs.age
        default_price = getattr(prefs, "default_max_price", None)
        if not slots.get("price_max") and default_price is not None:
            slots["price_max"] = default_price
        preferred_forms = getattr(prefs, "preferred_dosage_forms", None) or []
        if not slots.get("dosage_form") and preferred_forms:
            slots["dosage_form"] = preferred_forms[0]
        return slots

    def _question_for_slot(self, intent: IntentType | None, slot: str) -> str:
        if intent in self._intent_slot_questions:
            question = self._intent_slot_questions[intent].get(slot)
            if question:
                return question
        return self._default_slot_questions.get(slot, "Уточните, пожалуйста.")

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

    def _extract_form(self, message: str) -> Optional[str]:
        normalized = message.lower()
        for form, keywords in self._dosage_form_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                return form
        return None

    def _build_slot_questions(self, config: Dict[str, Any]) -> Dict[IntentType, Dict[str, str]]:
        mapping: Dict[IntentType, Dict[str, str]] = {}
        for intent_name, intent_cfg in (config.get("intents") or {}).items():
            try:
                intent = IntentType(intent_name)
            except ValueError:
                continue
            mapping[intent] = {str(k): str(v) for k, v in (intent_cfg.get("slot_questions") or {}).items()}
        return mapping

    def _load_dosage_forms(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        entities = config.get("entities") or {}
        dosage_config = entities.get("dosage_forms") or {}
        if not dosage_config:
            return {
                "tablets": ["таблет", "табл."],
                "syrup": ["сироп"],
                "spray": ["спрей"],
                "capsules": ["капсул"],
            }
        return {str(form): [str(keyword).lower() for keyword in keywords or []] for form, keywords in dosage_config.items()}

    @staticmethod
    def _load_config(config_path: Path) -> Dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}


_slot_manager: SlotManager | None = None


def get_slot_manager() -> SlotManager:
    global _slot_manager
    if _slot_manager is None:
        _slot_manager = SlotManager()
    return _slot_manager
