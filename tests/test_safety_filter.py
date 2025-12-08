from app.services.safety_filter import SafetyFilter


def test_dangerous_instruction_is_replaced() -> None:
    text = "Принимайте по 2 таблетки 3 раза в день в течение 7 дней."
    sanitized = SafetyFilter.sanitize_reply(text)
    assert sanitized == SafetyFilter.SAFE_REPLY


def test_safe_text_passes_through() -> None:
    text = "Леденцы от кашля облегчают симптомы."
    sanitized = SafetyFilter.sanitize_reply(text)
    assert sanitized == text

