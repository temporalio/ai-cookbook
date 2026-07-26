import re
from models.signals import ContentSignals
from models.verdict import Verdict

_BANNED_KEYWORDS = ["buy now", "click here", "free money", "guaranteed winner"]


def _hard_block(signals: ContentSignals) -> Verdict | None:
    """Return a block Verdict if any hard rule matches, otherwise None."""
    text_lower = signals.text.lower()

    for keyword in _BANNED_KEYWORDS:
        if keyword in text_lower:
            return Verdict(
                classification="block",
                confidence=1.0,
                reasoning=f"Hard rule: contains banned keyword '{keyword}'.",
                overridden_by_hard_rule=True,
            )

    if re.search(r"\b\d{3}[-.()]?\d{3}[-.]?\d{4}\b", signals.text):
        return Verdict(
            classification="block",
            confidence=1.0,
            reasoning="Hard rule: contains phone number (privacy policy violation).",
            overridden_by_hard_rule=True,
        )

    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", signals.text):
        return Verdict(
            classification="block",
            confidence=1.0,
            reasoning="Hard rule: contains email address (privacy policy violation).",
            overridden_by_hard_rule=True,
        )

    return None


def apply_hard_rules(signals: ContentSignals, llm_verdict: Verdict) -> Verdict:
    """Post-filter: override the LLM verdict if a hard rule matches.

    When a rule fires, the LLM's original reasoning is embedded in the
    returned verdict so the override is auditable.
    """
    if llm_verdict.classification == "block":
        return llm_verdict

    hard = _hard_block(signals)
    if hard is None:
        return llm_verdict

    return Verdict(
        classification=hard.classification,
        confidence=hard.confidence,
        overridden_by_hard_rule=True,
        reasoning=(
            f"{hard.reasoning}\n\n"
            f"[LLM classified as '{llm_verdict.classification}' — "
            f"reasoning: {llm_verdict.reasoning}]"
        ),
    )
