"""Host-authored text policies. Empty policies preserve the host's text."""
from __future__ import annotations

from collections.abc import Mapping
import re

VALIDATOR_NAMES = ("hedge", "agency", "causal", "apology", "adjustment", "tone")


def patterns(value) -> tuple[re.Pattern, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("validators must be a list of nonempty regular expressions")
    return tuple(re.compile(item, re.IGNORECASE) for item in value)


def validate_validators(value) -> dict:
    """Serializable host regex lists; no framework-authored language policy."""
    if not isinstance(value, Mapping) or set(value) - set(VALIDATOR_NAMES):
        raise ValueError("drift validators must map hedge, agency, causal, apology, adjustment or tone to patterns")
    for supplied in value.values():
        patterns(supplied)
    return dict(value)


def replacements(value) -> tuple[tuple[re.Pattern, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("text replacements must be a list")
    result = []
    for item in value:
        if (not isinstance(item, Mapping) or set(item) != {"pattern", "replacement"}
                or not isinstance(item["replacement"], str)):
            raise ValueError("each text replacement needs pattern and replacement strings")
        pattern = patterns([item["pattern"]])[0]
        # Check replacement backreferences even when no current text matches.
        pattern.sub(item["replacement"], "")
        result.append((pattern, item["replacement"]))
    return tuple(result)


def render(text: str | None, policy=()) -> str | None:
    if text is None:
        return None
    for pattern, replacement in replacements(policy):
        text = pattern.sub(replacement, text)
    return text
