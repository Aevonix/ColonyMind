"""Dependency-free hand identity rendering shared by core and host adapters."""

IDENTITY_CHARS = 900
NEUTRAL_IDENTITY = (
    "Return everything to your main conversation. Do not contact anyone from here. "
    "Be truthful and precise. Give receipts and uncertainty."
)


class BriefTooLarge(ValueError):
    """The required task cannot fit without changing its instructions."""


def render_identity(*, name: str = "Hand", persona_addendum: str = "", style: str = "") -> str:
    """Render host identity, neutral guidance, then host style without trimming."""
    persona = persona_addendum.replace("<name>", name).replace("{name}", name)
    identity = " ".join(part for part in (persona, NEUTRAL_IDENTITY, style) if part)
    if len(identity) > IDENTITY_CHARS:
        raise BriefTooLarge("identity exceeds 900 characters")
    return identity
