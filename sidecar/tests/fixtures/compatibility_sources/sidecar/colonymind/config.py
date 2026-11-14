"""Frozen compatibility oracle for the renderer behavior under comparison."""

from __future__ import annotations

from typing import Any, Iterable

CONSTITUTION_CHARS = 1500

CONSTITUTION_ITEMS = 12

CONSTITUTION_ITEM_CHARS = 160

def constitution_list(value: Any) -> list[str]:
    """``agent.values`` or ``agent.boundaries`` as the constitution reads them: at most 12 distinct
    one-line strings of 1 to 160 characters, in the owner's order."""
    if not isinstance(value, (list, tuple)):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if 1 <= len(text) <= CONSTITUTION_ITEM_CHARS and text not in items:
            items.append(text)
        if len(items) >= CONSTITUTION_ITEMS:
            break
    return items

def render_constitution(identity: Any, *, limit: int | None = CONSTITUTION_CHARS) -> str:
    """The constitution as one paragraph: ``You are <name>. Your values: a; b. Your boundaries: x; y.``

    Empty lists are omitted. Over ``limit`` (None: no limit, which ``constitution_length`` uses so init and
    doctor can say how far over a file is; init refuses one) whole values are held back from the last, then
    boundaries, and the text ends saying how many: never a cut mid-item, and the boundaries go last.
    """
    agent = identity.get("agent") if isinstance(identity, dict) else None
    agent = agent if isinstance(agent, dict) else {}
    name = " ".join(str(agent.get("name") or "").split())
    values, boundaries = constitution_list(agent.get("values")), constitution_list(agent.get("boundaries"))
    held = 0
    while True:
        text = " ".join(part for part in (
            f"You are {name}." if name else "", "Your values: " + "; ".join(values) + "." if values else "",
            "Your boundaries: " + "; ".join(boundaries) + "." if boundaries else "",
            f"({held} more item{'' if held == 1 else 's'} held back: over the {limit}-character limit.)" if held else "")
            if part)
        if limit is None or len(text) <= limit or not (values or boundaries):
            return text if limit is None else text[:limit]
        (values or boundaries).pop()
        held += 1
