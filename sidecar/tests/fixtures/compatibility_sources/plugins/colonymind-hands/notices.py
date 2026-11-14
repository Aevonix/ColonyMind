"""Frozen compatibility oracle for the renderer behavior under comparison."""

from __future__ import annotations

def compose(notices: list[dict]) -> str:
    lines = []
    for notice in notices:
        line = notice.get("line")
        if (not isinstance(line, str) or not line.strip() or len(line) > 200
                or any(character in line for character in ("\u2013", "\u2014", "\r", "\n"))):
            raise ValueError("Notice lines must be 1 to 200 characters without long dashes or newlines")
        lines.append(line)
    return "\n".join(lines)
