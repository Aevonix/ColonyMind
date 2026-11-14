"""Plain words and the two output formats (Markdown and a standalone HTML page) every report shares."""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import List, Optional, Sequence

FORMATS = ("md", "html")


def plural(count: int, singular: str, plural_form: Optional[str] = None) -> str:
    """``1 task``, ``3 tasks``; ``plural_form`` for an irregular noun."""
    return f"{count} {singular if count == 1 else (plural_form or singular + 's')}"


def percent(part: float, whole: float) -> str:
    """A share as a whole percentage, ``0%`` for an empty whole."""
    return f"{round(100 * part / whole)}%" if whole else "0%"


def points(value: float, *, signed: bool = True) -> str:
    """A difference in percentage points: whole where it is whole, else one decimal; ASCII signs only."""
    rounded = round(value, 1)
    text = f"{rounded:.0f}" if abs(rounded - round(rounded)) < 0.05 else f"{rounded:.1f}"
    if text in {"-0", "0"}:
        return "0"
    return f"+{text}" if signed and not text.startswith("-") else text


def listing(items: Sequence[str], *, conjunction: str = "and") -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def duration(seconds: float) -> str:
    """A span in the largest whole unit that reads naturally."""
    seconds = max(0.0, float(seconds))
    if seconds < 59.5:
        return plural(round(seconds), "second")
    if seconds < 90 * 60:
        return plural(max(1, round(seconds / 60)), "minute")
    if seconds < 36 * 3600:
        return plural(round(seconds / 3600), "hour")
    return plural(round(seconds / 86400), "day")


def sentence(text: str) -> str:
    """``text`` starting with a capital and ending as a sentence ends."""
    value = " ".join(str(text or "").split())
    if not value:
        return value
    value = value[0].upper() + value[1:]
    return value if value[-1] in ".!?" else value + "."


@dataclass
class Block:
    """One piece of a report: a heading (``level`` 1 to 3), a paragraph or a bulleted list."""

    kind: str
    text: str = ""
    level: int = 2
    items: List[str] = field(default_factory=list)


@dataclass
class Document:
    title: str
    blocks: List[Block] = field(default_factory=list)

    def heading(self, text: str, level: int = 2) -> None:
        self.blocks.append(Block("heading", text=text, level=level))

    def paragraph(self, text: str) -> None:
        if text:
            self.blocks.append(Block("paragraph", text=text))

    def bullets(self, items: Sequence[str]) -> None:
        items = [item for item in items if item]
        if items:
            self.blocks.append(Block("list", items=list(items)))

    def text(self) -> str:
        """Every sentence of the report, one per line (what a test or a scan for jargon reads)."""
        lines = [self.title]
        for block in self.blocks:
            lines.extend([block.text] if block.kind != "list" else block.items)
        return "\n".join(lines)


def markdown(document: Document) -> str:
    lines = [f"# {document.title}", ""]
    for block in document.blocks:
        if block.kind == "heading":
            lines.extend([f"{'#' * block.level} {block.text}", ""])
        elif block.kind == "paragraph":
            lines.extend([block.text, ""])
        else:
            lines.extend([*(f"- {item}" for item in block.items), ""])
    return "\n".join(lines).rstrip() + "\n"


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --text: #1d1d1f; --muted: #555; --background: #ffffff; --rule: #d9d9de; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --text: #ececf1; --muted: #a8a8b3; --background: #17171a; --rule: #34343a; }}
}}
body {{ margin: 0 auto; max-width: 46rem; padding: 2rem 1rem; background: var(--background); color: var(--text);
       font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.25rem; border-top: 1px solid var(--rule); padding-top: 1rem; }}
h3 {{ font-size: 1.05rem; }} li {{ margin: 0.2rem 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def html(document: Document) -> str:
    parts = [f"<h1>{escape(document.title)}</h1>"]
    for block in document.blocks:
        if block.kind == "heading":
            level = min(6, block.level)
            parts.append(f"<h{level}>{escape(block.text)}</h{level}>")
        elif block.kind == "paragraph":
            parts.append(f"<p>{escape(block.text)}</p>")
        else:
            parts.append("<ul>" + "".join(f"<li>{escape(item)}</li>" for item in block.items) + "</ul>")
    return PAGE.format(title=escape(document.title), body="\n".join(parts))


def render(document: Document, fmt: str) -> str:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    return markdown(document) if fmt == "md" else html(document)


__all__ = ["Block", "Document", "FORMATS", "duration", "html", "listing", "markdown", "percent", "plural", "points",
           "render", "sentence"]
