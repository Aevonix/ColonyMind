"""Text cut into pieces for a decision model, every secret hidden (docs/HERMES-GATES.md).

Secrets are found over the whole text before it is cut, never piece by piece: a value whose label ends one piece and
whose value starts the next, or a value cut in two, is still found whole and hidden in every piece it touches. Each
piece keeps its offsets in the text, so a notice can be placed before it; only the text sent has secrets hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Beyond the last character a gate may send, this much more of the text is read for secrets (my choice: longer
#: than a private key block), so a value that starts in a piece and runs past it is still found whole.
LOOKAHEAD = 8192


@dataclass(frozen=True)
class Piece:
    """``text[start:end]`` of the text it was cut from, and ``sent``: that piece with every secret hidden."""

    start: int
    end: int
    sent: str


def hidden_spans(text: str) -> list[tuple[int, int]]:
    """The spans of ``text`` a gate never sends: every value the capture detector finds, and every reference to a
    sealed value, merged where they touch."""
    from colonymind_hermes.secrets import REFERENCE, find_secrets
    spans = sorted([(hit.start, hit.end) for hit in find_secrets(text)]
                   + [match.span() for match in REFERENCE.finditer(text)])
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def hide(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> str:
    """``text[start:end]`` with each span that touches it replaced by one hidden marker, then read once more on its
    own (whatever the piece alone shows is hidden too)."""
    from colonymind_hermes.secrets import HIDDEN, hide_secrets
    parts, cursor = [], start
    for left, right in spans:
        if right <= start or left >= end:
            continue
        parts += [text[cursor:max(cursor, left)], HIDDEN]
        cursor = min(end, right)
    parts.append(text[cursor:end])
    return hide_secrets("".join(parts))


def folded(text: str) -> int:
    """How long ``text`` is as a model tier reads it: whitespace folded to single spaces."""
    return len(" ".join(text.split()))


def pieces(text: str, room: int, spans: list[tuple[int, int]], *, start: int = 0, end: int | None = None,
           limit: int | None = None) -> list[Piece]:
    """``text[start:end]`` cut in order into pieces whose sent text, folded, is at most ``room`` characters (a hidden
    marker can be longer than the value it hides, so a piece is cut shorter until it fits); at most ``limit``."""
    end = len(text) if end is None else end
    room = max(1, int(room))
    out: list[Piece] = []
    position = start
    while position < end and (limit is None or len(out) < limit):
        stop = min(end, position + room)
        while True:
            sent = hide(text, position, stop, spans)
            excess = folded(sent) - room
            if excess <= 0 or stop - position <= 1:
                break
            stop = max(position + 1, stop - excess)
        out.append(Piece(position, stop, sent))
        position = stop
    return out


__all__ = ["LOOKAHEAD", "Piece", "folded", "hidden_spans", "hide", "pieces"]
