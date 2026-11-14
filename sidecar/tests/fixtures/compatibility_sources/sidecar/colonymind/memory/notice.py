"""Frozen compatibility oracle for the renderer behavior under comparison."""

from __future__ import annotations

from contextlib import closing

from datetime import date, datetime

import json

import re

from colonymind.secrets.claims import withheld_claims

from colonymind.secrets.detect import HIDDEN, replace_secrets

from .recall import render_memory_context

_ROLE_LABEL = re.compile(r"^(?:user input|assistant response|user|assistant)\s*:\s*", re.IGNORECASE)

_MARKUP = re.compile(r"\*+|`+|(?<!\w)__|__(?!\w)")

def _plain(text: str, *, cut_start: bool = False) -> str:
    """Words to read in a chat line: no markdown emphasis, no role label, no leading bullet. A passage cut from
    the middle of a message loses its first, partial word. Runs after withholding, so it never changes what the
    withholding matched."""
    text = " ".join(_MARKUP.sub("", text).split())
    text = _ROLE_LABEL.sub("", text).lstrip("-#>•·: ")
    if cut_start and text and not (text[0].isupper() or text[0].isdigit() or text[0] in "[(\"'\u201c\u2018"):
        text = text.split(" ", 1)[1] if " " in text else text
    return text

def _preview(value: str, *, cut_start: bool = False) -> str:
    text, _ = replace_secrets(value, lambda hit: HIDDEN)
    text = _plain(text, cut_start=cut_start)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("MEDIA:", "MEDIA :").replace("[[audio_as_voice]]", "[audio as voice]")
    text = text.replace("[[as_document]]", "[as document]")
    words = text.split()
    if len(words) <= 8 and len(" ".join(words)) <= 60:
        return " ".join(words)
    kept = []
    for word in words[:8]:
        if len(" ".join([*kept, word])) > 59:
            break
        kept.append(word)
    return " ".join(kept) + "…"

def _iso_date(value) -> str | None:
    try:
        text = str(value or "")
        if len(text) == 10:
            return date.fromisoformat(text).isoformat()
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None

def _date(row) -> str | None:
    event = row.get("event_time")
    return (_iso_date(row.get("event_at")) or
            (_iso_date(event.get("start")) if isinstance(event, dict) and event.get("status") == "resolved" else None))

def _recorded(row) -> str | None:
    return next((value for key in ("recorded_at", "ingested_at", "created_at")
                 if (value := _iso_date(row.get(key)))), None)

def _payload(row) -> dict:
    # Only producer-declared structured records carry metadata. Quoted JSON is still prose.
    if row.get("content_format") not in {"source_assertions_v1", "source_conversation_v1"}:
        return {}
    try:
        value = json.loads(row.get("content", ""))
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}

def _metadata(row):
    yield row
    for name in ("scope", "metadata", "claim", "subject", "input", "response", "original"):
        value = row.get(name)
        if isinstance(value, dict):
            yield from _metadata(value)
    for name in ("assertions", "claims"):
        for value in row.get(name) or ():
            if isinstance(value, dict):
                yield from _metadata(value)

def _other_person(row, contact_id) -> bool:
    for metadata in _metadata(row):
        if metadata.get("type", metadata.get("kind")) in {"person", "contact"}:
            value = metadata.get("id")
            if value and str(value) != contact_id:
                return True
        for name in ("contact_id", "person_id", "scope_contact_id", "subject_contact_id", "about_contact_id",
                     "subject_person_id", "about_person_id"):
            value = metadata.get(name)
            if value and str(value) != contact_id:
                return True
        for name in ("contact_ids", "person_ids", "about_contact_ids", "subject_contact_ids"):
            values = metadata.get(name)
            if isinstance(values, (list, tuple)) and any(value and str(value) != contact_id for value in values):
                return True
    return False

def _source_ids(row):
    named = list(row.get("source_turn_ids") or ())
    if row.get("source_turn_id"):
        named.append(row["source_turn_id"])
    uri = str(row.get("source_uri") or "")
    if uri.startswith("turn:"):
        named.append(uri[5:])
    return list(dict.fromkeys(str(value) for value in named))

def _sources(ledger, rows):
    if ledger is None:
        return {}
    named = list(dict.fromkeys(value for row in rows for value in _source_ids(row)))
    if not named:
        return {}
    result = {}
    with closing(ledger._connect()) as conn:
        for start in range(0, len(named), 400):
            batch = named[start:start + 400]
            result.update((row["turn_id"], dict(row)) for row in conn.execute(
                "SELECT turn_id, contact_id, scope, ingested_at FROM turn_sources WHERE turn_id IN ("
                + ",".join("?" for _ in batch) + ")", batch))
    return result

def _card(row, payload):
    claim = row.get("claim")
    card = dict(claim) if isinstance(claim, dict) else dict(row)
    members = payload.get("assertions") or []
    if members and isinstance(members[0], dict):
        card.update({key: payload[key] for key in ("subject", "predicate") if key in payload})
        card.update(members[0])
    elif isinstance(payload.get("input"), dict):
        card.update(payload["input"])
    text = row.get("title") or (claim if isinstance(claim, str) else None)
    value = card.get("value")
    labelled = f"{card['predicate']}: {value}" if card.get("predicate") and value is not None else value
    # A passage's own retrieval text reads better than its content, which may be a partial excerpt or, for an
    # incomplete conversation pair, an instruction to open the sources.
    passage = row.get("ranking_text") or row.get("content")
    text = text or card.get("quote") or labelled or passage or "Recalled record"
    card["excerpt"] = str(text)
    card["_cut_start"] = text is passage and row.get("excerpt_truncated") is True and not row.get("ranking_text")
    return card

def recall_notice_details(rows: list[dict], *, content: str, contact_id: str, ledger=None) -> list[str]:
    """Describe only complete records in the final packet, bounded and redacted before any preview is cut.

    The renderer emits its shared passages first, then one line per record. Matching those complete lines
    also excludes records removed by the final block fit, including a cut first line.
    """
    if not rows or not content:
        return []
    lines = render_memory_context(rows).splitlines()[-len(rows):]
    kept, offset = [], 0
    for row, line in zip(rows, lines):
        found = content.find(line, offset)
        if found >= 0:
            kept.append(row)
            offset = found + len(line)
    rows = kept
    sources = _sources(ledger, rows)
    cards, dates, credentials, hidden = [], [], [], 0
    for row in rows:
        payload = _payload(row)
        credentials.extend(_metadata(row))
        credentials.extend(_metadata(payload))
        credentials.extend(dict(member, **{key: payload[key] for key in ("subject", "predicate") if key in payload})
                           for member in payload.get("assertions") or () if isinstance(member, dict))
        source_rows = [sources[value] for value in _source_ids(row) if value in sources]
        if any(_other_person(value, contact_id) for value in [row, payload, *source_rows]):
            hidden += 1
            continue
        card = _card(row, payload)
        cards.append(card)
        dates.append(_date(card) or _date(row) or _recorded(card) or _recorded(row) or
                     next((value for source in source_rows if (value := _recorded(source))), None))
    # Pass all card words together, as the console does, including labels beside credential values.
    cards = withheld_claims([*cards, *credentials])[:len(cards)]
    limit = 8 - bool(hidden)
    result = [f"• {_preview(card['excerpt'], cut_start=card.get('_cut_start') is True)} ({when or 'date unknown'})"
              for card, when in zip(cards[:limit], dates[:limit])]
    if hidden:
        people = "another person" if hidden == 1 else "other people"
        result.append(f"• {hidden} about {people}, not shown")
    if len(cards) > limit:
        result.append(f"• and {len(cards) - limit} more")
    return result
