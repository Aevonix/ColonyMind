"""The per-contact template digest (architecture 4.7 item 4).

Who the person is, how they are reachable, when the agent last talked with
them, what is open, and what they have said themselves (their own sources'
claims), in at most ``MAX_CHARS`` characters. The digest is read in the
contact's own context ("About this person") and composes messages to them, so
it never carries what the owner set for them or thinks of them: permission,
cadence, the tier and who introduced them stay in the owner's
``colonymind_people inspect``, read from the columns. The mind writes it daily for contacts with a recent interaction
(``Mind._digests``) only when no digest exists. Fact changes queue a debounced
background rebuild; the night retries only digests still marked stale.

Each digest records its lineage: the claim ids it was built from (``digest_sources``) and its writer
(``digest_writer``: ``template`` here, ``night`` for the generated one). A digest is shown only while every
claim it cites is still live and the person's own (``checked_digest``): one that cites an erased or retracted
claim, a claim whose source is no longer theirs (an attribution correction), or a source that is not a claim, is
withheld. Its text leaves the contact's record and it waits for a rebuild
(``digest_withheld_at``), so a claim that comes back later (a restore) never revives the old text.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_CHARS = 600
MAX_CLAIMS = 4
WRITER_TEMPLATE, WRITER_NIGHT = "template", "night"


def _utc(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _ago(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 3600:
        return f"{max(1, seconds // 60)} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def _field(contact: Any, name: str, default: Any = None) -> Any:
    if isinstance(contact, Mapping):
        return contact.get(name, default)
    return getattr(contact, name, default)


def render_digest(contact: Any, *, claims: Iterable[str], counts: Mapping[str, Any], last_interaction_at: Any,
                  now: datetime) -> str:
    """The digest text, at most ``MAX_CHARS`` characters (an ellipsis marks a cut)."""
    contact_id = str(_field(contact, "contact_id", "") or "")
    name = str(_field(contact, "display_name", None) or _field(contact, "given_name", None) or contact_id or "unknown")
    parts: List[str] = [f"{name}."]
    known: List[str] = []
    first_seen = _utc(_field(contact, "first_seen_at", None))
    if first_seen is not None:
        known.append(f"known since {first_seen.date().isoformat()}")
    handles = _field(contact, "handles", None) or []
    rendered = [f"{h.get('gateway')}:{h.get('address')}" for h in handles
                if isinstance(h, Mapping) and h.get("gateway") and h.get("address")][:3]
    if rendered:
        known.append("reachable at " + ", ".join(rendered))
    if known:
        parts.append("Known: " + "; ".join(known) + ".")
    last = _utc(last_interaction_at)
    conversations = int(_field(contact, "interaction_count", 0) or 0)
    if last is None:
        parts.append("Never talked.")
    else:
        talk = f"Last talked {_ago(now - last)}"
        if conversations:
            talk += f" ({conversations} conversation{'s' if conversations != 1 else ''})"
        parts.append(talk + ".")
    inbound, outbound = counts.get("inbound"), counts.get("outbound")
    if inbound is not None or outbound is not None:
        parts.append(f"Exchanges: {int(inbound or 0)} in / {int(outbound or 0)} out.")
    open_items = counts.get("open")
    if open_items:
        parts.append(f"{int(open_items)} open item{'s' if int(open_items) != 1 else ''}.")
    notes = [" ".join(str(claim).split()) for claim in claims if str(claim or "").strip()][:MAX_CLAIMS]
    if notes:
        parts.append("Notes: " + " ".join(note if note.endswith(".") else note + "." for note in notes))
    text = " ".join(parts)
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS - 1].rstrip() + "…"
    return text


def render_template(contact: Any, *, claims: Iterable[Tuple[str, str]], counts: Mapping[str, Any],
                    last_interaction_at: Any, now: datetime) -> Tuple[str, List[str]]:
    """The template digest and its lineage: the text, and the ids of the claims whose lines it carries.

    ``claims`` are ``(claim id, line)`` pairs, newest first (``host.claims_for``); the first ``MAX_CLAIMS``
    with words are rendered and cited."""
    kept = [(str(ident), str(line)) for ident, line in claims if str(line or "").strip()][:MAX_CLAIMS]
    text = render_digest(contact, claims=[line for _, line in kept], counts=counts,
                         last_interaction_at=last_interaction_at, now=now)
    return text, [ident for ident, _ in kept]


def not_live(conn: Any, sources: Iterable[Any], *, contact_id: str) -> List[str]:
    from colonymind.memory import curated
    sources = list(sources or [])
    return curated.fenced_read(conn, lambda: _not_live(conn, sources, contact_id=contact_id),
                              empty=list(dict.fromkeys(str(ref) for ref in sources)))


def _not_live(conn: Any, sources: Iterable[Any], *, contact_id: str) -> List[str]:
    """The cited refs the digest of ``contact_id`` may no longer rest on, in one query by id over the source
    ledger: a claim that is gone (erased), retracted, or from a source whose projections were erased; a claim
    that is not that contact's own (its source is someone else's, or not person-scoped, or its attribution was
    corrected away: a claim id a moved source gives again under its new person names the same words, now
    theirs); and any ref that is not a claim (nothing to check it against). The ownership rule is the one both
    writers read claims by (``SourceClaimProjection._rows``). Superseded and expired claims cannot support
    the summary of a person's current facts."""
    refs = list(dict.fromkeys(str(ref) for ref in sources or []))
    claims = [ref for ref in refs if ref.startswith("claim:")]
    live: set = set()
    for start in range(0, len(claims), 500):
        chunk = claims[start:start + 500]
        live.update(row[0] for row in conn.execute(
            "SELECT c.id FROM source_claims c JOIN turn_sources s ON s.turn_id = c.turn_id "
            "WHERE c.id IN (" + ",".join("?" for _ in chunk) + ") AND c.retracted_by IS NULL "
            "AND c.superseded_by IS NULL "
            "AND (c.valid_to IS NULL OR julianday(c.valid_to) > julianday('now')) "
            "AND s.contact_id = ? AND s.scope = 'person' "
            "AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id = c.turn_id) "
            "AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id = c.turn_id) "
            "AND NOT EXISTS (SELECT 1 FROM learning_items i WHERE i.store='claim' AND i.item_id=c.id AND i.tier='cold')",
            [*chunk, str(contact_id)]))
    from colonymind.memory import curated
    live = {ident for ident in live if not any(curated.is_forgotten(conn, source)
            for source in curated._claim_sources(conn, ident))}
    return [ref for ref in refs if ref not in live]


def source_ledger() -> Any:
    """The instance's source ledger, or None where it has none (then no cited claim can be checked)."""
    from colonymind import get_state_dir
    from colonymind.turns import get_turn_idempotency_ledger
    state = Path(get_state_dir())
    return get_turn_idempotency_ledger(state) if (state / "turn-idempotency.db").exists() else None


def _stale(ledger: Any, cited: Mapping[str, List[str]]) -> Dict[str, List[str]]:
    from contextlib import closing
    with closing(ledger._connect()) as conn:
        return {contact_id: not_live(conn, sources, contact_id=contact_id) for contact_id, sources in cited.items()}


async def _withhold(store: Any, contact_id: str, sources: List[str]) -> str:
    withhold = getattr(store, "withhold_digest", None)
    if not callable(withhold):
        return "unchecked"
    try:
        if not await withhold(contact_id, sources=sources):
            # The digest changed since it was read: withheld already (another reader), or written again, in
            # which case the next read checks the new one.
            now = await store.get(contact_id)
            if str(_field(now, "digest", None) or "").strip():
                return "unchecked"
    except Exception as error:
        logger.warning("digest not withheld (%s)", type(error).__name__)
        return "unchecked"
    return "withheld"


async def checked_digests(contacts: Iterable[Any], *, store: Any, ledger: Any) -> Dict[str, Tuple[str, str]]:
    """``checked_digest`` for several contacts (a contact list), by contact id: the lineage of all of them is
    read over one ledger connection, and a contact without a digest, or whose digest cites nothing, reads
    none."""
    results: Dict[str, Tuple[str, str]] = {}
    cited: Dict[str, Tuple[str, List[str]]] = {}
    for contact in contacts:
        contact_id = str(_field(contact, "contact_id", ""))
        text = str(_field(contact, "digest", None) or "").strip()
        sources = [str(ref) for ref in _field(contact, "digest_sources", None) or []]
        if not text:
            results[contact_id] = ("", "none")
        elif not sources:
            results[contact_id] = (text, "shown")
        else:
            cited[contact_id] = (text, sources)
    if not cited:
        return results
    stale: Optional[Dict[str, List[str]]] = None
    if ledger is not None:
        try:
            from colonymind.memory import producers
            stale = await producers.read(_stale, ledger, {cid: refs for cid, (_, refs) in cited.items()})
        except Exception as error:
            logger.warning("digest lineage unreadable (%s)", type(error).__name__)
    for contact_id, (text, sources) in cited.items():
        if stale is None:
            results[contact_id] = ("", "unchecked")
        elif not stale.get(contact_id):
            results[contact_id] = (text, "shown")
        else:
            from colonymind.memory import turn_budget
            if turn_budget.current() is not None:
                turn_budget.defer_write(_withhold, store, contact_id, sources)
                results[contact_id] = ("", "unchecked")
            else:
                results[contact_id] = ("", await _withhold(store, contact_id, sources))
    return results


async def checked_digest(contact: Any, *, store: Any, ledger: Any) -> Tuple[str, str]:
    """The contact's digest as a reader may show it, and what the check found: ``(text, state)``.

    ``shown``: every cited claim stands (or it cites none). ``none``: there is no digest. ``withheld``: a cited
    claim is not live, so nothing is shown and the store drops the text and marks the digest for a rebuild
    (``withhold_digest``). ``unchecked``: the lineage could not be read, or the withhold not written: nothing
    is shown and, apart from a withhold that failed, nothing changed. Every reader of a person digest goes
    through here or ``checked_digests``: the "About this person" section, the owner's inspect, the host's
    contact routes (get, list, resolve, the timezone write, an introduction), the forget route and an
    attribution correction."""
    results = await checked_digests([contact], store=store, ledger=ledger)
    return results[str(_field(contact, "contact_id", ""))]


__all__ = ["MAX_CHARS", "MAX_CLAIMS", "WRITER_NIGHT", "WRITER_TEMPLATE", "checked_digest", "checked_digests",
           "not_live", "render_digest", "render_template", "source_ledger"]
