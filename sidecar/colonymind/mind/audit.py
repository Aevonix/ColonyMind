"""The audit log: ``log``, ``why`` and ``stats`` over the initiatives table (architecture 7.8).

Every intention row already holds the drive, concern, evidence, class,
decision, reason, Hermes reference, outcome, verification and verdict. This
module only renders them, with secrets redacted through ``P/redact``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from colonymind.redact import redact_sensitive_text

from colonymind.initiatives.models import StoredInitiative, OWNER_KINDS
from colonymind.util.temporal import now_utc

MAX_TEXT = 400
# The mind's reporting to the owner: never a drive's work, never a contact message. The last four
# arrived with the people milestone: a grant the owner gave over a ``never`` contact, a recipient the
# owner named that the store cannot resolve, and a name-only identity link or cadence match that needs
# the owner's word. A contradiction question (the memory milestone) is not a notice: it is an action. The one
# clarification of a question the owner's reply left unclear (``question_again``) is a notice, and so are an ask put
# in front of the owner again, the one plain question about their unclear answer and the one line saying what an
# answer did (``ask_show``, ``ask_clarify``, ``ask_ack``). So is the answer tripwire's warning that a tool read the
# instance key or named the answer routes (``answer_forgery_notice``): a drive turned off never cancels it. So is the
# line saying an ask was answered in an owner channel's app (``ask_answered``).
NOTICE_TYPES = ("ask_notice", "digest", "breaker_notice", "health_notice", "grant_refused", "recipient_unknown",
                "link_proposal", "cadence_confirm", "task_outcome", "question_again", "ask_in_full", "ask_show",
                "ask_clarify", "ask_ack", "answer_forgery_notice", "ask_answered")
ACTION_KINDS = ("task", "goal", "message")
# What a row that is not one of the agent's actions shows where the agent reads its own log (``split``).
NOTE_FIELDS = ("id", "created_at", "kind", "type", "title", "status")
# The owner's own words to a third party, sent as the owner gave them: a notice formed only from a request of
# the owner's that capture confirmed (``commitments.extract.request_confirmed``, ``drives.granted_message``).
DICTATED_TYPES = ("commitment_notice",)
# A reference to a sealed value as stored words carry it, and what stands for a value or a reference in a
# message to anyone but the owner that the mind composed itself. The same text as ``colonymind.secrets.detect``
# (REFERENCE, HIDDEN; a test keeps them equal), kept here so this predicate runs on its own.
SEALED_REFERENCE = re.compile(r"\[secret S-[A-Z2-7]{8}: [A-Za-z0-9 _.@+-]{1,60}\]")
WITHHELD_VALUE = "(a private value, withheld)"


def _clip(text: Any, limit: int = MAX_TEXT) -> str:
    value = "" if text is None else str(text)
    value = redact_sensitive_text(value)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _context(row: StoredInitiative) -> Dict[str, Any]:
    return row.context if isinstance(row.context, dict) else {}


def _when(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def stored_text(row: StoredInitiative) -> str:
    """The words a message row holds (its text, else its title)."""
    return str(_context(row).get("text") or row.description or "")


def to_owner(row: StoredInitiative, owner_id: str | None) -> bool:
    """Whether a message goes to the owner; without a known owner, never."""
    return bool(owner_id) and (row.entity_id or owner_id) == owner_id


def dictated(row: StoredInitiative) -> bool:
    """Whether a message is the owner's own words to someone else: a notice of their confirmed request, granted
    to its recipient or waiting for their word on who that recipient is (``ask_owner``)."""
    context = _context(row)
    return (row.kind == "message" and row.type in DICTATED_TYPES
            and (context.get("grant") == "owner" or context.get("ask_owner") is True))


def outgoing_text(row: StoredInitiative, *, owner_id: str | None = None) -> str:
    """The words a message row sends (``outbox.message_payload`` sends exactly these). To the owner, and the
    owner's own dictated words to someone else: the stored words as they are (the body resolves references at
    the send, ``api/routers/secrets.py`` deciding which). Anything else the mind composed, to anyone but the
    owner: every secret hidden and every reference withheld. Without ``owner_id`` the recipient counts as not
    the owner, so a caller that forgets it can only hide."""
    words = stored_text(row)
    if to_owner(row, owner_id) or dictated(row):
        return words
    return hidden_text(words)


def _detected_hidden(text: str) -> str:
    """``text`` with every value the capture detector finds, and every reference, replaced by ``WITHHELD_VALUE``.
    A copy of this module run on its own, without the package, has the redactor and the reference rule alone."""
    try:
        from colonymind.secrets.detect import hide_secrets
    except ImportError:
        return SEALED_REFERENCE.sub(WITHHELD_VALUE, text)
    return hide_secrets(text)


def hidden_text(text: str) -> str:
    """``text`` with every secret hidden: the capture detector's values and references first (a whole value, never
    a masked part of one), then what ``redact_sensitive_text`` hides, to its fixpoint: one pass may leave a part
    of a value (``notfor...act``) that the next hides, and any later rendering (which redacts again) must show
    these very characters."""
    text = _detected_hidden(str(text or ""))
    for _ in range(8):
        hidden = redact_sensitive_text(text)
        if hidden == text:
            return text
        text = hidden
    return text


def frozen_for_ask(row: StoredInitiative, *, owner_id: str | None) -> Optional[Dict[str, Any]]:
    """The context a message takes as it is asked about: its words fixed as the text it sends
    (``outgoing_text``: hidden for anyone but the owner unless they are the owner's dictated words; ``hidden``
    marks a change). None when nothing changes. What the owner approves and what leaves are then one string."""
    if row.kind != "message":
        return None
    context = _context(row)
    words = stored_text(row)
    shown = outgoing_text(row, owner_id=owner_id)
    if context.get("text") == shown:
        return None
    return {**context, "text": shown, **({"hidden": True} if shown != words else {})}


def asked_words(row: StoredInitiative) -> Optional[str]:
    """The exact words an open ask on a message sends once the owner says yes (the stored words, byte for byte:
    the ask fixed them as ``outgoing_text`` sends them, ``frozen_for_ask``); None for anything else. Every
    place the owner is asked (the ask notice, the digest, ``asks``, ``why``, the context packet) shows them: a
    yes approves what the owner saw, never a title standing in for the text."""
    if row.kind != "message" or row.status != "asked":
        return None
    return stored_text(row)


def entry(row: StoredInitiative) -> Dict[str, Any]:
    """One audit row as the API and the CLI show it."""
    if row is None or row.kind in OWNER_KINDS:
        raise ValueError("Unknown intention")
    context = _context(row)
    return {
        "id": row.id,
        "created_at": _when(row.created_at),
        "kind": row.kind,
        "type": row.type,
        "source_type": row.source_type,
        "source_id": row.source_id,
        "drive": row.drive,
        "cls": row.cls,
        "title": _clip(row.description, 160),
        "concern": _clip(context.get("concern"), 200),
        "evidence": [_clip(item, 160) for item in (context.get("evidence") or [])][:8],
        "recipient": row.entity_id,
        "decision": row.decision,
        "decision_reason": _clip(row.decision_reason, 200),
        "status": row.status,
        "ask_code": row.ask_code,
        "hermes_kind": row.hermes_kind,
        "hermes_ref": row.hermes_ref,
        "outcome": row.outcome,
        "verified": row.verified,
        "verdict": row.verdict,
        "summary": _clip(row.result, 300),
        "cost_tokens": row.cost_tokens,
        "due_at": _when(row.due_at),
        "expires_at": _when(row.expires_at),
        "message": asked_words(row),
    }


def is_action(entry: Dict[str, Any]) -> bool:
    """Whether an audit row is one of the agent's own actions: a task, goal or message it decided to act on
    or ask about. Internal notes (the nightly consolidation, a deliberation that formed nothing, owner
    switches) and notices (digest, ask notice, breaker and health notices) are not. The self-narrative's
    evidence and the benchmark's record of what the agent did both use this one predicate."""
    return (entry.get("kind") in ACTION_KINDS and entry.get("decision") in {"act", "ask"}
            and entry.get("type") not in NOTICE_TYPES)


def split(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(actions, notes)`` of rendered rows, in their order: the agent's own actions (``is_action``) and
    everything else the log holds (an internal note such as the nightly consolidation, a notice, a decision
    not to act), each note cut to what it is (``NOTE_FIELDS``): no drive or decision that reads as a reason."""
    actions = [item for item in entries if is_action(item)]
    notes = [{key: item.get(key) for key in NOTE_FIELDS} for item in entries if not is_action(item)]
    return actions, notes


def log(store: Any, *, limit: int = 20, since: Optional[datetime] = None,
        status: Optional[List[str]] = None, kind: Optional[List[str]] = None,
        recipient: Optional[str] = None) -> List[Dict[str, Any]]:
    """Newest first; every filter, ``recipient`` included, selects before ``limit`` applies."""
    extra = {"recipient": recipient} if recipient else {}
    return [entry(row) for row in store.intentions(status=status, kind=kind, since=since, limit=limit, **extra)
            if row.kind not in {"owner_goal", "owner_subgoal", "owner_goal_progress"}]


def why(store: Any, intention_id: str) -> Optional[Dict[str, Any]]:
    """One row rendered as a sentence, with its fields alongside."""
    row = store.get(intention_id)
    if row is None or not row.kind or row.kind in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        return None
    value = entry(row)
    history = [{"at": item.timestamp.isoformat(), "action": item.action, "details": item.details}
               for item in reversed(store.get_history(intention_id, limit=50))]
    parts = [f"{value['drive'] or 'unknown'} drive"]
    if value["concern"]:
        parts.append(f"concern: {value['concern']}")
    if value["evidence"]:
        parts.append("evidence: " + "; ".join(value["evidence"]))
    parts.append(f"class {value['cls']}, decided {value['decision']} ({value['decision_reason']})")
    if value["hermes_ref"]:
        parts.append(f"Hermes {value['hermes_kind']} {value['hermes_ref']}")
    elif value["ask_code"] and value["status"] == "asked":
        parts.append(f"waiting for the owner (code {value['ask_code']})")
    if value["message"] is not None:
        parts.append(f"would send to {value['recipient']}: \"{value['message']}\"")
    parts.append(f"outcome {value['outcome'] or value['status']}")
    parts.append(f"verified: {value['verified'] or 'none'}")
    if value["verdict"]:
        parts.append(f"owner verdict {value['verdict']}")
    value["action"] = is_action(value)
    value["sentence"] = f"{value['title']}: " + ", ".join(parts) + "." + (
        "" if value["action"] else " Not an action: an internal note or notice.")
    value["history"] = history
    value["text"] = value["sentence"]
    return value


def stats(store: Any, *, now: Optional[datetime] = None, days: int = 7,
          ledger_turns: Optional[int] = None) -> Dict[str, Any]:
    """The in-vivo panel (evals section 8), descriptive only."""
    now = now or now_utc()
    since = now - timedelta(days=days)
    rows = store.intentions(since=since, limit=5000)
    acted = [row for row in rows if row.decision == "act" and row.type not in NOTICE_TYPES
             and row.kind in {"task", "message", "goal"}]
    asks = [row for row in rows if row.decision == "ask"]
    notices = [row for row in rows if row.kind == "message" and row.type in NOTICE_TYPES]
    done = [row for row in rows if row.outcome == "done"]
    failed = [row for row in rows if row.outcome == "failed"]
    verdicts = [row for row in rows if row.verdict]
    accepted = [row for row in verdicts if row.verdict in {"actioned", "useful"}]
    verified = [row for row in done if row.verified in {"owner", "check"}]
    commitments = [row for row in rows if row.type == "commitment_overdue"]
    off_uses = [row for row in rows if row.type == "off_switch"]
    blocked = [row for row in rows if row.decision == "defer"]
    by_class: Dict[str, int] = {}
    for row in failed:
        by_class[row.cls or "unknown"] = by_class.get(row.cls or "unknown", 0) + 1
    tokens = sum(int(row.cost_tokens or 0) for row in rows)
    span = max(1.0, (now - since).total_seconds() / 86400.0)
    return {
        "window_days": days,
        "intentions": len(rows),
        "acted": len(acted),
        "asks": len(asks),
        "asks_per_day": round(len(asks) / span, 2),
        "notices_per_day": round(len(notices) / span, 2),
        "initiative_acceptance": (round(len(accepted) / len(verdicts), 3) if verdicts else None),
        "commitment_fulfilment": (round(sum(1 for row in commitments if row.outcome == "done") / len(commitments), 3)
                                  if commitments else None),
        "corrections_per_100_turns": (round(100.0 * sum(1 for row in verdicts if row.verdict == "wrong")
                                            / ledger_turns, 2) if ledger_turns else None),
        "blocked_work": len(blocked),
        "off_switch_uses": len(off_uses),
        "verified_share": (round(len(verified) / len(done), 3) if done else None),
        "lesson_use_rate": None,
        "failures_per_class_per_week": {cls: round(count / span * 7, 2) for cls, count in by_class.items()},
        "mind_tokens": tokens,
    }


def _when_text(item: Dict[str, Any]) -> str:
    return (item.get("created_at") or "")[:16].replace("T", " ")


def _row_line(item: Dict[str, Any]) -> str:
    """One row with its whole id: an id is cited as the log shows it, and a prefix is no id."""
    tail = item.get("outcome") or item.get("status")
    code = f" [{item['ask_code']}]" if item.get("ask_code") and item.get("status") == "asked" else ""
    return (f"{_when_text(item)}  {item['id']}  {item['kind']:<7} {item['decision']:<5} {tail:<10} "
            f"{item['drive']}/{item['type']}: {item['title']}{code}")


def render_log(entries: List[Dict[str, Any]]) -> str:
    lines = [_row_line(item) for item in entries]
    return "\n".join(lines) if lines else "(no intentions yet)"


def render_split(actions: List[Dict[str, Any]], notes: List[Dict[str, Any]]) -> str:
    """The agent's own reading of its log: its actions, then apart the rows that are not one."""
    lines = (["Your actions (cite these ids):", *(_row_line(item) for item in actions)] if actions
             else ["Your actions: none in this window."])
    if notes:
        lines += ["", "Not actions (notes, notices, decisions not to act; never report one as something you did):",
                  *(f"{_when_text(item)}  {item['id']}  {item['kind']}/{item['type']}: {item['title']}"
                    for item in notes)]
    return "\n".join(lines)


def render_stats(value: Dict[str, Any]) -> str:
    return "\n".join(f"{key}: {json.dumps(item) if isinstance(item, (dict, list)) else item}"
                     for key, item in value.items())


__all__ = ["ACTION_KINDS", "DICTATED_TYPES", "NOTE_FIELDS", "NOTICE_TYPES", "SEALED_REFERENCE", "WITHHELD_VALUE",
           "dictated", "entry", "is_action", "log", "outgoing_text", "render_log", "render_split", "render_stats",
           "split", "stats", "stored_text", "to_owner", "why"]
