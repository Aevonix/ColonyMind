"""Outcomes: what later happened to a decision, read from the instance's own stores, and the owner's own label
(docs/DECISIONS.md, Calibration and learning).

**Resolvers** are read-only joins over what the mind already records, run with the nightly pass; none is a hook in a
caller. Each labels a decision right, wrong or nothing from what the stores show after it, and never from the answer
any tier gave: the applied answer is what is graded, never the grade. The first set:

| Point | Right | Wrong | Source |
|---|---|---|---|
| ``owner_send`` | a message the gate sent, then engaged, welcomed or asked more about | sent and answered with a stop in the owner's own words (explicit) or a "not useful" (behavioural) | behavioural unless noted; interpreted where a model read the reaction |
| ``owner_instruction`` | a stop still in force a day on | a stop the owner lifted before the next night's check (within a day) | behavioural |
| ``opt_out`` | | a model-read opt-out the owner set back from ``never`` | explicit |
| ``question_answer`` | | an applied answer whose statements the owner disputes again (a later question on one of its claims) | behavioural |
| ``report_substance`` | a report read as a finding, sent, then engaged or welcomed | read as a finding, sent, then found not useful | behavioural; interpreted where a model read the reaction |
| ``owner_verdict`` | | a withdrawn verdict the owner restates (a later verdict in the same session within a day that stood) | behavioural |
| ``ask_answer`` | a model's yes the owner confirmed when asked back and did not undo (explicit), or whose approval stood a day (behavioural) | a model's yes the owner answered no when asked back, or whose approval they undid (confirmed or not, "I didn't say yes" included); a model's no they took back (undo, "I didn't say no") | explicit unless noted; interpreted where a model read the owner's answer to the confirm-back |

A reaction counts as the owner's word only when the phrase tables read it from their own words. One a model read
(a decision model's reading of the reply, an appraisal's dismissal, or a stop whose speaker the
``owner_instruction`` point had to settle) is ``interpreted``: kept and shown as its own teacher source, never a
label and never a count for the tighten rule.

What the stores do not record labels nothing: a message the gate held and the owner acted on from the digest, a report
read as empty (it is never shown), a contact who writes again wanting replies, the owner saying they never said stop,
a model's no to an ask that stands (the ask is closed; the owner's label grades it). A model's yes or no to an ask is
graded from the ask's own history (``model_yes``, ``model_no``), which records the decision that asked the owner back,
approved or declined; since a later undo can overturn it, each pass grades it again and replaces its own earlier grade
(``REVISED``). The windows are my choices. Only an explicit wrong counts toward the tighten rule (two at a point
within seven days tighten its model tiers at once, ``fit.tighten``); behavioural labels only ever tighten a refit.

**The owner's label** (``label``, ``colonymind instinct label <decision> <label>``) is explicit: the right label for
that decision, checked against the point's labels; it overrides whatever a resolver recorded for the decision, and
their later label replaces their earlier one; the tighten rule runs at once.

**Erasure.** A decision whose references name a turn an erasure touched (erased whole, or left with fewer messages
when an erased message's copy was removed from it) is forgotten (``forget_erased``): the row and its outcomes go,
what it counted stays. So is one whose grade rests on a later turn so touched (``GRADED_BY``: an ask's answer to its
question back, or an undo; ``RESTS_ON``: the owner's restatement of a withdrawn verdict, or their reaction to a
message the gate sent or a finding reported).

**Re-rendering** (``rerender``, ``replay_share``). A row keeps references and a keyed hash of the state its point
read, never the words. Where its point has a loader (``LOADERS``), the references locate the fields again in the
stores, the point's template renders them, and the hash under the instance's secret must match: then the row
re-renders. It does not when a source was erased or changed, or a field depends on when it was read (the question
reader's earlier turns, read after a later turn arrived; an appraisal's earlier quotes, read beside a turn whose
person had a view already). Nor does it when a turn it read, as its own or as context in its session, is one every
projection of the ledger leaves out (``Sources.withheld``: an answer that drew on a source whose speaker was
corrected, or a turn an erasure kept a copy out of): the loaders read turns through ``_Reading``, which never hands
them one. A loader reads what its caller read, as the caller reads it: the owner's or a contact's
turn (``owner_instruction``, ``opt_out``, ``owner_verdict``, and the pre-screens ``opinion_none`` and
``appraisal_empty`` over the turn's messages), the mind's rows (``outreach_reply``, ``report_substance``,
``question_answer``), and for ``no_reminders`` the item the turn's capture made whose description the decision's
``about`` reference names by its keyed hash, beside the turn's words. A point without one has a reason code
(``NO_LOADER``, shown by ``colonymind report instinct``):

- ``redacted``: ``owner_send`` keeps the gate's house facts and its message text only redacted, and the facts are the
  gate's view of that moment;
- ``unreferenced``: a field is not among the references: ``interest_settled``'s topic (the references name only the
  turn, and the person's open interests change as they are settled), ``mind_wish_scope``'s words (the references name
  only the capability);
- ``unlocated``: the references do not locate every field as it was read: ``claim_relation`` read the two statements
  as the night's grouping and timezone showed them (a restated claim moves to its newest witness, and the topic names
  the first claim of its group); ``ask_answer`` read the owner's words on the answer route, which may run before the
  turn is captured and then logs no turn, and its ask's sentence names a contact as the contacts store named them;
- ``external``: a point a gate outside the sidecar asks (``cron_*``, ``inbound_triage``, ``tool_screen`` and every
  operator point): its state's words live only in the caller's own store, never in this instance.

A loader a caller would need to log more for is not written here. The share that re-render is in ``colonymind report
instinct``. Learning from rows (training by reference) can use only the rows that re-render.

**Training rows by reference** (``replay_rows``). Per decision that re-renders ``matched`` and has an outcome, one
row in memory: the state text, its keyed hash, the label and teacher of its strongest outcome (``log.precedence``)
and the narrowest audience its sources show. The teacher follows the outcome's source (``teacher``): the owner's own
label (``label``) is ``owner_label``; a resolver's grade from the owner's own words or acts (``explicit``) is
``verified_outcome``; a behavioural one is ``behavioural_proxy``, which only ever pushes away from the label the owner's
behaviour showed wrong (``push_away``; a behavioural "right" teaches nothing, since a proxy is never a positive); a
synthetic one is ``synthetic``; an interpreted one is no label. Where an outcome names only the label that was wrong
(no right label known), a yes/no row takes the other label, as the refit does, and a choice row pushes away from the
wrong one. The audience (``audience``) is ``owner`` where any source is the owner's or the mind's own record,
``contact`` where the narrowest is a contact the owner filed, ``guest`` where every source is a sender the agent
remembered on its own; a row whose sources show none (the owner or a person the contacts store cannot place) is left
out. The label comes from the strongest outcome whose grade may still be read (``sound``): one resting on a later
turn now erased or withheld from projections (``RESTS_ON``, ``GRADED_BY``) is no label, whether or not the nightly
pass has forgotten its decision yet. The state text is never written: no database, file or log line holds it.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from colonymind.util.temporal import parse_utc as _utc

logger = logging.getLogger(__name__)

#: ``interpreted``: what a model (or a reader's fallback) made of the owner's reaction, not their plain word: kept
#: and shown as its own teacher source, never a label for the fit and never a count for the tighten rule.
SOURCES = ("explicit", "behavioural", "synthetic", "interpreted")
VERDICTS = ("right", "wrong")
#: How a reaction's class was read from the owner's own words: by the phrase tables (``by`` ``phrase``), on a reply
#: linked to the message by its topic or its position. Anything else (a decision model's reading, recorded as
#: ``read``; an appraisal's dismissal, ``by`` ``appraisal``) is an interpretation.
OWN_WORDS = frozenset({"phrase", "topic", "position"})
#: The classes the ``owner_instruction`` point settles when the reader cannot tell whose instruction a cue is.
INSTRUCTED = frozenset({"stop", "pause_today"})
#: How far back the nightly pass looks for decisions to resolve (my choice: two weeks of outcomes).
RESOLVE_WINDOW = timedelta(days=14)
#: A stop lifted before the first nightly check after it (at most a day: the store keeps no time of a resume, my
#: choice) is wrong; one still in force after a day is right.
RESUME_WINDOW = timedelta(days=1)
#: A withdrawn verdict restated in the same session within this long (my choice).
RESTATE_WINDOW = timedelta(days=1)
#: A model's yes that approved an ask is right once the approval stood this long with no undo (one day, a chosen value).
STANDS = timedelta(hours=24)
ENGAGED = frozenset({"engaged", "positive", "welcome"})
TURN_DB, INITIATIVES_DB, MIND_DB = "turn-idempotency.db", "initiatives.db", "mind.db"
COMMITMENTS_DB = "colonymind-commitments.db"
#: How much of a turn capture reads, and the longest message an appraisal takes as evidence (the callers' own
#: limits, in ``commitments.extract`` and ``self_model.appraisals``; a change there reads ``changed`` here).
CAPTURE_CHARS, APPRAISAL_MESSAGE_CHARS = 6000, 12000


def turn_ref(key: str, value: Any) -> Optional[str]:
    """The turn id a reference names: a ``turn`` reference, or a value ``turn:<id>``."""
    text = str(value or "")
    if key == "turn" and text and not text.startswith("h:"):
        return text
    if text.startswith("turn:") and len(text) > 5:
        return text[5:]
    return None


class Sources:
    """Read-only views of the instance's stores. Each is opened only to read (``query_only``); a store that is
    missing or unreadable reads as empty."""

    def __init__(self, state_dir: Any, *, contacts_db: Any = None, owner_id: Optional[str] = None) -> None:
        self.state_dir = Path(state_dir)
        self.contacts_db = Path(contacts_db) if contacts_db else self._contacts_path()
        self.owner_id = owner_id if owner_id is not None else (os.environ.get("COLONYMIND_OWNER_CONTACT_ID") or None)

    def _contacts_path(self) -> Path:
        configured = os.environ.get("COLONYMIND_CONTACTS_DB")
        if configured:
            return Path(configured)
        for name in ("contacts.db", "colonymind-contacts.db"):
            if (self.state_dir / name).is_file():
                return self.state_dir / name
        return self.state_dir / "contacts.db"

    def _query(self, path: Path, sql: str, args: Iterable[Any] = ()) -> List[sqlite3.Row]:
        if not Path(path).is_file():
            return []
        try:
            with closing(sqlite3.connect(str(path), timeout=5)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only = ON")
                return conn.execute(sql, list(args)).fetchall()
        except sqlite3.Error as error:
            logger.info("instinct outcomes: %s unreadable (%s)", Path(path).name, type(error).__name__)
            return []

    @contextmanager
    def _reading(self, path: Path):
        """``_query`` for several reads of one store on one connection; each read that fails reads as empty, as
        there."""
        conn, failure = None, None
        if Path(path).is_file():
            try:
                conn = sqlite3.connect(str(path), timeout=5)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only = ON")
            except sqlite3.Error as error:
                if conn is not None:
                    conn.close()
                conn, failure = None, error

        def query(sql: str, args: Iterable[Any] = ()) -> List[sqlite3.Row]:
            error = failure
            if conn is not None:
                try:
                    return conn.execute(sql, list(args)).fetchall()
                except sqlite3.Error as raised:
                    error = raised
            if error is not None:
                logger.info("instinct outcomes: %s unreadable (%s)", Path(path).name, type(error).__name__)
            return []

        try:
            yield query
        finally:
            if conn is not None:
                conn.close()

    # -- turns -----------------------------------------------------------------------------------------------

    @staticmethod
    def _messages(raw: Any) -> List[tuple]:
        from colonymind.turns.audio import source_text
        try:
            messages = json.loads(raw or "[]")
        except ValueError:
            return []
        out = []
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            content = source_text(content) if isinstance(content, list) else content
            if isinstance(content, str):
                out.append((str(message.get("role") or ""), content))
        return out

    def _turn(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {"turn_id": str(row["turn_id"]), "contact_id": str(row["contact_id"]),
                "session_id": str(row["session_id"]), "at": _utc(row["at"]),
                "messages": self._messages(row["messages_json"])}

    def turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """A recorded turn: its contact, session, time and messages (role, text); None when missing or erased."""
        if not turn_id or turn_id in self.erased([turn_id]):
            return None
        rows = self._query(self.state_dir / TURN_DB,
                           "SELECT turn_id, contact_id, session_id, messages_json, coalesce(occurred_at, ingested_at) "
                           "AS at FROM turn_sources WHERE turn_id=?", (turn_id,))
        return self._turn(rows[0]) if rows else None

    def session(self, contact_id: str, session_id: str) -> List[Dict[str, Any]]:
        """A session's recorded turns, oldest first."""
        rows = self._query(self.state_dir / TURN_DB,
                           "SELECT turn_id, contact_id, session_id, messages_json, coalesce(occurred_at, ingested_at) "
                           "AS at FROM turn_sources WHERE contact_id=? AND session_id=? ORDER BY at, rowid",
                           (contact_id, session_id))
        return [self._turn(row) for row in rows]

    def erased(self, turn_ids: Iterable[str]) -> Set[str]:
        """Which of these turns an erasure touched: erased whole, or left with fewer messages (an erased message's
        copy removed from it, which the ledger records under the surviving turn's id in
        ``source_erasure_revisions``). A decision may have read the message that went, so either forgets it."""
        wanted = [str(ident) for ident in dict.fromkeys(turn_ids) if ident]
        out: Set[str] = set()
        if wanted:
            with self._reading(self.state_dir / TURN_DB) as query:
                # All the ids in one parameter and the matches in one result row, as ``withheld`` reads.
                for table, column in (("source_erasures", "turn_id"), ("source_erasure_revisions", "source_turn_id")):
                    rows = query(f"SELECT json_group_array({column}) AS turn_ids FROM {table} "
                                 f"WHERE {column} IN (SELECT value FROM json_each(?))", (json.dumps(wanted),))
                    out.update(str(ident) for row in rows for ident in json.loads(row["turn_ids"]))
        path = self.state_dir / TURN_DB
        if path.is_file():
            from colonymind.memory import curated
            with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                reader = curated.reader_of(conn)
                return curated.fenced_read(reader, lambda: out | curated.forgotten_among(reader, wanted, conn=conn),
                                           empty=set(wanted))
        return out

    def withheld(self, turn_ids: Iterable[str]) -> Set[str]:
        """Which of these turns every projection of the ledger leaves out although their words are kept: withheld
        from projections (``source_projection_erasures``: an erased message's copy was kept out of it, or it drew on
        a source whose speaker was corrected) or with its speaker attribution withdrawn
        (``source_attribution_invalidations``). Nothing learned may read them either (``_Reading``)."""
        wanted = [str(ident) for ident in dict.fromkeys(turn_ids) if ident]
        out: Set[str] = set()
        if not wanted:
            return out
        with self._reading(self.state_dir / TURN_DB) as query:
            for table, column in (("source_projection_erasures", "turn_id"),
                                  ("source_attribution_invalidations", "source_id")):
                # All the ids in one parameter, one index probe each (a turn can have hundreds of rows), and the
                # ids that have a row in one result row: each statement step and each row fetched hands the
                # interpreter lock over and back.
                rows = query(f"SELECT json_group_array(w.value) AS turn_ids FROM json_each(?) AS w "
                             f"WHERE EXISTS (SELECT 1 FROM {table} WHERE {column}=w.value)", (json.dumps(wanted),))
                out.update(str(ident) for row in rows for ident in json.loads(row["turn_ids"]))
        return out

    def raw_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """A recorded turn as its callers read it: its contact, session and scope, and its messages as stored (JSON
        objects). None when missing, erased (``erased``), withheld from projections (an erased message's copy kept
        out of it), or its speaker attribution withdrawn."""
        if not turn_id or turn_id in self.erased([turn_id]):
            return None
        rows = self._query(self.state_dir / TURN_DB, "SELECT turn_id, contact_id, session_id, scope, messages_json "
                           "FROM turn_sources WHERE turn_id=?", (turn_id,))
        if not rows or self.withheld([turn_id]):
            return None
        try:
            messages = json.loads(rows[0]["messages_json"] or "[]")
        except ValueError:
            return None
        return {"turn_id": str(rows[0]["turn_id"]), "contact_id": str(rows[0]["contact_id"]),
                "session_id": str(rows[0]["session_id"]), "scope": str(rows[0]["scope"] or ""),
                "messages": [message for message in messages if isinstance(message, dict)]
                if isinstance(messages, list) else []}

    def commitments_from(self, turn_id: str, *, person: str) -> List[Dict[str, Any]]:
        """The items a turn's capture made for a person (``metadata.source_turn``): id and description."""
        rows = self._query(self.state_dir / COMMITMENTS_DB, "SELECT id, description FROM commitments WHERE "
                           "person_id=? AND json_extract(metadata, '$.source_turn')=? ORDER BY made_at, rowid",
                           (person, turn_id))
        return [{"id": str(row["id"]), "description": str(row["description"] or "")} for row in rows]

    def person(self, contact_id: str) -> Optional[str]:
        """Who a speaker is to the owner: ``owner``; ``contact`` (a person the contacts store files); ``guest`` (a
        sender the agent remembered on its own that the owner has not filed). None when the owner is not known, or
        the store has no such person (or deleted them)."""
        if not contact_id or not self.owner_id:
            return None
        if contact_id == self.owner_id:
            return "owner"
        rows = self._query(self.contacts_db, "SELECT import_source, trust_tier FROM contacts WHERE contact_id=? AND "
                           "deleted_at IS NULL", (contact_id,))
        if not rows:
            return None
        remembered = str(rows[0]["import_source"] or "").startswith("auto:") and rows[0]["trust_tier"] == "unknown"
        return "guest" if remembered else "contact"

    # -- the mind's rows ---------------------------------------------------------------------------------------

    def intention(self, ident: Any) -> Any:
        """An intention (the mind's row: a message, a task, a question) by id, or None."""
        from colonymind.initiatives.models import StoredInitiative
        if not ident:
            return None
        rows = self._query(self.state_dir / INITIATIVES_DB, "SELECT * FROM initiatives WHERE id=?", (str(ident),))
        return StoredInitiative.from_row(dict(rows[0])) if rows else None

    def questions(self) -> List[Any]:
        """The mind's questions to the owner (contradictions), oldest first."""
        from colonymind.initiatives.models import StoredInitiative
        from colonymind.mind.consolidate import QUESTION_KEY
        from colonymind.mind.outbox import QUESTION_TYPES
        marks = ",".join("?" * len(QUESTION_TYPES))
        rows = self._query(self.state_dir / INITIATIVES_DB,
                           f"SELECT * FROM initiatives WHERE type IN ({marks}) AND dedup_key LIKE ? ORDER BY created_at",
                           (*sorted(QUESTION_TYPES), f"{QUESTION_KEY}%"))
        return [StoredInitiative.from_row(dict(row)) for row in rows]

    def history(self, ident: Any) -> List[tuple]:
        """An intention's history, oldest first: ``(action, at, details)``."""
        rows = self._query(self.state_dir / INITIATIVES_DB, "SELECT action, timestamp, details FROM "
                           "assignment_history WHERE initiative_id=? ORDER BY timestamp, id", (str(ident or ""),))
        out = []
        for row in rows:
            try:
                details = json.loads(row["details"] or "{}")
            except ValueError:
                details = {}
            out.append((str(row["action"]), _utc(row["timestamp"]), details if isinstance(details, dict) else {}))
        return out

    def gate_verdicts(self, decision: str) -> List[Dict[str, Any]]:
        """The owner gate's verdicts that carried this decision, oldest first (ids and codes only)."""
        rows = self._query(self.state_dir / MIND_DB,
                           "SELECT id, at, ref, source, kind, decision, stage, digested_at FROM owner_gate WHERE "
                           "json_extract(scores_json, '$.instinct') = ? ORDER BY at, rowid", (decision,))
        return [dict(row) for row in rows]

    def pause(self) -> Optional[Dict[str, Any]]:
        """The owner's pause on outreach as the mind holds it now, or None."""
        from colonymind.mind.outreach import PAUSE_KEY
        rows = self._query(self.state_dir / MIND_DB, "SELECT level, text, causes_json, updated_at FROM mind_state "
                           "WHERE key=?", (PAUSE_KEY,))
        if not rows:
            return None
        try:
            causes = [str(cause) for cause in json.loads(rows[0]["causes_json"] or "[]")]
        except ValueError:
            causes = []
        return {"level": float(rows[0]["level"] or 0.0), "text": rows[0]["text"], "causes": causes}

    def contact_audit(self, contact_id: str) -> List[Dict[str, Any]]:
        """A contact's audit rows, oldest first, with their detail parsed."""
        rows = self._query(self.contacts_db, "SELECT id, action, detail, performed_by, created_at FROM contact_audit "
                           "WHERE contact_id=? ORDER BY created_at, rowid", (contact_id,))
        out = []
        for row in rows:
            try:
                detail = json.loads(row["detail"] or "{}")
            except ValueError:
                detail = {}
            out.append({"id": row["id"], "action": row["action"], "detail": detail if isinstance(detail, dict) else {},
                        "by": row["performed_by"], "at": _utc(row["created_at"])})
        return out


@dataclass
class Context:
    store: Any
    sources: Sources
    now: datetime
    _instructed: Optional[Set[str]] = field(default=None, repr=False)

    def instructed(self) -> Set[str]:
        """The turns whose instruction the ``owner_instruction`` point was asked to settle: the reader could not tell
        whose words a cue was, so a model's reading (or the fallback) decided it."""
        if self._instructed is None:
            rows = self.store.rows(point="owner_instruction") if self.store is not None else []
            self._instructed = {str((row.get("refs") or {}).get("turn")) for row in rows
                                if (row.get("refs") or {}).get("turn")}
        return self._instructed


def _outcome(row: Mapping[str, Any], context: Context, resolver: str, verdict: str, *, source: str,
             judged: Optional[str], label: Optional[str] = None, evidence: Optional[Mapping[str, Any]] = None
             ) -> Dict[str, Any]:
    return {"decision": row["id"], "point": row["point"], "resolver": resolver, "source": source, "verdict": verdict,
            "judged": judged, "label": label, "at": context.now.isoformat(), "evidence": dict(evidence or {})}


def _reaction(intention: Any) -> Dict[str, Any]:
    metadata = intention.result_metadata if isinstance(getattr(intention, "result_metadata", None), dict) else {}
    value = metadata.get("reaction")
    return value if isinstance(value, dict) else {}


def _interpreted(reaction: Mapping[str, Any], context: Context) -> bool:
    """Whether a reaction's class is a reading of what the owner meant rather than their own words: a decision
    model read it, it came from an appraisal, or it is a stop or a pause whose speaker the ``owner_instruction``
    point had to settle."""
    if reaction.get("read") or str(reaction.get("by") or "phrase") not in OWN_WORDS:
        return True
    return reaction.get("class") in INSTRUCTED and str(reaction.get("turn") or "") in context.instructed()


def _reacted(reaction: Mapping[str, Any]) -> Dict[str, str]:
    """The turn a reaction was read from, for a grade's evidence (``RESTS_ON``): the grade rests on it."""
    turn = str(reaction.get("turn") or "")
    return {"reacted": turn} if turn else {}


def _applied(row: Mapping[str, Any], *labels: str) -> bool:
    return bool(row.get("applied")) and row.get("label") in labels


# -- the resolvers ---------------------------------------------------------------------------------------------------

def owner_send(row: Mapping[str, Any], context: Context) -> Optional[Dict[str, Any]]:
    """A message the gate sent, as the owner took it (the reaction the mind recorded on it)."""
    if not _applied(row, "send_now"):
        return None
    verdicts = [verdict for verdict in context.sources.gate_verdicts(row["id"]) if verdict["decision"] == "send"]
    if not verdicts or verdicts[-1].get("source") != "mind" or not verdicts[-1].get("ref"):
        return None                       # an outside caller's alert: nothing here records what the owner did
    message = context.sources.intention(verdicts[-1]["ref"])
    reaction = _reaction(message) if message is not None else {}
    kind = reaction.get("class")
    if kind not in {"stop", "negative", *ENGAGED}:
        return None
    read = _interpreted(reaction, context)
    evidence = {"intention": str(verdicts[-1]["ref"]), "reaction": kind, "read": "model" if read else "own words",
                **_reacted(reaction)}
    if kind == "stop":
        return _outcome(row, context, "owner_send", "wrong", source="interpreted" if read else "explicit",
                        judged="send_now", evidence=evidence)
    if kind == "negative":
        return _outcome(row, context, "owner_send", "wrong", source="interpreted" if read else "behavioural",
                        judged="send_now", evidence=evidence)
    return _outcome(row, context, "owner_send", "right", source="interpreted" if read else "behavioural",
                    judged="send_now", label="send_now", evidence=evidence)


def owner_instruction(row: Mapping[str, Any], context: Context) -> Optional[Dict[str, Any]]:
    """A stop read as the owner's own instruction: lifted before the next check (wrong), or still standing a day
    on (right). The pause the mind holds names the turn that set it."""
    if not _applied(row, "stop"):
        return None
    turn, at = row["refs"].get("turn"), _utc(row["at"])
    if not turn or at is None:
        return None
    pause = context.sources.pause()
    standing = pause is not None and pause["level"] > 0 and f"turn:{turn}" in pause["causes"]
    lifted = pause is None or pause["level"] <= 0
    evidence = {"turn": str(turn)}
    if context.now - at <= RESUME_WINDOW:
        if lifted:
            return _outcome(row, context, "owner_instruction", "wrong", source="behavioural", judged="stop",
                            evidence={**evidence, "pause": "lifted"})
        return None
    if standing:
        return _outcome(row, context, "owner_instruction", "right", source="behavioural", judged="stop", label="stop",
                        evidence={**evidence, "pause": "standing"})
    return None


def opt_out(row: Mapping[str, Any], context: Context) -> Optional[Dict[str, Any]]:
    """A contact's opt-out the model read, which the owner then set back (only the owner raises a permission)."""
    if not _applied(row, "yes"):
        return None
    contact, source = row["refs"].get("contact"), row["refs"].get("source")
    at = _utc(row["at"])
    if not contact or not source or at is None:
        return None
    audit = context.sources.contact_audit(str(contact))
    lowered = [entry for entry in audit if entry["action"] == "opt_out"
               and entry["detail"].get("source_ref") == source and entry["at"] is not None
               and entry["at"] >= at - timedelta(minutes=5)]
    if not lowered:
        return None
    since = lowered[0]["at"]
    for entry in audit:
        if (entry["action"] == "may_contact_set" and entry["at"] is not None and entry["at"] >= since
                and entry["detail"].get("from") == "never" and entry["detail"].get("to") in {"ask", "auto"}):
            return _outcome(row, context, "opt_out", "wrong", source="explicit", judged="yes", label="no",
                            evidence={"contact": str(contact), "audit": str(entry["id"])})
    return None


def question_answer(row: Mapping[str, Any], context: Context) -> Optional[Dict[str, Any]]:
    """An applied answer whose statements come back as a question later: the owner disputes them again."""
    answer = row.get("label")
    if not _applied(row, "first", "second", "both", "neither"):
        return None
    question = context.sources.intention(row["refs"].get("question"))
    if question is None:
        return None
    metadata = question.result_metadata if isinstance(question.result_metadata, dict) else {}
    state = metadata.get("question") if isinstance(metadata.get("question"), dict) else {}
    if state.get("state") != "resolved" or state.get("answer") != answer:
        return None                       # the answer never changed the memory
    context_ = question.context if isinstance(question.context, dict) else {}
    asked = context_.get("question") if isinstance(context_.get("question"), dict) else {}
    claims = {str(claim) for claim in asked.get("claims") or []}
    decided = _utc(row["at"])
    if not claims or decided is None:
        return None
    for later in context.sources.questions():
        created = _utc(later.created_at)
        other = later.context.get("question") if isinstance(later.context, dict) else None
        if later.id == question.id or created is None or created <= decided or not isinstance(other, dict):
            continue
        if claims & {str(claim) for claim in other.get("claims") or []}:
            return _outcome(row, context, "question_answer", "wrong", source="behavioural", judged=answer,
                            evidence={"question": question.id, "again": later.id})
    return None


def report_substance(row: Mapping[str, Any], context: Context) -> Optional[Dict[str, Any]]:
    """A report read as a finding, sent to the owner, as the owner took it."""
    if not _applied(row, "finding"):
        return None
    finding = context.sources.intention(row["refs"].get("intention"))
    if finding is None:
        return None
    metadata = finding.result_metadata if isinstance(finding.result_metadata, dict) else {}
    outreach = metadata.get("outreach") if isinstance(metadata.get("outreach"), dict) else {}
    message = context.sources.intention(outreach.get("row")) if outreach.get("state") == "sent" else None
    reaction = _reaction(message) if message is not None else {}
    kind = reaction.get("class")
    if kind not in {"negative", *ENGAGED}:
        return None
    read = _interpreted(reaction, context)
    source = "interpreted" if read else "behavioural"
    evidence = {"intention": finding.id, "message": getattr(message, "id", None), "reaction": kind,
                "read": "model" if read else "own words", **_reacted(reaction)}
    if kind in ENGAGED:
        return _outcome(row, context, "report_substance", "right", source=source, judged="finding",
                        label="finding", evidence=evidence)
    return _outcome(row, context, "report_substance", "wrong", source=source, judged="finding", evidence=evidence)


def owner_verdict(row: Mapping[str, Any], context: Context) -> Optional[Dict[str, Any]]:
    """A verdict the model withdrew that the owner restates: a later verdict in the same session within a day that
    stood."""
    if not _applied(row, "no"):
        return None
    turn = context.sources.turn(str(row["refs"].get("turn") or ""))
    decided = _utc(row["at"])
    if turn is None or decided is None or turn["at"] is None:
        return None
    later_rows = context.store.rows(since=decided, point="owner_verdict")
    for later in reversed(later_rows):          # oldest first
        if later["id"] == row["id"] or _applied(later, "no"):
            continue
        later_at = _utc(later["at"])
        if later_at is None or later_at <= decided or later_at - decided > RESTATE_WINDOW:
            continue
        other = context.sources.turn(str(later["refs"].get("turn") or ""))
        if (other is not None and other["session_id"] == turn["session_id"] and other["contact_id"] ==
                turn["contact_id"] and other["at"] is not None and other["at"] > turn["at"]):
            return _outcome(row, context, "owner_verdict", "wrong", source="behavioural", judged="no", label="yes",
                            evidence={"turn": turn["turn_id"], "restated": other["turn_id"]})
    return None


def model_yes(history: Iterable[tuple], *, now: datetime) -> List[Dict[str, Any]]:
    """A model's yeses to one ask (``ask_answer``), each graded from what the ask's own history (``(action, at,
    details)``) recorded after it; never from any tier's answer:

    - asked back (``ask_clarified`` read ``approve`` by the decision): right when the owner's next own answer approved
      the ask and the approval stood, wrong when it declined it or they undid the approval (explicit); ungraded while
      neither came, and for good once the ask was asked again first (the owner's answer is then the later question's:
      one answer grades one confirm-back). An answer a model read (``_owner_read`` fails) grades it ``interpreted``;
    - approved (``queued`` whose answer the decision read ``approve``): wrong when the owner undid it (explicit), right
      once it stood ``STANDS`` (behavioural); ungraded before.

    Each: ``decision``, ``at``, ``how`` (``asked_back`` or ``approved``), ``fingerprint`` (what the reading was made
    under), ``verdict`` (``right``, ``wrong`` or None), ``source`` and ``turns``: the turns of the owner's answers
    that graded it (their answer to the question back, their undo) where the history kept them, beside the turn the
    decision itself read (its own reference). Erasing any of them forgets the decision (``forget_erased``)."""
    from colonymind.mind.asks_report import ASK_CLARIFIED, ASK_UNDONE
    items = sorted(((str(action), _utc(at), dict(details or {})) for action, at, details in history
                    if _utc(at) is not None), key=lambda item: item[1])

    def undo(after: Sequence[tuple]) -> Optional[Mapping[str, Any]]:
        return next((more for next_action, _, more in after if next_action == ASK_UNDONE), None)

    def turns(*rows: Optional[Mapping[str, Any]]) -> List[str]:
        return [str(row["turn"]) for row in rows if row and row.get("turn")]

    out: List[Dict[str, Any]] = []
    for index, (action, at, details) in enumerate(items):
        answer = details.get("answer") if isinstance(details.get("answer"), dict) else {}
        later = items[index + 1:]
        if action == ASK_CLARIFIED and details.get("reader") == "decision" and details.get("read") == "approve":
            verdict = source = None
            graded_by: List[str] = []
            for position, (next_action, _, more) in enumerate(later):
                if next_action == ASK_CLARIFIED:
                    break           # asked again before any answer: the answer is the later question's, not this
                said = more.get("answer") if isinstance(more.get("answer"), dict) else {}
                if said.get("by") != "owner" or next_action not in {"queued", "outcome_denied"}:
                    continue
                # The owner's own word only where the tables read their words (or they answered at the console);
                # a model's reading of their reply is interpreted: never a label, never trust.
                source = "explicit" if _owner_read(said) else "interpreted"
                if next_action == "queued":
                    # An approval the owner then undid did not stand: no right reading, whoever said yes.
                    undone = undo(later[position + 1:])
                    verdict = "wrong" if undone is not None else "right"
                    source = "explicit" if undone is not None else source
                    graded_by = turns(said, undone)
                else:
                    verdict, graded_by = "wrong", turns(said)
                break
            out.append({"decision": details.get("decision"), "at": at, "how": "asked_back",
                        "fingerprint": str(details.get("fingerprint") or ""), "verdict": verdict,
                        "source": source, "turns": graded_by})
        elif action == "queued" and answer.get("reader") == "decision" and answer.get("read") == "approve":
            undone = undo(later)
            if undone is not None:
                verdict, source = "wrong", "explicit"
            elif now - at >= STANDS:
                verdict, source = "right", "behavioural"
            else:
                verdict = source = None
            out.append({"decision": answer.get("decision"), "at": at, "how": "approved",
                        "fingerprint": str(answer.get("fingerprint") or ""), "verdict": verdict, "source": source,
                        "turns": turns(undone)})
    return out


def model_no(history: Iterable[tuple]) -> List[Dict[str, Any]]:
    """A model's noes to one ask (``ask_answer``: ``outcome_denied`` whose answer the decision read ``decline``),
    each graded from the ask's own history: wrong (explicit) once the owner took it back (an ``ask_undone`` before
    anything else settled or reopened the ask: undo words, or "I didn't say no"), ungraded otherwise (a no that stands
    closes the ask; the owner's label grades it). Each as ``model_yes`` gives them, ``how`` ``declined``."""
    from colonymind.mind.asks_report import ASK_UNDONE
    items = sorted(((str(action), _utc(at), dict(details or {})) for action, at, details in history
                    if _utc(at) is not None), key=lambda item: item[1])
    out: List[Dict[str, Any]] = []
    for index, (action, at, details) in enumerate(items):
        answer = details.get("answer") if isinstance(details.get("answer"), dict) else {}
        if action != "outcome_denied" or answer.get("reader") != "decision" or answer.get("read") != "decline":
            continue
        undone = None
        for next_action, _, more in items[index + 1:]:
            if next_action == ASK_UNDONE:
                undone = more
                break
            if next_action in {"queued", "ask_revived"} or next_action.startswith("outcome_"):
                break
        out.append({"decision": answer.get("decision"), "at": at, "how": "declined",
                    "fingerprint": str(answer.get("fingerprint") or ""),
                    "verdict": "wrong" if undone is not None else None,
                    "source": "explicit" if undone is not None else None,
                    "turns": [str(undone["turn"])] if undone is not None and undone.get("turn") else []})
    return out


def _readings(history: List[tuple], *, now: datetime) -> List[Dict[str, Any]]:
    """A model's readings of the owner's answers to one ask that its history grades: its yeses and its noes."""
    return [*[{**item, "judged": "approve"} for item in model_yes(history, now=now)],
            *[{**item, "judged": "decline"} for item in model_no(history)]]


def _ask_answer_turns(rows: Sequence[Mapping[str, Any]], sources: "Sources") -> Dict[str, List[str]]:
    """Per ``ask_answer`` decision among ``rows``, the turns of the owner's answers that graded it after it was made
    (``model_yes``, ``model_no``), read from its ask's history."""
    from colonymind.util.temporal import now_utc
    out: Dict[str, List[str]] = {}
    histories: Dict[str, List[tuple]] = {}
    now = now_utc()                     # the turns do not depend on it; only the verdicts do
    for row in rows:
        ask = str((row.get("refs") or {}).get("ask") or "")
        if not ask:
            continue
        if ask not in histories:
            histories[ask] = sources.history(ask)
        for item in _readings(histories[ask], now=now):
            if item["decision"] == row["id"] and item["turns"]:
                out.setdefault(row["id"], []).extend(item["turns"])
    return out


def _owner_read(answer: Mapping[str, Any]) -> bool:
    """Whether a settling row's ``answer`` is the owner's own word: their words read by the phrase tables (a reply,
    a reaction, a code) or their answer at the console; never a model's reading of them (``reader: decision``)."""
    reader = answer.get("reader")
    return answer.get("by") == "owner" and (reader == "tables" or (not reader and answer.get("path") == "console"))


def ask_answer(row: Mapping[str, Any], context: Context) -> Optional[Dict[str, Any]]:
    """A model's yes or no to an ask, graded by the ask's own history (``model_yes``, ``model_no``)."""
    ask = (row.get("refs") or {}).get("ask")
    if not ask:
        return None
    graded = [item for item in _readings(context.sources.history(ask), now=context.now)
              if item["decision"] == row["id"] and item["verdict"]]
    if not graded:
        return None
    item = graded[0]
    return _outcome(row, context, "ask_answer", item["verdict"], source=item["source"], judged=item["judged"],
                    label=item["judged"] if item["verdict"] == "right" else None,
                    evidence={"ask": str(ask), "how": item["how"]})


# -- re-rendering (training by reference needs the rows to re-render) ----------------------------------------------

def _user_text(turn: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The first user message of a turn, as the reader that asked was given it."""
    if turn is None:
        return None
    return next((text for role, text in turn["messages"] if role == "user"), None)


def _owner_body(turn: Optional[Mapping[str, Any]]) -> Optional[str]:
    from colonymind.mind.reactions import strip_prefix
    text = _user_text(turn)
    return strip_prefix(text) if text is not None else None


def _load_owner_instruction(row: Mapping[str, Any], sources: Sources) -> Optional[Dict[str, Any]]:
    body = _owner_body(sources.turn(str(row["refs"].get("turn") or "")))
    return None if body is None else {"message": body}


def _load_outreach_reply(row: Mapping[str, Any], sources: Sources) -> Optional[Dict[str, Any]]:
    message = sources.intention(row["refs"].get("intention"))
    if message is None:
        return None
    body = _owner_body(sources.turn(str(_reaction(message).get("turn") or "")))
    context = message.context if isinstance(message.context, dict) else {}
    return None if body is None else {"text": body, "topic": str(context.get("topic") or "a finding")}


def _load_opt_out(row: Mapping[str, Any], sources: Sources) -> Optional[Dict[str, Any]]:
    text = _user_text(sources.turn(turn_ref("source", row["refs"].get("source")) or ""))
    return None if text is None else {"text": text}


def _load_report_substance(row: Mapping[str, Any], sources: Sources) -> Optional[Dict[str, Any]]:
    finding = sources.intention(row["refs"].get("intention"))
    if finding is None:
        return None
    context = finding.context if isinstance(finding.context, dict) else {}
    topic = str(context.get("topic") or context.get("concern") or finding.description)[:160]
    return {"topic": topic, "report": str(finding.result or "")}


def _load_owner_verdict(row: Mapping[str, Any], sources: Sources) -> Optional[Dict[str, Any]]:
    """The owner's message and the agent's reply before it in its session (as the night's lesson packet reads
    them: text messages only, the reply folded and cut to the packet's length)."""
    from colonymind.mind.lessons import REPLY_CHARS
    turn = sources.turn(str(row["refs"].get("turn") or ""))
    if turn is None:
        return None
    reply = ""
    for other in sources.session(turn["contact_id"], turn["session_id"]):
        for role, text in other["messages"]:
            if not text.strip():
                continue
            if other["turn_id"] == turn["turn_id"] and role == "user":
                return {"reply": reply, "text": text}
            if role != "user":
                reply = " ".join(text.split())[:REPLY_CHARS]
    return None


def _load_question_answer(row: Mapping[str, Any], sources: Sources) -> Optional[Dict[str, Any]]:
    """The question asked, the owner's reply and the conversation before it (as the question reader read it at
    once: the last turns of that session before the reply)."""
    from colonymind.mind.questions import EARLIER_TURNS, asked_of
    question = sources.intention(row["refs"].get("question"))
    turn = sources.turn(str(row["refs"].get("turn") or ""))
    reply = _user_text(turn)
    if question is None or turn is None or reply is None:
        return None
    asked = asked_of(question)
    before = []
    for other in sources.session(turn["contact_id"], turn["session_id"]):
        if other["turn_id"] == turn["turn_id"]:
            break
        if other["session_id"] != "mind" and not other["turn_id"].startswith("mind:"):
            before.append(other)
    lines = [f"{'Owner' if role == 'user' else 'Assistant'}: {' '.join(text.split())}"
             for other in before[-3:] for role, text in other["messages"] if text.strip()]
    while lines and len("\n".join(lines)) > 1500:
        lines.pop(0)
    return {"question": asked.get("text"), "first": asked.get("first"), "second": asked.get("second"),
            "earlier": " / ".join(lines[-EARLIER_TURNS * 2:]), "reply": reply}


def _load_no_reminders(row: Mapping[str, Any], sources: Any) -> Any:
    """The item as capture described it and the person's words (``commitments.extract``: the turn's user messages,
    its last ``CAPTURE_CHARS``): the item the turn's capture made whose description the ``about`` reference names.
    ``CHANGED`` when the turn's items are there and none is that one any more."""
    from colonymind.commitments.extract import _sides
    from colonymind.commitments.store import _normalize_desc

    from .log import Hashed
    refs = row.get("refs") or {}
    about = str(refs.get("about") or "")
    turn = sources.raw_turn(str(refs.get("turn") or ""))
    if turn is None or not about:
        return None
    made = sources.commitments_from(turn["turn_id"], person=turn["contact_id"])
    item = next((entry["description"] for entry in made
                 if sources.reference(Hashed(_normalize_desc(entry["description"]))) == about), None)
    if item is None:
        return CHANGED if made else None
    return {"item": item, "text": _sides(turn["messages"], CAPTURE_CHARS)[0]}


def _load_opinion_none(row: Mapping[str, Any], sources: Any) -> Optional[Dict[str, Any]]:
    """The turn as the opinion pass's packet reads it (``mind.opinions.build_packet``): its first four user messages,
    each cut to the packet's length, and the assistant's reply."""
    from colonymind.mind.opinions import REPLY_CHARS, TEXT_CHARS, _message_text
    turn = sources.raw_turn(str((row.get("refs") or {}).get("turn") or ""))
    if turn is None:
        return None
    said = [text for text in (_message_text(message).strip() for message in turn["messages"]
                              if message.get("role") == "user") if text]
    reply = "\n".join(text for text in (_message_text(message).strip() for message in turn["messages"]
                                        if message.get("role") == "assistant") if text)
    return {"said": "\n".join(text[:TEXT_CHARS] for text in said[:4]), "reply": reply[:REPLY_CHARS]}


def _load_appraisal_empty(row: Mapping[str, Any], sources: Any) -> Optional[Dict[str, Any]]:
    """The turn as a first appraisal takes it as evidence (``self_model.appraisals``): its user messages and runtime
    observations, each non-empty and at most ``APPRAISAL_MESSAGE_CHARS``. An appraisal that also quoted the person's
    earlier words reads otherwise, and does not re-render."""
    from colonymind.self_model.appraisals import _text
    turn = sources.raw_turn(str((row.get("refs") or {}).get("turn") or ""))
    if turn is None:
        return None
    texts = []
    for message in turn["messages"]:
        runtime = message.get("_native_runtime_observation") == "native-runtime-observation-v1"
        if message.get("role") != "user" and not runtime:
            continue
        text = _text(message)
        if text.strip() and len(text) <= APPRAISAL_MESSAGE_CHARS:
            texts.append(text)
    return {"text": "\n".join(texts)}


def _load_interest_settled(row: Mapping[str, Any], sources: Any) -> Any:
    """Read the retained interest by its hashed key, even after its level fell to zero."""
    from colonymind.commitments.extract import _sides

    from .log import Hashed
    refs = row.get("refs") or {}
    turn = sources.raw_turn(str(refs.get("turn") or ""))
    if turn is None or not refs.get("interest"):
        return None
    if not sources.owner_id or turn["contact_id"] != sources.owner_id or turn["scope"] != "person":
        return None
    entries = sources._query(sources.state_dir / MIND_DB,
                             "SELECT key, text FROM mind_state WHERE key LIKE 'interest:%'")
    for entry in entries:
        if sources.reference(Hashed(entry["key"])) == refs["interest"]:
            sources.mind = True
            return {"topic": entry["text"] or entry["key"].partition(":")[2],
                    "text": _sides(turn["messages"], CAPTURE_CHARS)[0]}
    return None


def _load_mind_wish_scope(row: Mapping[str, Any], sources: Any) -> Any:
    """The wish's ask record, first in its concern and then in the formed intention.

    Refused wishes have no retained ask. Withheld words render differently and cannot teach their old state.
    """
    ident = (row.get("refs") or {}).get("wish")
    if not ident:
        return None
    entries = sources._query(sources.state_dir / INITIATIVES_DB,
                             "SELECT id FROM initiatives WHERE json_extract(context, '$.wish.id')=?", (ident,))
    wish = None
    if entries:
        intention = sources.intention(entries[0]["id"])
        wish = intention.context.get("wish") if intention is not None else None
    else:
        entries = sources._query(sources.state_dir / MIND_DB,
                                 "SELECT detail_json FROM concerns WHERE intention_id IS NULL AND "
                                 "json_extract(detail_json, '$.extra.wish.id')=?", (ident,))
        if entries:
            wish = json.loads(entries[0]["detail_json"]).get("extra", {}).get("wish")
    if not isinstance(wish, dict):
        return None
    sources.mind = True
    return {"capability": wish.get("capability"), "topic": wish.get("topic"), "words": wish.get("words")}


def _load_ask_name(ask: Any, reference: str, sources: Any) -> Any:
    """Find the sentence's name by its keyed reference; a renamed contact cannot supply the old name."""
    from .log import Hashed
    candidates = ["a contact", "you"] if str(ask.entity_id) == sources.owner_id else ["a contact"]
    if ask.entity_id and str(ask.entity_id) != sources.owner_id:
        rows = sources._query(sources.contacts_db,
                              "SELECT display_name, given_name FROM contacts WHERE contact_id=? "
                              "AND deleted_at IS NULL", (str(ask.entity_id),))
        if rows:
            name = str(rows[0]["display_name"] or rows[0]["given_name"] or "").strip()
            if name and name != str(ask.entity_id):
                candidates = [name]
            else:
                handles = sources._query(sources.contacts_db,
                                         "SELECT address FROM contact_handles WHERE contact_id=? "
                                         "ORDER BY is_primary DESC, created_at", (str(ask.entity_id),))
                address = next((str(item["address"] or "").strip() for item in handles
                                if str(item["address"] or "").strip()), "")
                if address and str(ask.entity_id) not in address:
                    candidates = [address]
    return next((name for name in candidates if sources.reference(Hashed(name)) == reference), CHANGED)


def _load_ask_answer(row: Mapping[str, Any], sources: Any) -> Any:
    """Read a captured answer directly or through its message key; never guess a turn by text or time."""
    from dataclasses import replace

    from colonymind.mind import audit
    from colonymind.mind.outbox import ask_sentence

    refs = row.get("refs") or {}
    if not all(refs.get(key) for key in ("ask", "message", "name")):
        return None
    ask = sources.intention(refs["ask"])
    turn_id = str(refs.get("turn") or "")
    if not turn_id:
        entries = sources._query(sources.state_dir / MIND_DB,
                                 "SELECT key, turn_id FROM answer_turns WHERE turn_id IS NOT NULL")
        turn_id = next((str(entry["turn_id"]) for entry in entries
                        if sources.reference(entry["key"]) == refs["message"]), "")
        if not turn_id:
            turn_id = turn_ref("message", refs["message"]) or ""
    turn = sources.raw_turn(turn_id)
    if ask is None or turn is None or not sources.owner_id or turn["contact_id"] != sources.owner_id:
        return None
    if turn["scope"] != "person":
        return None
    # An ask's stored words still describe what was asked after the answer settled it.
    ask = replace(ask, status="asked")
    name = _load_ask_name(ask, refs["name"], sources) if audit.asked_words(ask) is not None else "a contact"
    if name is CHANGED:
        return CHANGED
    body = _user_text({"messages": sources._messages(json.dumps(turn["messages"]))})
    if body is None:
        return None
    return {"ask": audit.hidden_text(ask_sentence(ask, name=name)), "words": audit.hidden_text(body)}


def _load_claim_witness(sources: Any, ident: str, turn_id: str, message_hash: str) -> Any:
    """One claim as its pinned message stated it, even after a restatement moved its anchor."""
    from colonymind.turns.idempotency import source_message_hash
    from colonymind.mind.consolidate import Consolidation

    turn = sources.raw_turn(turn_id)
    if turn is None or turn["scope"] != "person":
        return None
    if message_hash not in {source_message_hash(turn["session_id"], message) for message in turn["messages"]}:
        return CHANGED
    rows = sources._query(sources.state_dir / TURN_DB,
        "SELECT c.id,c.turn_id,c.message_hash,c.subject_key,c.predicate,c.data_json,s.occurred_at,s.ingested_at "
        "FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
        "WHERE c.id=? AND c.turn_id=? AND c.message_hash=? UNION ALL "
        "SELECT c.id,w.turn_id,w.message_hash,c.subject_key,c.predicate,w.data_json,s.occurred_at,s.ingested_at "
        "FROM source_claim_witnesses w JOIN source_claims c ON c.id=w.claim_id "
        "JOIN turn_sources s ON s.turn_id=w.turn_id WHERE c.id=? AND w.turn_id=? AND w.message_hash=?",
        (ident, turn_id, message_hash, ident, turn_id, message_hash))
    if not rows:
        return None
    stored = rows[0]
    data = json.loads(stored["data_json"])
    if not isinstance(data, dict) or not Consolidation._scalar(data):
        return CHANGED
    # Scalar claims may name a subject from an earlier message. That message must remain readable too.
    basis = data.get("subject_basis_message")
    if isinstance(basis, dict):
        from colonymind.beliefs.source_projection import message_subject_basis
        named = sources.raw_turn(str(basis.get("turn_id") or ""))
        if named is None or named["contact_id"] != turn["contact_id"] or named["scope"] != "person":
            return None
        if basis.get("message_hash") not in {
                source_message_hash(named["session_id"], message) for message in named["messages"]}:
            return CHANGED
        with closing(sqlite3.connect((sources.state_dir / TURN_DB).resolve().as_uri() + "?mode=ro",
                                     uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            if message_subject_basis(conn, data, contact_id=turn["contact_id"]) is None:
                return None
    return {**data, **{key: stored[key] for key in ("id", "turn_id", "message_hash", "subject_key", "predicate")},
            "observed_at": stored["occurred_at"], "recorded_at": stored["ingested_at"],
            "contact_id": turn["contact_id"]}


def _load_claim_day(claim: Mapping[str, Any], reference: str, sources: Any) -> Any:
    """Recover the recorded local day without reading today's timezone setting or logging a date's words."""
    from .log import Hashed

    when = _utc(claim.get("observed_at") or claim.get("recorded_at"))
    candidates = [""] if when is None else [
        f"{day.strftime('%B')} {day.day}" for day in
        (when.astimezone(timezone.utc) + timedelta(days=offset) for offset in (-1, 0, 1))]
    return next((day for day in candidates if sources.reference(Hashed(day)) == reference), CHANGED)


def _load_claim_topic(claims: Sequence[Mapping[str, Any]], kind: str, reference: str, sources: Any) -> Any:
    """The pair's topic, or a group's first subject recovered from a permitted witness of the same key."""
    from colonymind.mind.consolidate import Consolidation, _clean
    from .log import Hashed

    first, second = claims

    def about(claim: Mapping[str, Any], predicate: str) -> str:
        if claim["subject_key"] == "speaker":
            return f"{'your' if claim['contact_id'] == sources.owner_id else 'their'} {predicate}"
        return f"the {predicate} of {_clean(claim.get('subject') or claim['subject_key'], 60)}"

    def matches(topic: str) -> bool:
        return sources.reference(Hashed(topic)) == reference

    if kind == "pair":
        left, right = [Consolidation._words(claim["predicate"]) for claim in claims]
        topic = (about(first, f"{left} and {right}") if first["subject_key"] == second["subject_key"] else
                 f"{about(first, left)} and {about(second, right)}")
        return topic if matches(topic) else CHANGED
    if kind != "group" or (first["subject_key"], first["predicate"]) != (
            second["subject_key"], second["predicate"]):
        return CHANGED
    predicate = Consolidation._words(first["predicate"])
    for claim in claims:
        topic = about(claim, predicate)
        if matches(topic):
            return topic
    # A group's first claim can differ from both endpoints. Locate candidates by keys, then read each source
    # through _Reading before looking at its words. A hidden witness must never supply the subject.
    args = (first["contact_id"], first["subject_key"], first["predicate"])
    candidates = sources._query(sources.state_dir / TURN_DB,
        "SELECT c.id,c.turn_id,c.message_hash FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
        "WHERE s.contact_id=? AND s.scope='person' AND c.subject_key=? AND c.predicate=? UNION "
        "SELECT c.id,w.turn_id,w.message_hash FROM source_claim_witnesses w "
        "JOIN source_claims c ON c.id=w.claim_id JOIN turn_sources s ON s.turn_id=w.turn_id "
        "WHERE s.contact_id=? AND s.scope='person' AND c.subject_key=? AND c.predicate=?", (*args, *args))
    hidden = False
    for candidate in candidates:
        claim = _load_claim_witness(sources, candidate["id"], candidate["turn_id"], candidate["message_hash"])
        if claim is None:
            hidden = True
            continue
        if claim is CHANGED:
            continue
        topic = about(claim, predicate)
        if matches(topic):
            return topic
    return None if hidden else CHANGED


def _load_claim_relation(row: Mapping[str, Any], sources: Any) -> Any:
    """The night's exact pair, pinned to its witnesses, with the topic and local days named by keyed hashes."""
    from colonymind.mind.consolidate import _statement

    from .log import Hashed
    refs = row.get("refs") or {}
    pins = [refs.get(name) for name in ("claims", "turns", "messages", "days", "times")]
    if any(not isinstance(items, list) or len(items) != 2 for items in pins):
        return None
    identifiers, turns, messages, days, times = pins
    claims = [_load_claim_witness(sources, ident, turn_ref("source", turn) or "", message)
              for ident, turn, message in zip(identifiers, turns, messages)]
    if any(claim is None for claim in claims):
        return None
    if any(claim is CHANGED for claim in claims):
        return CHANGED
    if claims[0]["contact_id"] != claims[1]["contact_id"]:
        return None
    if any(sources.reference(Hashed(str(claim.get("observed_at") or claim.get("recorded_at") or ""))) != stamp
           for claim, stamp in zip(claims, times)):
        return CHANGED
    dates = [_load_claim_day(claim, day, sources) for claim, day in zip(claims, days)]
    if any(day is CHANGED for day in dates):
        return CHANGED
    topic = _load_claim_topic(claims, str(refs.get("topic_kind") or ""), str(refs.get("about") or ""), sources)
    if topic is None or topic is CHANGED:
        return topic
    return {"about": topic, "older": _statement(claims[0]), "older_on": dates[0],
            "newer": _statement(claims[1]), "newer_on": dates[1]}


def _load_percept_attend(row: Mapping[str, Any], sources: Any) -> Any:
    """Read the owner's observation source and retained event metadata, including expired observations."""
    from colonymind.mind.observations import PREFIX, SOURCE_KIND
    from colonymind.turns.idempotency import canonical_turn_digest

    ident = str((row.get("refs") or {}).get("observation") or "")
    turn = sources.raw_turn(ident)
    if turn is None or not sources.owner_id or turn["contact_id"] != sources.owner_id:
        return None
    if turn["scope"] != "person" or turn["session_id"] != "mind":
        return None
    blocks = [block for message in turn["messages"] if isinstance(message.get("content"), list)
              for block in message["content"] if isinstance(block, dict)
              and block.get("type") == "observation" and block.get("kind") == SOURCE_KIND]
    if len(blocks) != 1:
        return None
    states = sources._query(sources.state_dir / MIND_DB,
                            "SELECT text FROM mind_state WHERE key LIKE ?", (PREFIX + "%",))
    for state in states:
        data = json.loads(state["text"] or "{}")
        entries = [*data.get("observations", []),
                   *(event.get("observation") for event in data.get("events", []))]
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("source_id") != ident:
                continue
            if entry.get("source_version") != canonical_turn_digest(turn["messages"]):
                return CHANGED
            return {"source": entry["source"], "severity": entry["severity"],
                    **{key: blocks[0][key] for key in ("subject", "feature", "summary")}}
    return None


LOADERS: Dict[str, Callable[[Mapping[str, Any], Any], Any]] = {
    "owner_instruction": _load_owner_instruction, "outreach_reply": _load_outreach_reply, "opt_out": _load_opt_out,
    "report_substance": _load_report_substance, "owner_verdict": _load_owner_verdict,
    "question_answer": _load_question_answer, "no_reminders": _load_no_reminders,
    "opinion_none": _load_opinion_none, "appraisal_empty": _load_appraisal_empty,
    "interest_settled": _load_interest_settled, "mind_wish_scope": _load_mind_wish_scope,
    "ask_answer": _load_ask_answer, "percept_attend": _load_percept_attend,
    "claim_relation": _load_claim_relation,
}
#: A loader's answer when the sources are there but no longer as the decision read them.
CHANGED = object()
#: Why a point has no loader (module docstring, Re-rendering), and what the report says of each reason.
NO_LOADER: Dict[str, str] = {"owner_send": "redacted"}
#: References these loaders need before they can distinguish a changed or missing source from an old log row.
LOADER_REFS: Dict[str, tuple] = {
    "interest_settled": ("turn", "interest"),
    "mind_wish_scope": ("wish",),
    "ask_answer": ("ask", "message", "name"),
    "claim_relation": ("claims", "turns", "messages", "about", "days", "times", "topic_kind"),
    "percept_attend": ("observation",),
}
NO_LOADER_REASONS: Dict[str, str] = {
    "redacted": "its fields are kept only redacted",
    "unreferenced": "a field it read is not among its references",
    "unlocated": "its references do not locate every field as it was read",
    "external": "its words live only in the store of the caller that asked it",
    "before_references": "logged before its references",
}
#: What a re-render found: ``matched`` (the hash matches), ``changed`` (the fields were found but render otherwise),
#: ``missing`` (the references locate nothing: an erased or deleted source), ``no_loader`` (the point's references
#: cannot locate its state), ``version`` (asked under another wording of its point), ``no_state`` (no state was
#: built: the rules spoke first).
RERENDERS = ("matched", "changed", "missing", "no_loader", "version", "no_state")


def no_loader_reason(name: str, *, row: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Why a point has no loader (a ``NO_LOADER_REASONS`` code), or None when it has one. A point the mind did not
    register is one a gate outside the sidecar asks (``external``). With ``row``, also distinguish decisions logged
    before a loader's required references from decisions whose referenced records are now missing."""
    from . import REGISTRY
    if name in LOADERS:
        if row is not None:
            refs = row.get("refs") or {}
            if any(not refs.get(key) for key in LOADER_REFS.get(name, ())):
                return "before_references"
        return None
    return NO_LOADER.get(name) or ("external" if name not in REGISTRY else None)


class _Reading:
    """The stores as one re-render reads them (``Sources``), with the instance's secret for a hashed reference
    (``reference``); it notes whose records it read, for the row's audience (``audience``)."""

    def __init__(self, sources: Sources, secret: bytes) -> None:
        self._sources, self._secret = sources, secret
        self.people: Set[str] = set()
        self.mind = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sources, name)

    def reference(self, value: Any) -> str:
        """A value as the log keeps it among a decision's references (``log.references``)."""
        from .log import references
        return str(references({"value": value}, self._secret).get("value") or "")

    def _read(self, turn: Optional[Mapping[str, Any]]) -> Any:
        if turn is not None:
            self.people.add(str(turn.get("contact_id") or ""))
        return turn

    def turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """A turn as ``Sources.turn`` reads it, None when it is withheld from projections (``Sources.withheld``)."""
        found = self._sources.turn(turn_id)
        if found is not None and self._sources.withheld([found["turn_id"]]):
            return None
        return self._read(found)

    def raw_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        return self._read(self._sources.raw_turn(turn_id))

    def session(self, contact_id: str, session_id: str) -> List[Dict[str, Any]]:
        """A session's turns less those a projection may not read (erased in part, ``Sources.erased``, or withheld,
        ``Sources.withheld``): a state that read one no longer renders as it did, so it does not re-render."""
        turns = self._sources.session(contact_id, session_id)
        ids = [turn["turn_id"] for turn in turns]
        hidden = self._sources.erased(ids) | self._sources.withheld(ids)
        return [self._read(turn) for turn in turns if turn["turn_id"] not in hidden]

    def intention(self, ident: Any) -> Any:
        found = self._sources.intention(ident)
        self.mind = self.mind or found is not None
        return found

    def questions(self) -> List[Any]:
        found = self._sources.questions()
        self.mind = self.mind or bool(found)
        return found

    def audience(self) -> Optional[str]:
        """The narrowest audience of what was read: ``owner`` where the owner's or the mind's own records were
        read; else ``contact`` where a filed contact's were; else ``guest``. None when nothing shows one."""
        if self.mind:
            return "owner"
        kinds = {self._sources.person(person) for person in self.people}
        if not kinds or None in kinds:
            return None
        return next(kind for kind in ("owner", "contact", "guest") if kind in kinds)


def _located(row: Mapping[str, Any], sources: Sources, secret: bytes, table: Mapping[str, Any]) -> tuple:
    """``(what a re-render found, the rendered state or None, the reading)``: the state only when it matches."""
    import hmac

    from .log import keyed_hash
    reading = _Reading(sources, secret)
    if not row.get("state_hash"):
        return "no_state", None, reading
    loader, point = LOADERS.get(row["point"]), table.get(row["point"])
    if loader is None or point is None or no_loader_reason(row["point"], row=row) is not None:
        return "no_loader", None, reading
    if point.version != row["version"]:
        return "version", None, reading
    try:
        fields = loader(row, reading)
    except Exception as error:
        logger.info("instinct re-render of %s failed (%s)", row["point"], type(error).__name__)
        fields = None
    if fields is CHANGED:
        return "changed", None, reading
    if fields is None:
        return "missing", None, reading
    state = point.render(fields)
    if not hmac.compare_digest(keyed_hash(secret, state), str(row["state_hash"])):
        return "changed", None, reading
    return "matched", state, reading


def rerender(row: Mapping[str, Any], sources: Sources, secret: bytes, *,
             registry: Optional[Mapping[str, Any]] = None) -> str:
    """Whether a logged decision re-renders from its references and matches its keyed hash (``RERENDERS``)."""
    from . import REGISTRY
    return _located(row, sources, secret, REGISTRY if registry is None else registry)[0]


def replay_share(store: Any, sources: Sources, *, since: Optional[datetime] = None,
                 registry: Optional[Mapping[str, Any]] = None) -> Dict[str, Counter]:
    """Per point, how its logged decisions re-render (``RERENDERS``). Empty without the instance's secret: a report
    never makes one (a new secret would match nothing logged)."""
    from .log import KEY_FILE
    out: Dict[str, Counter] = defaultdict(Counter)
    if not (Path(store.path).parent / KEY_FILE).is_file():
        return out
    secret = store.secret()
    for row in store.rows(since=since):
        out[row["point"]][rerender(row, sources, secret, registry=registry)] += 1
    return out


# -- training rows by reference (docs/DECISIONS.md, Calibration and learning) ---------------------------------------

#: Per outcome source, the teacher of a training row and where its label came from (``source_kind``); the owner's
#: own label (resolver ``log.OWNER``) is ``owner_label``, any other explicit outcome a resolver's ``verified_outcome``.
#: An interpreted outcome is no label.
TEACHERS: Dict[str, tuple] = {"explicit": ("verified_outcome", "resolver"),
                              "behavioural": ("behavioural_proxy", "behavioural"),
                              "synthetic": ("synthetic", "synthetic")}
OWNER_TEACHER = ("owner_label", "explicit")
#: The teachers that may only push away from a label (a margin on the disfavoured label, never a positive).
PUSH_AWAY = frozenset({"behavioural_proxy"})


def teacher(outcome: Mapping[str, Any]) -> Optional[tuple]:
    """``(teacher, source_kind)`` of an outcome (``TEACHERS``), or None when it is no label."""
    from .log import OWNER
    source = str(outcome.get("source") or "")
    if source == "explicit" and outcome.get("resolver") == OWNER:
        return OWNER_TEACHER
    return TEACHERS.get(source)


def _target(point: Any, outcome: Mapping[str, Any], taught: str) -> Optional[tuple]:
    """``(label, push_away)`` a row teaches from ``outcome``: the right label, or, where only the wrong one is known
    (or the teacher only pushes away), the label to push away from. None: it teaches nothing."""
    labels = set(point.labels)
    gold, judged = outcome.get("label"), outcome.get("judged")
    wrong = judged if outcome.get("verdict") == "wrong" and judged in labels and judged != gold else None
    if taught in PUSH_AWAY:
        return (wrong, True) if wrong is not None else None
    if gold in labels:
        return gold, False
    if wrong is None:
        return None
    if point.type == "yes_no":
        return ("no" if wrong == "yes" else "yes"), False
    return wrong, True


def replay_rows(store: Any, sources: Sources, *, point: Optional[str] = None, since: Optional[datetime] = None,
                registry: Optional[Mapping[str, Any]] = None) -> Iterable[Dict[str, Any]]:
    """The training rows of the logged decisions (module docstring, Training rows by reference), oldest first: per
    decision that re-renders ``matched`` and has an outcome that is a label, ``ref``, ``point``, ``version``, ``state``
    (in memory only), ``state_hash``, ``label``, ``teacher``, ``push_away``, ``source_kind``, ``audience`` and ``at``
    (ISO 8601, UTC). The outcome is the strongest of those whose grade may still be read (``sound``). Nothing
    without the instance's secret (none is ever made here)."""
    from . import REGISTRY
    from .fit import strongest
    from .log import KEY_FILE
    table = REGISTRY if registry is None else registry
    if not (Path(store.path).parent / KEY_FILE).is_file():
        return
    secret = store.secret()
    kept: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in store.outcomes(point=point):
        kept[item["decision"]].append(item)
    for row in reversed(store.rows(since=since, point=point)):
        if row["id"] not in kept or row["point"] not in LOADERS:
            continue                          # no outcome, or a point whose rows never re-render
        # The strongest outcome among those whose grade may still be read (``sound``).
        standing = [item for item in kept[row["id"]] if sound(item, row, sources)]
        outcome, spec = strongest(standing).get(row["id"]), table.get(row["point"])
        taught = teacher(outcome) if outcome is not None else None
        if taught is None or spec is None:
            continue
        target = _target(spec, outcome, taught[0])
        if target is None:
            continue
        found, state, reading = _located(row, sources, secret, table)
        if found != "matched":
            continue
        audience = reading.audience()
        at = _utc(row.get("at"))
        if audience is None or at is None:
            continue
        yield {"ref": row["id"], "point": row["point"], "version": row["version"], "state": state,
               "state_hash": row["state_hash"], "label": target[0], "teacher": taught[0], "push_away": target[1],
               "source_kind": taught[1], "audience": audience, "at": at.astimezone(timezone.utc).isoformat()}


RESOLVERS: Dict[str, Callable[[Mapping[str, Any], Context], Optional[Dict[str, Any]]]] = {
    "owner_send": owner_send, "owner_instruction": owner_instruction, "opt_out": opt_out,
    "question_answer": question_answer, "report_substance": report_substance, "owner_verdict": owner_verdict,
    "ask_answer": ask_answer,
}
#: The resolvers whose grade a later event in the stores can overturn: an ask's history may show an undo after the
#: confirmation a first pass graded right. Each pass reads them again and replaces their own earlier grade when the
#: history now grades the decision otherwise (never the owner's own label, which is kept under its own name).
REVISED = frozenset({"ask_answer"})


def _same(kept: Mapping[str, Any], outcome: Mapping[str, Any]) -> bool:
    return all(kept.get(key) == outcome.get(key) for key in ("source", "verdict", "judged", "label", "evidence"))


def resolve(store: Any, sources: Sources, *, now: datetime, window: timedelta = RESOLVE_WINDOW,
            resolvers: Optional[Mapping[str, Callable[..., Any]]] = None) -> Dict[str, Counter]:
    """Run every resolver over the decisions of the last ``window`` it has not labelled yet (and those of
    ``REVISED`` again); per point the count of right, wrong, revised and unresolved. Explicit wrongs run the tighten
    rule at once."""
    from . import fit
    table = RESOLVERS if resolvers is None else resolvers
    done = {(item["decision"], item["resolver"]): item
            for item in store.outcomes(since=now - window - RESOLVE_WINDOW)}
    context = Context(store, sources, now)
    counts: Dict[str, Counter] = defaultdict(Counter)
    explicit: Set[str] = set()
    for row in reversed(store.rows(since=now - window)):        # oldest first
        resolver = table.get(row["point"])
        kept = done.get((row["id"], row["point"]))
        if resolver is None or (kept is not None and row["point"] not in REVISED):
            continue
        try:
            outcome = resolver(row, context)
        except Exception as error:
            logger.warning("instinct resolver %s failed (%s)", row["point"], type(error).__name__)
            counts[row["point"]]["failed"] += 1
            continue
        if outcome is None:
            counts[row["point"]]["unresolved"] += kept is None
            continue
        if kept is not None and _same(kept, outcome):
            continue
        if store.put_outcome(outcome, replace=kept is not None):
            counts[row["point"]][outcome["verdict"]] += 1
            counts[row["point"]]["revised"] += kept is not None
            if outcome["source"] == "explicit" and outcome["verdict"] == "wrong":
                explicit.add(row["point"])
    for point in sorted(explicit):
        fit.tighten(store, point, now=now)
    return counts


def label(store: Any, decision: str, value: str, *, now: datetime, registry: Optional[Mapping[str, Any]] = None,
          by: str = "owner") -> Dict[str, Any]:
    """The owner's explicit label for a decision: the right label there, judged against what was decided (the
    applied label, else what the existing path kept). It takes precedence over any outcome a resolver recorded for
    the decision (``log.precedence``). KeyError for an unknown decision, ValueError for a label the point does not
    have. The tighten rule runs at once."""
    from . import REGISTRY, fit
    table = REGISTRY if registry is None else registry
    row = store.row(decision)
    if row is None:
        raise KeyError(decision)
    point = table.get(row["point"])
    if point is None or value not in point.labels:
        raise ValueError(f"{row['point']} has no label {value!r}"
                         + (f" ({', '.join(point.labels)})" if point is not None else ""))
    # What was decided there: the applied label, else what the existing path kept (nothing applied at a point
    # whose caller keeps its own reading is that reading).
    judged = row.get("label") if row.get("applied") else fit.existing_answer(row)
    outcome = {"decision": row["id"], "point": row["point"], "resolver": by, "source": "explicit",
               "verdict": "right" if judged == value else "wrong", "judged": judged, "label": value,
               "at": now.isoformat(), "evidence": {}}
    store.put_outcome(outcome, replace=True)          # the owner's word replaces an earlier one of theirs
    fit.tighten(store, row["point"], now=now, registry=table)
    return outcome


#: Per point, the turns of the owner's answers that graded its decisions after they were made (beyond each
#: decision's own references), read from the stores: ``{decision: [turn, ...]}`` for the given rows.
GRADED_BY: Dict[str, Callable[[Sequence[Mapping[str, Any]], Sources], Dict[str, List[str]]]] = {
    "ask_answer": _ask_answer_turns,
}


def _restatement(outcome: Mapping[str, Any], sources: Sources) -> List[str]:
    """``owner_verdict``: the turn of the later verdict that restated the withdrawn one."""
    turn = (outcome.get("evidence") or {}).get("restated")
    return [str(turn)] if turn else []


def _reaction_turn(key: str) -> Callable[[Mapping[str, Any], Sources], List[str]]:
    """The turn of the owner's reaction a grade was read from: kept in its evidence (``reacted``), or, for a grade
    kept before it was, the reaction the message (the evidence's ``key``) holds now."""
    def turns(outcome: Mapping[str, Any], sources: Sources) -> List[str]:
        evidence = outcome.get("evidence") or {}
        turn = evidence.get("reacted")
        if not turn and evidence.get(key):
            message = sources.intention(evidence[key])
            turn = _reaction(message).get("turn") if message is not None else None
        return [str(turn)] if turn else []
    return turns


#: Per resolver, the turns a grade it kept rests on beyond its decision's own references (read from the outcome and
#: the stores): the owner's restatement of a withdrawn verdict, their reaction to a message the gate sent or a
#: finding reported. An erased one forgets the decision (``forget_erased``); an erased or withheld one makes the
#: grade no label for training (``sound``).
RESTS_ON: Dict[str, Callable[[Mapping[str, Any], Sources], List[str]]] = {
    "owner_verdict": _restatement, "owner_send": _reaction_turn("intention"),
    "report_substance": _reaction_turn("message"),
}


def sound(outcome: Mapping[str, Any], row: Mapping[str, Any], sources: Sources) -> bool:
    """Whether every turn a grade rests on beyond its decision's references (``RESTS_ON``, ``GRADED_BY``) may still
    be read: none erased (``Sources.erased``) and none withheld from projections (``Sources.withheld``). The owner's
    own label rests on nothing else."""
    from .log import OWNER
    if outcome.get("resolver") == OWNER:
        return True
    rests = RESTS_ON.get(str(outcome.get("resolver") or ""))
    turns = rests(outcome, sources) if rests is not None else []
    graded_by = GRADED_BY.get(str(row.get("point") or ""))
    if graded_by is not None:
        turns += graded_by([row], sources).get(row["id"], [])
    return not turns or not (sources.erased(turns) | sources.withheld(turns))


def forget_erased(store: Any, sources: Sources, *, erasure: Optional[Callable[[Sequence[str]], Any]] = None) -> int:
    """Forget the decisions whose references name an erased turn, or whose grade rests on an answer or a reaction in
    one (``GRADED_BY``, ``RESTS_ON``); their counts stay. The number forgotten."""
    rows = store.rows()
    named: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        for key, value in (row.get("refs") or {}).items():
            for item in value if isinstance(value, list) else [value]:
                turn = turn_ref(key, item)
                if turn:
                    named[turn].append(row["id"])
    for point, graded_by in GRADED_BY.items():
        for decision, turns in graded_by([row for row in rows if row["point"] == point], sources).items():
            for turn in turns:
                named[turn].append(decision)
    for outcome in store.outcomes():
        rests = RESTS_ON.get(str(outcome.get("resolver") or ""))
        for turn in rests(outcome, sources) if rests is not None else []:
            named[turn].append(outcome["decision"])
    erased = sources.erased(named)
    ids = [ident for turn in erased for ident in named[turn]]
    return store.forget(ids, reason="erased", **({"erasure": erasure} if erasure is not None else {}))


__all__ = ["APPRAISAL_MESSAGE_CHARS", "CAPTURE_CHARS", "CHANGED", "COMMITMENTS_DB", "ENGAGED", "GRADED_BY",
           "INSTRUCTED", "LOADERS", "NO_LOADER", "NO_LOADER_REASONS", "OWNER_TEACHER", "OWN_WORDS", "PUSH_AWAY",
           "RERENDERS", "RESOLVERS", "RESOLVE_WINDOW", "RESTATE_WINDOW", "RESTS_ON", "RESUME_WINDOW", "REVISED",
           "SOURCES", "STANDS", "TEACHERS", "VERDICTS", "Context", "Sources", "ask_answer", "forget_erased", "label",
           "model_no", "model_yes", "no_loader_reason", "replay_rows", "replay_share", "rerender", "resolve", "sound",
           "teacher", "turn_ref"]
