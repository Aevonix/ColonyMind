"""Durable evidence ancestry for accepted hand results, relays and lessons.

The controller owns evidence levels. These helpers never infer a level from
assistant prose, and source entries keep receipts after a branch expires.
Lifecycle methods can join a controller transaction or run independently.
"""

from __future__ import annotations

from colonymind.memory.curated import forget_sources

import asyncio
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Mapping

from .store import DB_FILE


LEVELS = {f"E{number}": number for number in range(6)}
# Assumptions for bounded recall display and reads; the erasure batch size is
# the canonical ledger's existing 100-source API limit.
READ_TIMEOUT_SECONDS = 1
DISPLAY_NAME_CHARS = 120
ERASE_BATCH_SIZE = 100
LABELS = {"E0": "own analysis, not checked", "E1": "execution observed, not checked",
          "E2": "artifact validated, requirements not checked", "E3": "requirement tested",
          "E4": "independently reviewed", "E5": "owner accepted"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _receipts(evidence):
    receipts = evidence.get("receipts", evidence.get("check", []))
    return [item for item in receipts if isinstance(item, Mapping)] if isinstance(receipts, list) else []


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _review_receipt(evidence, *, producer_families=(), final_reference=None):
    """Normalize a controller-owned review of the exact final result."""
    scope = evidence.get("scope") or {}
    families = set(producer_families) | set(scope.get("producer_families") or [])
    if (scope.get("verdict") != "pass" or not scope.get("reviewer_family") or not families
            or scope["reviewer_family"] in families or not scope.get("reference")
            or not _hash(scope.get("sha256"))
            or final_reference is not None and scope["reference"] != final_reference):
        return None
    return {"kind": "independent_review", "scope": scope, "passed": True}


def _supported_evidence(hand, ledger, owner_id):
    """Keep runner receipts or embed the trusted review/verdict-reader receipt.

    The upstream verdict reader decides that an owner answer accepts this
    result. This function verifies attribution and retains the exact source;
    it never classifies an arbitrary owner message as acceptance.
    """
    evidence = {**(hand.get("evidence") or {}), "receipts": hand.get("check", [])}
    if evidence["receipts"]:
        return evidence
    scope = evidence.get("scope") or {}
    receipt = None
    if evidence.get("level") == "E4":
        receipt = _review_receipt(evidence, producer_families=hand.get("families_used") or [],
                                  final_reference=hand.get("branch") or "")
    elif evidence.get("level") == "E5":
        source_id = (scope.get("details") or {}).get("owner_answer_ref")
        if isinstance(source_id, str):
            source_id = source_id.removeprefix("turn:")
            source = _owner_source(ledger, owner_id, source_id)
            owner_messages = [message for message in source["messages"] if message.get("role") == "user"] if source else []
            if (owner_messages and (source["scope"] == "person" or source["session_id"] == hand["parent_session"])
                    and scope.get("reference") == hand.get("branch") and _hash(scope.get("sha256"))):
                from colonymind.turns.idempotency import canonical_turn_digest
                receipt = {"kind": "owner_acceptance", "scope": scope, "passed": True,
                    "source_id": source_id, "source_version": canonical_turn_digest(source["messages"]),
                    "contact_id": owner_id, "session_id": source["session_id"],
                    "occurred_at": source["occurred_at"], "owner_messages": owner_messages}
    evidence["receipts"] = [receipt] if receipt else []
    return evidence


def lesson_allowed(evidence, *, owner_verdict=False, observed_failure=False):
    """Gate trusted controller evidence; a claimed failure alone proves nothing."""
    if not isinstance(evidence, Mapping) or evidence.get("disputed") or evidence.get("erased"):
        return False
    if owner_verdict:
        return True
    receipts = _receipts(evidence)
    if observed_failure:
        return any(receipt.get("passed") is False and (
            receipt.get("output_hash") or receipt.get("scope")) for receipt in receipts)
    if not receipts and "receipts" not in evidence and "check" not in evidence and evidence.get("level") == "E4":
        return _review_receipt(evidence) is not None
    return LEVELS.get(evidence.get("level"), -1) >= 3 and bool(receipts)


def _read_db(path):
    """Read existing state without creating a database during recall."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=READ_TIMEOUT_SECONDS)
    connection.row_factory = sqlite3.Row
    return connection


def _stored_hand(path, hand_id):
    try:
        with closing(_read_db(path)) as db:
            row = db.execute("SELECT * FROM hands WHERE id=?", (hand_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            for key in ("evidence", "check", "decisions", "families_used"):
                result[key] = json.loads(result[key]) if result[key] else None
            return result
    except (sqlite3.Error, OSError, ValueError):
        return None


def _owner_source(ledger, owner_id, source_id):
    from colonymind.memory import curated
    if curated.is_forgotten(ledger, source_id):
        return None
    with closing(ledger._connect()) as db:
        row = db.execute("SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=? "
                         "AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=turn_id)",
                         (source_id, owner_id)).fetchone()
    if row is None:
        return None
    return dict(row, messages=json.loads(row["messages_json"]))


def _owner_verdict(ledger, owner_id, source_id):
    row = _owner_source(ledger, owner_id, source_id) if source_id else None
    return bool(row and any(message.get("role") == "user" for message in row["messages"]))


def lineage_hands(ledger, owner_id, lineage):
    """Resolve hand ancestry from stored assistant metadata, never lesson prose."""
    hands = set()
    if ledger is None or not owner_id:
        return hands
    for identifier in lineage:
        source = _owner_source(ledger, owner_id, str(identifier).removeprefix('turn:'))
        for message in source['messages'] if source else ():
            if message.get('role') != 'assistant':
                continue
            metadata = message.get('metadata') or {}
            if metadata.get('origin') == 'hand' and metadata.get('hand_id'):
                hands.add(metadata['hand_id'])
            for ref in metadata.get('hand_refs') or []:
                if isinstance(ref, Mapping) and ref.get('hand_id'):
                    hands.add(ref['hand_id'])
    return hands


def hand_lesson_allowed(ledger, *, owner_id, lineage, evidence=None,
                        owner_verdict=False, observed_failure=False):
    """Authorize hand lessons from stored evidence, never a supplied level.

    A retained accepted source embeds its checks. A failure or owner verdict
    instead names the hand and the actual controller receipt or owner source.
    """
    if ledger is None or not owner_id:
        return not evidence and not lineage
    path = Path(ledger.db_path).parent / DB_FILE
    def allowed_hand(identifier, captured_level=None):
        hand = _stored_hand(path, identifier)
        if hand is None or any((hand.get("decisions") or {}).get(k) for k in ("erased", "disputed")):
            return False
        trusted = _supported_evidence(hand, ledger, owner_id)
        if captured_level is not None and LEVELS.get(captured_level, -1) < 3:
            return False
        if evidence and evidence.get("hand_id") == identifier:
            if _owner_verdict(ledger, owner_id, evidence.get("owner_ref")):
                return lesson_allowed(trusted, owner_verdict=True)
            if observed_failure or evidence.get("observed_failure"):
                return lesson_allowed(trusted, observed_failure=True)
        with closing(_read_db(path)) as db:
            accepted = hand["state"] == "done" or db.execute("SELECT 1 FROM events WHERE hand_id=? "
                "AND type='accepted_judgment'", (identifier,)).fetchone()
        return bool(accepted and lesson_allowed(trusted))

    for identifier in lineage:
        row = _owner_source(ledger, owner_id, str(identifier).removeprefix("turn:"))
        for message in row["messages"] if row else ():
            metadata = message.get("metadata") or {}
            if message.get("role") != "assistant":
                continue
            refs = metadata.get("hand_refs")
            if refs:
                if not isinstance(refs, list) or any(not isinstance(ref, Mapping) or not allowed_hand(
                        ref.get("hand_id"), ref.get("evidence_level", ref.get("level", "E0"))) for ref in refs):
                    return False
            if metadata.get("origin") != "hand":
                continue
            hand = _stored_hand(path, metadata.get("hand_id"))
            if hand is None or any((hand.get("decisions") or {}).get(k) for k in ("erased", "disputed")):
                return False
            with closing(_read_db(path)) as db:
                derivation = db.execute("SELECT state FROM derivations WHERE derived_kind='autobiography' "
                    "AND derived_id=? AND hand_id=?", (row["turn_id"], hand["id"])).fetchone()
            supported = {**(metadata.get("evidence") or {}), "receipts": metadata.get("receipts", [])}
            if not (derivation and derivation[0] == "live" and lesson_allowed(supported)):
                return False
    if evidence is not None:
        return isinstance(evidence, Mapping) and bool(evidence.get("hand_id")) and allowed_hand(evidence["hand_id"])
    # Existing mind lessons keep their ordinary intake. Hand lineage was
    # checked above, including a relay admitted by the nightly lesson reader.
    return True


def relay_marker(hand_refs, *, controller_path=None, derived_id=None):
    """Label the shown evidence, bounded by both capture and current records."""
    if not isinstance(hand_refs, list):
        return ""
    markers = []
    for ref in hand_refs:
        if not isinstance(ref, Mapping) or not ref.get("hand_id"):
            continue
        level = ref.get("evidence_level", ref.get("level", "E0"))
        level = level if level in LEVELS else "E0"
        state = ref.get("state", "live")
        hand = _stored_hand(controller_path, ref["hand_id"]) if controller_path else None
        decisions = (hand or {}).get("decisions") or {}
        if hand and (hand.get("evidence") or {}).get("level") in LEVELS:
            current = hand["evidence"]["level"]
            level = min((level, current), key=LEVELS.__getitem__)
        if controller_path and derived_id:
            try:
                with closing(_read_db(controller_path)) as db:
                    row = db.execute("SELECT state FROM derivations WHERE derived_kind='relay' "
                        "AND derived_id=? AND hand_id=?", (derived_id, ref["hand_id"])).fetchone()
                if row:
                    state = row[0]
            except (sqlite3.Error, OSError, ValueError):
                pass
        label = ("evidence retracted" if decisions.get("erased") or state == "retracted" else
                 "disputed by owner" if decisions.get("disputed") or state == "disputed" else
                 "evidence expired" if decisions.get("evidence_expired") or state == "downgraded" else LABELS[level])
        if LEVELS[level] >= 3 and label == LABELS[level]:
            scope = ((hand or {}).get("evidence") or {}).get("scope") or ref.get("scope")
            label += "; scope: " + (_json(scope) if scope else "unavailable")
        name = str(ref.get("name") or ref["hand_id"]).replace("\n", " ")[:DISPLAY_NAME_CHARS]
        markers.append(f"from {name}'s result, {level}, {label}")
    return "; ".join(dict.fromkeys(markers))


def recalled_hand_marker(memory, *, ledger_path=None, controller_path=None):
    """Resolve provenance for an already authorized quotation or typed pair."""
    from colonymind import get_state_dir
    from colonymind.turns.idempotency import source_message_hash

    identifier, message_hash = memory.get("source_turn_id"), memory.get("source_message_hash")
    if memory.get("content_format") == "source_conversation_v1":
        try:
            pair = json.loads(memory.get("content", ""))
            if memory.get("epistemic_state") == "correction_evidence":
                pair = pair["original"]["content"]
                pair = json.loads(pair) if isinstance(pair, str) else pair
            response = pair["response"]
            if response.get("role") != "assistant":
                return ""
            identifier, message_hash = response["source_id"], response["source_message_hash"]
        except (ValueError, TypeError, KeyError):
            return ""
    elif memory.get("kind") != "source_quote" or memory.get("role") != "assistant":
        return ""
    if not identifier or not message_hash or not memory.get("contact_id"):
        return ""
    # The same ledger the server opens (get_turn_idempotency_ledger(get_state_dir())).
    ledger_path = Path(ledger_path) if ledger_path else get_state_dir() / "turn-idempotency.db"
    controller_path = controller_path or ledger_path.parent / DB_FILE
    try:
        with closing(_read_db(ledger_path)) as db:
            row = db.execute("SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=? "
                "AND (scope='person' OR session_id=?)", (identifier, memory["contact_id"],
                                                        memory.get("session_id"))).fetchone()
        for message in json.loads(row["messages_json"]) if row else ():
            if message.get("role") == "assistant" and source_message_hash(row["session_id"], message) == message_hash:
                metadata = message.get("metadata") or {}
                if metadata.get("origin") == "hand":
                    evidence = metadata.get("evidence") or {}
                    hand = _stored_hand(controller_path, metadata.get("hand_id"))
                    decisions = (hand or {}).get("decisions") or {}
                    state = "retracted" if decisions.get("erased") else "disputed" if decisions.get("disputed") else "live"
                    # The entry's receipt remains support after branch expiry.
                    return relay_marker([{"hand_id": metadata.get("hand_id"),
                        "evidence_level": evidence.get("level", "E0"), "scope": evidence.get("scope"), "state": state}])
                return relay_marker(metadata.get("hand_refs"),
                                    controller_path=controller_path, derived_id=identifier)
    except (sqlite3.Error, OSError, ValueError, TypeError):
        return ""
    return ""


_LESSON_ADMISSION = object()


class Provenance:
    """Lifecycle seam shared by controller acceptance and memory maintenance."""

    def __init__(self, controller, ledger, *, owner_id, lessons=None):
        self.controller, self.ledger = controller, ledger
        self.owner_id, self.lessons = owner_id, lessons

    async def _mutate(self, operation, callback):
        if self.controller._writer is asyncio.current_task():
            return await callback()
        return await self.controller.write(None, operation, {}, callback, record=False)

    async def _derive(self, kind, identifier, hand, reference, state="live"):
        await self.controller.execute("INSERT OR IGNORE INTO derivations "
            "(derived_kind,derived_id,hand_id,evidence_level,evidence_ref,state) VALUES (?,?,?,?,?,?)",
            (kind, identifier, hand["id"], (hand.get("evidence") or {}).get("level", "E0"), reference, state))

    async def record_accepted(self, hand_id, text):
        """One owner source per accepted result, including the full receipts."""
        async def apply():
            c = self.controller
            hand = await c.get_hand(hand_id)
            if not self.owner_id or self.ledger is None or not str(text).strip():
                return None
            if self.controller.parent_forgotten(hand) or any(hand["decisions"].get(k) for k in ("erased", "disputed")):
                return None
            evidence = _supported_evidence(hand, self.ledger, self.owner_id)
            if not evidence["receipts"] or not lesson_allowed(evidence):
                return None
            accepted = hand["state"] == "done" or await c.one(
                "SELECT 1 FROM events WHERE hand_id=? AND type='accepted_judgment'", (hand_id,))
            if not accepted:
                return None
            prior = await c.one("SELECT derived_id,state FROM derivations WHERE hand_id=? "
                                "AND derived_kind='autobiography'", (hand_id,))
            if prior:
                return prior["derived_id"] if prior["state"] == "live" and _owner_source(
                    self.ledger, self.owner_id, prior["derived_id"]) else None
            reference = sha256(_json(evidence).encode()).hexdigest()
            identifier = f"hand:{hand_id}:{reference}"
            if _owner_source(self.ledger, self.owner_id, identifier):
                await self._derive("autobiography", identifier, hand, reference)
                return identifier
            metadata = {"origin": "hand", "hand_id": hand_id, "evidence": hand["evidence"],
                        "receipts": evidence["receipts"], "parent_session": hand["parent_session"]}
            message = {"role": "assistant", "content": str(text).strip(), "metadata": metadata}
            with closing(self.ledger._connect()) as db:
                parents = [row[0] for row in db.execute("SELECT turn_id FROM turn_sources "
                    "WHERE contact_id=? AND session_id=?", (self.owner_id, hand["parent_session"]))]
            parents.extend(receipt["source_id"] for receipt in evidence["receipts"]
                           if receipt.get("kind") == "owner_acceptance")
            if parents:
                message["_supplied_sources"] = self.ledger.source_references(
                    parents, contact_id=self.owner_id, session_id=hand["parent_session"])
            from colonymind.turns.idempotency import SourceErased
            try:
                self.ledger.record_source(identifier, contact_id=self.owner_id, session_id=hand["parent_session"],
                    messages=[message], scope="person", occurred_at=hand["ended_at"] or hand["created_at"],
                    derive_claims=False)
            except SourceErased:
                return None
            await self._derive("autobiography", identifier, hand, reference)
            return identifier
        return await self._mutate("/provenance/accepted", apply)

    async def record_relay(self, turn_id, hand_ids):
        """Capture exactly the controller results shown in a turn's digest."""
        async def apply():
            refs = []
            for identifier in dict.fromkeys(hand_ids):
                hand = await self.controller.get_hand(identifier)
                if hand["decisions"].get("erased") or self.controller.parent_forgotten(hand):
                    continue
                level = (hand.get("evidence") or {}).get("level", "E0")
                state = "downgraded" if any(hand["decisions"].get(k) for k in ("evidence_expired", "disputed")) else "live"
                await self._derive("relay", turn_id, hand, hand.get("branch") or "", state)
                refs.append({"hand_id": identifier, "evidence_level": level, "name": hand["name"]})
            return refs
        return await self._mutate("/provenance/relay", apply)

    async def admit_lesson(self, hand_id, fields, *, owner_ref=None, observed_failure=False):
        """Admit a supported lesson once per hand, with an erasure dependency."""
        async def apply():
            if self.lessons is None:
                return None
            c = self.controller
            hand = await c.get_hand(hand_id)
            if self.controller.parent_forgotten(hand) or any(hand["decisions"].get(k) for k in ("erased", "disputed")):
                return None
            prior = await c.one("SELECT * FROM derivations WHERE hand_id=? AND derived_kind='lesson'", (hand_id,))
            if prior:
                return self.lessons.get(prior["derived_id"]) if prior["state"] == "live" else None
            # A ledger commit can outlive a controller rollback. Recover its
            # original lesson instead of admitting a second one on replay.
            existing = next((lesson for lesson in self.lessons.all(include_closed=True)
                             if f"hand:{hand_id}" in lesson.evidence and lesson.origin == "hand"), None)
            if existing is not None:
                await self._derive("lesson", existing.id, hand, hand.get("branch") or "",
                                   "live" if existing.status in {"active", "candidate"} else "downgraded")
                return existing if existing.status in {"active", "candidate"} else None
            entries = await c.all("SELECT derived_id FROM derivations WHERE hand_id=? "
                                 "AND derived_kind='autobiography' AND state='live'", (hand_id,))
            lineage = [row["derived_id"] for row in entries]
            if owner_ref:
                lineage.append(owner_ref)
            evidence = {**_supported_evidence(hand, self.ledger, self.owner_id), "hand_id": hand_id,
                        "owner_ref": owner_ref,
                        "observed_failure": observed_failure}
            if not hand_lesson_allowed(self.ledger, owner_id=self.owner_id, lineage=lineage, evidence=evidence):
                return None
            lesson = self.lessons.admit(fields, verified="owner" if owner_ref else "check", origin="hand",
                status="active", evidence=[f"hand:{hand_id}"], lineage=lineage, hand_evidence=evidence,
                _hand_admission=_LESSON_ADMISSION)
            if lesson is not None:
                await self._derive("lesson", lesson.id, hand, hand.get("branch") or "")
            return lesson
        return await self._mutate("/provenance/lesson", apply)

    async def erase_parent(self, session):
        """Retract sources before the controller removes their ancestry rows."""
        async def apply():
            c = self.controller
            hands = await c.all("SELECT * FROM hands WHERE parent_session=?", (session,))
            identifiers = set()
            for hand in hands:
                for row in await c.all("SELECT * FROM derivations WHERE hand_id=?", (hand["id"],)):
                    identifiers.add(f"mind:lesson:{row['derived_id']}:admitted" if row["derived_kind"] == "lesson"
                                    else row["derived_id"])
                await c.update("hands", "id", hand["id"], decisions={**hand["decisions"], "erased": True})
                await c.execute("UPDATE derivations SET state='retracted',evidence_ref='' WHERE hand_id=?", (hand["id"],))
            self._erase_sources(identifiers)
            if self.lessons is not None:
                self.lessons.prune_vectors()
            return {"erased": len(hands)}
        return await self._mutate("/provenance/erase", apply)

    def _erase_sources(self, identifiers):
        identifiers = sorted(identifiers)
        for offset in range(0, len(identifiers), ERASE_BATCH_SIZE):
            # An earlier batch can erase later entries through source lineage.
            existing = [identifier for identifier in identifiers[offset:offset + ERASE_BATCH_SIZE]
                        if _owner_source(self.ledger, self.owner_id, identifier)]
            if existing:
                forget_sources(self.ledger, self.owner_id, source_ids=existing)

    async def expire_branch(self, branch_ref):
        """Expire relay support without deleting entries with embedded checks."""
        async def apply():
            hands = await self.controller.all("SELECT * FROM hands WHERE branch=?", (branch_ref,))
            for hand in hands:
                await self.controller.update("hands", "id", hand["id"], decisions={
                    **hand["decisions"], "evidence_expired": True})
                await self.controller.execute("UPDATE derivations SET state='downgraded' "
                    "WHERE hand_id=? AND derived_kind='relay' AND state='live'", (hand["id"],))
            return {"expired": len(hands)}
        return await self._mutate("/provenance/expiry", apply)

    async def dispute(self, hand_id, owner_ref):
        """Keep the accepted entry with an attributed owner correction."""
        async def apply():
            if not _owner_verdict(self.ledger, self.owner_id, owner_ref):
                raise ValueError("owner verdict source required")
            c = self.controller
            hand = await c.get_hand(hand_id)
            if hand["decisions"].get("erased"):
                return {"disputed": False}
            from colonymind.turns.idempotency import canonical_turn_digest
            from colonymind.turns.source_annotations import append
            for row in await c.all("SELECT * FROM derivations WHERE hand_id=?", (hand_id,)):
                if row["derived_kind"] == "autobiography":
                    source = _owner_source(self.ledger, self.owner_id, row["derived_id"])
                    if source:
                        append(self.ledger, contact_id=self.owner_id, session_id=source["session_id"],
                            annotation_id=sha256(f"{hand_id}:{owner_ref}".encode()).hexdigest(),
                            source_id=row["derived_id"], source_version=canonical_turn_digest(source["messages"]),
                            excerpt=source["messages"][0]["content"][:4096],
                            correction="Owner disputed this hand result.", author_principal=self.owner_id)
                elif row["derived_kind"] == "lesson":
                    lessons = self.lessons
                    if lessons is None:
                        from colonymind.mind.lessons import Lessons
                        from colonymind.mind.outcomes import Autobiography
                        lessons = Lessons(ledger=self.ledger, store=None, owner_id=self.owner_id,
                            autobiography=Autobiography(self.ledger, owner_id=self.owner_id, clock=c.clock), clock=c.clock)
                    lessons.set_status(row["derived_id"], "retired", reason="Owner disputed the source result", by="owner")
            await c.update("hands", "id", hand_id, decisions={**hand["decisions"], "disputed": True})
            await c.execute("UPDATE derivations SET state='downgraded' WHERE hand_id=? AND state='live'", (hand_id,))
            return {"disputed": True}
        return await self._mutate("/provenance/dispute", apply)
