"""Capture shared producers on their loop; hydrate only detached ledger evidence off-loop."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import time

from .builder import CardInputs

# Observation devices and automation must never become direct conversations.
DIRECT_PLATFORMS = ("voice", "companion", "telegram", "whatsapp", "sms", "rcs", "discord")


@dataclass(frozen=True)
class Obligation:
    source: str
    source_type: str
    status: str
    description: str
    due_at: str


@dataclass(frozen=True)
class CommsSnapshot:
    """Bounded locator metadata, detached from the live communications store."""
    rows: tuple[tuple[str, str, str], ...]

    def read_connection(self):
        # The canonical reader's compatibility queries run on a thread-local
        # memory DB. No live communications connection crosses the loop boundary.
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE communications(contact_id TEXT, channel TEXT, source_lineage_json TEXT)")
        conn.executemany("INSERT INTO communications VALUES (?,?,?)", self.rows)
        return conn


@dataclass(frozen=True)
class PreparedInputs:
    inputs: CardInputs
    obligations: tuple[Obligation, ...]
    digest_created_at: str
    comms: CommsSnapshot | None


class InputCollector:
    def __init__(self, state_dir: Path, *, host, mind_provider, owner_provider, name_provider, day_provider):
        self.state_dir = state_dir
        self.host = host
        self.mind_provider = mind_provider
        self.owner_provider = owner_provider
        self.name_provider = name_provider
        self.day_provider = day_provider

    def prepare(self) -> PreparedInputs:
        """Small synchronous capture on the event loop that owns these producers."""
        from colonymind.mind.digest import TITLE
        from colonymind.memory.recent import LOCATOR_LIMIT

        owner = self.owner_provider()
        if not owner:
            raise ValueError("context_card_owner_unconfigured")
        refs, unavailable, work, obligations = [], [], [], []
        preferences = narrative = digest = digest_created_at = ""
        learner = getattr(self.host, "_preference_learner", None)
        if learner is not None:
            preferences = learner.build_brief(source_ids=refs)
            perspective = getattr(learner, "perspective", None)
            if perspective is not None:
                preferences += "\n" + perspective.brief(source_ids=refs)
        else:
            unavailable.append("preferences")

        mind = self.mind_provider()
        if mind is not None and mind.owner_id == owner:
            view = mind.narrative()
            narrative = view.get("text", "") if view.get("enabled") else ""
            refs.extend(str(ref) for ref in view.get("cites", ()))
            section = mind.section()
            if section:
                work.append(section)
        else:
            unavailable.append("mind")

        commitments = getattr(self.host, "_commitment_store", None)
        if commitments is not None:
            for item in commitments.list(person_id=owner, status=["pending", "overdue"], limit=3)["commitments"]:
                obligations.append(Obligation(str((item.get("metadata") or {}).get("source_turn") or ""),
                    str(item.get("source_type") or ""), str(item["status"]), str(item["description"]),
                    str(item.get("due_at") or "")))

        store = getattr(self.host, "_initiative_store", None)
        if store is not None:
            rows = store.intentions(recipient=owner, types=["digest"], status=["sent"], limit=1)
            if rows and rows[0].status == "sent":
                digest = str((rows[0].context or {}).get("text") or "")
                digest_created_at = str(rows[0].created_at or "")
                if not digest.startswith(TITLE):
                    digest = ""

        comms, log = None, getattr(self.host, "_comms_log", None)
        if log is not None:
            # Preserve legacy source-linked locators, never communications bodies.
            # Capture both the reader's latest eligible links and a future marker.
            captured, as_of = [], time.time()
            with closing(log.read_connection()) as conn:
                for platform in DIRECT_PLATFORMS:
                    where = """contact_id=? AND source_lineage_json IS NOT NULL
                        AND (channel=? OR substr(channel,1,?)=?)"""
                    args = (owner, platform, len(platform) + 1, platform + ":")
                    at = "julianday(json_extract(source_lineage_json,'$.occurred_at'))"
                    rows = conn.execute(f"""SELECT contact_id,channel,source_lineage_json FROM communications
                        WHERE {where} AND ({at} IS NULL OR {at}<=julianday(?,'unixepoch'))
                        GROUP BY channel,source_lineage_json ORDER BY {at} DESC LIMIT ?""",
                        (*args, as_of, LOCATOR_LIMIT + 1)).fetchall()
                    future = conn.execute(f"""SELECT contact_id,channel,source_lineage_json FROM communications
                        WHERE {where} AND {at}>julianday(?,'unixepoch') LIMIT 1""", (*args, as_of)).fetchall()
                    captured.extend(tuple(row) for row in [*rows, *future])
            comms = CommsSnapshot(tuple(captured))
        return PreparedInputs(CardInputs(owner, self.name_provider(), self.day_provider(), preferences=preferences,
            narrative=narrative, work=tuple(work), digest=digest, source_refs=tuple(dict.fromkeys(refs)),
            unavailable=tuple(unavailable)), tuple(obligations), digest_created_at, comms)

    def collect(self) -> CardInputs:
        """Synchronous entry point for offline callers, on the producer-owning thread."""
        return self.collect_ledger(self.prepare())

    def collect_ledger(self, prepared: PreparedInputs) -> CardInputs:
        """Only detached data and ledger connections opened by this thread."""
        from colonymind.turns import get_turn_idempotency_ledger
        from colonymind.memory import curated
        ledger = get_turn_idempotency_ledger(self.state_dir)
        expected = []
        def read():
            watermark = curated.watermark(ledger, prepared.inputs.owner_id)
            if expected and expected[0] != watermark:
                raise ValueError("context_card_erasure_during_collection")
            expected[:] = [watermark]
            return self._read_collect_ledger(prepared)
        result = curated.fenced_read(ledger, read, empty=None)
        if result is None:
            raise ValueError("context_card_erasure_during_collection")
        return result

    def _read_collect_ledger(self, prepared):
        from colonymind.turns import get_turn_idempotency_ledger
        from colonymind.memory.recent import read_recent
        from colonymind.reports.instance import parse_time

        inputs = prepared.inputs
        owner = inputs.owner_id
        ledger = get_turn_idempotency_ledger(self.state_dir)
        from colonymind.memory import curated
        watermark = curated.watermark(ledger, owner)
        refs, highlights, work = list(inputs.source_refs), [], list(inputs.work)
        for item in prepared.obligations:
            if item.source:
                live = ledger.source_references([item.source], contact_id=owner, session_id="context-card")
                with closing(ledger._connect()) as conn:
                    erased = conn.execute("""SELECT 1 FROM source_erasures e
                        LEFT JOIN source_erasure_revisions r USING(sequence)
                        WHERE e.contact_id=? AND coalesce(r.source_turn_id,e.turn_id)=? LIMIT 1""",
                        (owner, item.source)).fetchone()
                from colonymind.memory import curated
                if not live or erased or curated.is_forgotten(ledger, item.source):
                    continue
                refs.append(item.source)
            elif item.source_type != "manual":
                continue
            work.append(f"Open obligation ({item.status}): {item.description}"
                        + (f"; due {item.due_at}" if item.due_at else ""))

        digest = inputs.digest
        if digest:
            with closing(ledger._connect()) as conn:
                erased_at = conn.execute("SELECT erased_at FROM source_erasures WHERE contact_id=? "
                                         "ORDER BY sequence DESC LIMIT 1", (owner,)).fetchone()
                curated_at = conn.execute("SELECT max(applied_at) FROM curated_memory_events WHERE person_id=? AND operation='forget'", (owner,)).fetchone()[0]
            erased_at = max((stamp for stamp in (parse_time(erased_at[0]) if erased_at else None,
                                                 parse_time(curated_at) if curated_at else None) if stamp), default=None)
            created = parse_time(prepared.digest_created_at)
            if created is None or (erased_at and created <= erased_at):
                digest = ""

        since = datetime.fromisoformat(inputs.local_day) - timedelta(days=1)
        for platform in DIRECT_PLATFORMS:
            packet = read_recent(ledger, contact_id=owner, session_id="context-card",
                                 platform=platform, limit=1, comms_log=prepared.comms)
            for entry in packet["entries"]:
                if str(entry["occurred_at"])[:10] < since.date().isoformat():
                    continue
                with closing(ledger._connect()) as conn:
                    device = conn.execute("""SELECT 1 FROM turn_sources s, json_each(s.messages_json) m
                        WHERE s.turn_id=? AND (json_extract(m.value,'$._observed_conversation') IS NOT NULL
                        OR EXISTS (SELECT 1 FROM json_each(CASE WHEN json_type(m.value,'$.content')='array'
                                   THEN json_extract(m.value,'$.content') ELSE '[]' END) b
                                   WHERE json_extract(b.value,'$.type')='observed_conversation')) LIMIT 1""",
                        (entry["source_id"],)).fetchone()
                if device:
                    continue
                highlights.append(f"{platform}, {str(entry['occurred_at'])[:10]}: {entry['content']}")
                refs.extend(ref["source_id"] for ref in packet["source_refs"])
        if curated.watermark(ledger, owner) != watermark:
            raise ValueError("context_card_erasure_during_collection")
        return replace(inputs, work=tuple(work), highlights=tuple(highlights), digest=digest,
                       source_refs=tuple(dict.fromkeys(refs)), erasure_watermark=watermark)
