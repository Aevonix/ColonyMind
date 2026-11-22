"""Bounded, source-checked owner offers. The mind tick owns the only synthesis.

The model selects exact quotations, a supported work family and display copy.
Copy is checked against the private bindings; connector offers may use verified fallbacks.
Connector originals live in the canonical source ledger; private payloads use
the existing vault. Nothing in this module records taste or engagement.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
import re
from typing import Any

from colonymind.mind.rank import Candidate, rank
from .ideas_copy import PROMPT, RETRY_PROMPT, fallback_copy, named, validate_copy
from .ideas_presentation import BUILD_STATUS, KINDS, presentation

MARKER = "grounded-ideas-v1"
COPY_VERSION = 2
MAX_CARDS = 6
MAX_SOURCES = 48
REFRESH_DEADLINE = 300.0
COPY_RETRIES = 2
# One synthesis must finish inside the role's per-call timeout on a shared lane. Beyond SPLIT_GROUPS
# source groups the evidence is split into SYNTHESIS_CALLS sequential calls of disjoint groups, each
# offering at most three; up to SPLIT_GROUPS groups one call can only write that many cards.
SYNTHESIS_CALLS = 2
SPLIT_GROUPS = 4


class CopyUnavailable(ValueError):
    """Verified evidence whose display copy failed; the reason code may drive a copy-only retry."""

    def __init__(self, reason):
        super().__init__("copy_unavailable")
        self.reason = reason
RETRY_AFTER = timedelta(minutes=15)
logger = logging.getLogger(__name__)
FAMILIES = {
    "packet": ("Compare the documents", "I’ll check amounts and terms against the originals.",
               "A cited comparison and discrepancy list.", "Prepare an unsent reply",
               "I’ll draft a reply using the checked details and flag any gaps.", "An unsent reply draft."),
    "renewal": ("Check the renewal terms", "I’ll verify the price and decision dates in the original email.",
                "A dated terms summary and open questions.", "Prepare an unsent reply",
                "I’ll draft a reply about the terms for your review.", "An unsent renewal reply draft."),
    "document": ("Check the form and email thread", "I’ll list what is required and flag missing information.",
                 "A form checklist with source references.", "Prepare an unsent follow-up",
                 "I’ll draft a follow-up using the checked requirements.", "An unsent follow-up draft."),
    "calendar": ("Assemble the meeting brief", "I’ll gather the material requested in the calendar entry.",
                 "A dated brief and preparation checklist.", "List questions for the meeting",
                 "I’ll turn gaps in the material into questions you can bring.", "An agenda of open questions."),
}


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


class AnswerUnavailable(ValueError):
    """No JSON answer in a reply; ``shape`` describes the reply without any of its text."""

    def __init__(self, key, text):
        super().__init__(f"{key}_unavailable")
        self.shape = {"chars": len(text), "first": text[:1] if text[:1] in "{[`<" else ("" if not text else "text"),
                      "fence": "```" in text, "think_tag": "</think>" in text or "<think>" in text}


def answer(text, key):
    """The reply's JSON object: the whole reply, or one object set in prose or a fence.

    A reasoning model whose thinking ran to its budget can wrap its answer. Anything ambiguous (an array,
    a second or truncated object, any bracket outside the object) is refused rather than guessed.
    """
    text = (text or "").strip()
    # Thinking is never the answer: only what follows the last closing tag counts, and an unclosed one is refused.
    shown = text[text.rfind("</think>") + len("</think>"):] if "</think>" in text else text
    if "<think>" in shown:
        raise AnswerUnavailable(key, text)
    text = shown.strip()
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        value, first, last = None, text.find("{"), text.rfind("}")
        if first != -1 and last > first and not any(c in "[]{}" for c in text[:first] + text[last + 1:]):
            try:
                value = json.loads(text[first:last + 1])
            except (ValueError, RecursionError):
                pass
    if isinstance(value, dict) and key in value:
        return value
    raise AnswerUnavailable(key, text)


def instant(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except ValueError:
        return None


def marked(row):
    return bool(row and getattr(row, "kind", None) == "task" and row.type == "grounded_offer"
                and row.decision in ("ask", "act")
                and isinstance(row.context, dict) and isinstance(row.context.get("idea"), dict)
                and row.context["idea"].get("generator") == MARKER)


def started(row):
    """An idea the mind started by itself (``Ideas.start``): its work is a task of the agent's own on the board."""
    return marked(row) and row.decision == "act"


# The build state a started idea's card shows, read from its row's status.
STARTED_STATES = {"approved": "accepted", "dispatched": "working", "done": "done", "failed": "failed",
                  "uncertain": "failed"}
# The done-check of a started idea: its report names its result (``outcomes.evaluate_check``).
RESULT_CHECK = {"kind": "result_field", "field": "result"}


class Ideas:
    def __init__(self, mind, *, vault=None, queue=None):
        self.mind = mind
        self.store = mind.store
        self.state = mind.mind_state
        self.owner = mind.owner_id
        self._vault = vault
        self.queue = queue
        if self.queue is None:
            from .ideas_build import HandsQueue
            self.queue = HandsQueue()
        self.job = None
        self.build_lock = asyncio.Lock()
        self.config = dict(mind.config.get("ideas") or {})

    @property
    def vault(self):
        if self._vault is None:
            from colonymind.secrets.config import current_settings
            from colonymind.secrets.vault import get_vault
            self._vault = get_vault(self.mind.state_dir, key_store=current_settings().key_store)
        return self._vault

    def seal(self, value):
        ref = self.vault.seal(dumps(value), contact_id=self.owner, kind="private_idea", label="Private owner offer")
        if not ref:
            raise ValueError("private_payload_unavailable")
        return ref

    def open(self, ref):
        return json.loads(self.vault.resolve(ref))

    def meta(self):
        return json.loads((self.state.get("ideas:refresh") or {}).get("text") or "{}")

    def save_meta(self, value):
        self.state.set("ideas:refresh", text=dumps(value), now=self.mind.clock())

    def sources(self):
        ref = (self.state.get("ideas:sources") or {}).get("text")
        return self.open(ref) if ref else []

    def trigger(self, now=None):
        now = now or self.mind.clock()
        meta = self.meta()
        # Trailing debounce. An active refresh records its own watermark; a
        # trigger arriving during it remains pending for the following tick.
        meta.update(pending_at=now.isoformat(), generation=meta.get("generation", 0) + 1)
        self.save_meta(meta)

    def collection(self):
        meta = self.meta()
        now = self.mind.clock()
        last = instant(meta.get("collected_at"))
        due = last is None or now - last >= timedelta(minutes=5)
        accounts = self.config.get("accounts", [])[:2]
        return {"due": due and self.mind.enabled and bool(accounts), "accounts": accounts}

    def ingest(self, observations, *, now=None):
        """Only the trusted body submits exact account-selected read results.

        No owner turn is fabricated: these are tool sources, with ingestion
        provenance and no assertion/commitment extraction or appraisal learning.
        """
        now = now or self.mind.clock()
        if not self.owner or self.mind.ledger is None:
            raise ValueError("owner_sources_unavailable")
        if len(observations) > 32 or len(dumps(observations).encode()) > 600_000:
            raise ValueError("source_batch_too_large")
        allowed = {(a["server"], a["account"]) for a in self.config.get("accounts", [])[:2]}
        old = self.sources()
        indexed = {s["key"]: s for s in old}
        changed = False
        for obs in observations:
            server, account, tool = obs["server"], obs["account"], obs["tool"]
            if (server, account) not in allowed or tool not in {"gmail_get", "calendar_events"}:
                raise ValueError("unselected_source")
            envelope = obs["result"]
            if envelope.get("account") != account or not isinstance(envelope.get("data"), dict):
                raise ValueError("account_mismatch")
            data = envelope["data"]
            records = data.get("items", [])[:16] if tool == "calendar_events" else [data]
            if tool == "calendar_events" and not data.get("continuation"):
                calendar = obs.get("calendar_id", "primary")
                found = {r.get("id") for r in records}
                for key, previous in list(indexed.items()):
                    if (previous["account"] == account and previous["server"] == server
                            and previous.get("calendar_id") == calendar and previous["provider_id"] not in found):
                        del indexed[key]
                        changed = True
            for record in records:
                provider_id = record.get("id")
                if not isinstance(provider_id, str) or not provider_id or len(provider_id) > 1024:
                    continue
                calendar = obs.get("calendar_id", "primary") if tool == "calendar_events" else None
                key = digest([server, account, tool, calendar, provider_id])
                if record.get("status") == "cancelled":
                    changed = indexed.pop(key, None) is not None or changed
                    continue
                original = dumps(record)
                if len(original.encode()) > 16_000:
                    continue  # do not certify truncated originals
                revision = digest(record)
                sid = "google-read:" + digest([key, revision])
                private = self.seal(record)
                messages = [{"role": "tool", "content": private,
                             "provenance": {"kind": MARKER, "server": server, "account": account,
                                            "tool": tool, "provider_id": provider_id, "digest": revision}}]
                self.mind.ledger.record_source(sid, contact_id=self.owner, session_id=MARKER,
                    messages=messages, scope="person", derive_claims=False)
                refs = self.mind.ledger.source_references([sid], contact_id=self.owner, session_id=MARKER)
                if not refs:
                    indexed.pop(key, None)  # erasure cannot be resurrected by collection
                    continue
                prior = indexed.get(key)
                source = {"key": key, "ref": refs[0], "private": private, "server": server, "account": account,
                          "tool": tool, "provider_id": provider_id, "calendar_id": calendar,
                          "action_id": record.get("threadId") or provider_id, "revision": revision,
                          "read_at": now.isoformat()}
                if calendar:
                    source["due_at"] = (record.get("start") or {}).get("dateTime") or (record.get("start") or {}).get("date")
                else:
                    # Conservative first slice: only explicit ISO deadlines,
                    # never a model's guessed year or local date interpretation.
                    text = " ".join(str(record.get(k, "")) for k in ("subject", "snippet"))
                    dates = re.findall(r"\b(?:by|before|on|closes|due)\s+(\d{4}-\d{2}-\d{2})\b", text, re.I)
                    if dates:
                        source["due_at"] = min(dates)
                indexed[key] = source
                changed = changed or prior is None or prior["revision"] != revision
        sources = sorted(indexed.values(), key=lambda s: s["read_at"], reverse=True)[:MAX_SOURCES]
        self.state.set("ideas:sources", text=self.seal(sources), now=now)
        meta = self.meta()
        meta["collected_at"] = now.isoformat()
        self.save_meta(meta)
        if changed:
            self.trigger(now)
        return {"retained": len(sources), "changed": changed}

    def evidence(self, now):
        sources = []
        for source in self.sources():
            if self.source_current(source, now):
                value = self.open(source["private"])
                sources.append({**source, **({"thread_version": self.thread_version(source)} if source["tool"] == "gmail_get" else {}),
                                "text": dumps(value), "value": value})
        commitments = self.mind.commitments
        if commitments is not None:
            for item in commitments.list(person_id=self.owner, status=["pending", "overdue"], limit=16)["commitments"]:
                # Canonical commitment revisions are re-read on publication/read/Build.
                sources.append({"key": "commitment:" + item["id"], "revision": digest(item),
                                "tool": "commitment", "provider_id": item["id"], "action_id": item["id"],
                                "account": "owner", "text": dumps(item), "value": item,
                                "read_at": now.isoformat(), "due_at": item.get("due_at")})
        return sources[:64]

    def thread_version(self, source):
        return digest(sorted((s["key"], s["revision"]) for s in self.sources()
                             if s["tool"] == "gmail_get" and s["account"] == source["account"]
                             and s["server"] == source["server"] and s["action_id"] == source["action_id"]))

    def source_current(self, source, now, *, fresh=True):
        from colonymind.memory import curated
        if source.get('ref') and curated.is_forgotten(self.mind.ledger, source['ref']):
            return False
        if source["tool"] == "commitment":
            if self.mind.commitments is None:
                return False
            rows = self.mind.commitments.list(person_id=self.owner, status=["pending", "overdue"], limit=100)["commitments"]
            return any(r["id"] == source["provider_id"] and digest(r) == source["revision"] for r in rows)
        read = instant(source.get("read_at"))
        if read is None or (fresh and not timedelta(0) <= now - read < timedelta(hours=24)):
            return False
        due = instant(source.get("due_at"))
        if fresh and due and due <= now:
            return False
        selected = {(a["server"], a["account"]) for a in self.config.get("accounts", [])[:2]}
        if (source["server"], source["account"]) not in selected:
            return False
        if source.get("thread_version") and source["thread_version"] != self.thread_version(source):
            return False
        if not any(s["key"] == source["key"] and s["revision"] == source["revision"] for s in self.sources()):
            return False
        refs = self.mind.ledger.source_references([source["ref"]["source_id"]], contact_id=self.owner, session_id=MARKER)
        return source["ref"] in refs

    def valid(self, row, *, now=None, accepted=False, current=None):
        if not marked(row) or row.entity_id != self.owner:
            return False
        idea = row.context["idea"]
        accepted = started(row) or (accepted and bool((idea.get("execution") or {}).get("taskId")))
        if idea.get("owner") != self.owner or row.status in {"cancelled", "expired", "dropped"}:
            return False
        now = now or self.mind.clock()
        # Accepted work has its own frozen receipt; stale source-derived copy
        # and results are never disclosed after correction or erasure.
        if not accepted and (row.expires_at is None or row.expires_at <= now):
            return False
        try:
            current = current or self.source_current  # a caller may share one per-request memo
            return all(current(s, now, fresh=not accepted) for s in self.open(idea["private"])["sources"])
        except Exception:
            return False

    def rows(self):
        return [r for r in self.store.intentions(types=["grounded_offer"], recipient=self.owner, limit=500) if marked(r)]

    def catalog(self):
        return sorted([r for r in self.rows() if not r.context["idea"].get("execution") and not started(r)
                       and self.valid(r)], key=lambda r: (-r.priority, r.id))[:MAX_CARDS]

    def card(self, row):
        payload = self.open(row.context["idea"]["private"])
        execution = row.context["idea"].get("execution") or {}
        state = execution.get("status") if execution.get("taskId") else None
        if started(row):
            state = STARTED_STATES.get(row.status)
        artwork = presentation(payload["family"], state)
        # Items show their icon and copy only: no labels, badges, action or "deliverable" type.
        items = [{**{k: v for k, v in item.items() if k != "type"},
                  "isSelected": item["id"] in execution["selected"] if execution else item["isSelected"],
                  "presentation": {"icon": artwork["icon"], "labels": [], "badges": []}} for item in payload["items"]]
        return {"id": row.id, "kind": payload["family"], "content": payload["content"], "items": items,
                "sharing": None, "presentation": artwork, "iconUrl": artwork["icon"]["variants"][0]["url"],
                # secondaryLabel only names the section; rows carry no kind label.
                "mimeType": "image/svg+xml", "secondaryLabel": KINDS[payload["family"]][1],
                # Unbuilt cards omit the build fields; the live hub rejects null text.
                **({"buildStatus": BUILD_STATUS.get(state), "build_status": BUILD_STATUS.get(state), "buildState": state}
                   if state is not None else {}),
                **({k: execution[k] for k in ("executionId", "taskId") if k in execution} if execution.get("taskId") else {}),
                **({"result": self.open(execution["result_ref"])} if execution.get("result_ref") else {}),
                # A started idea's result is what its task reported.
                **({"result": {"packet": row.result}} if started(row) and row.status == "done" and row.result else {})}

    def start(self, row, now):
        """Decide a passive card by the mind's authority (``Authority.decide``, on the worker's toolsets): work the
        level lets act starts as a task of the agent's own, queued for the board by the mind's decision path, with
        a done-check and the idea's own deadline as the window it must be picked up in. Any other verdict leaves the
        card passive, with the verdict's reason, for the owner's Build."""
        authority = getattr(self.mind, "authority", None)
        if authority is None:
            return None
        try:
            payload = self.open(row.context["idea"]["private"])
        except Exception:
            return None
        copy = payload["content"]
        verdict = authority.decide(kind="task", recipient=row.entity_id, type=row.type, now=now,
                                   text=f"{copy.get('title', '')}\n{copy.get('summary', '')}",
                                   toolsets=self.mind.policy.worker_toolsets)
        if verdict.decision != "act":
            reason = f"Passive card ({verdict.reason}); only an owner Build starts work"
            if row.decision_reason != reason:
                self.store.update(row.id, decision_reason=reason)
            return None
        self.store.update(row.id, description=FAMILIES[payload["family"]][0], success_check=RESULT_CHECK)
        updated = self.mind._apply_decision(self.store.get(row.id), verdict, now, reconsidered=False)
        return self.store.update(updated.id, expires_at=row.expires_at) if updated is not None else None

    def brief(self, row):
        """The body of a started idea's board task; None while its sources are not current (nothing is handed out)."""
        from .ideas_build import task_brief
        try:
            return task_brief(self.open(row.context["idea"]["private"])) if self.valid(row) else None
        except Exception:
            return None

    def lineage(self, row):
        """The ledger sources a started idea's memory entries are bound to, so erasing one erases them: () for any
        other row, None once the idea's sources are no longer current."""
        if not started(row):
            return ()
        try:
            if not self.valid(row):
                return None
            return [s["ref"]["source_id"] for s in self.open(row.context["idea"]["private"])["sources"] if s.get("ref")]
        except Exception:
            return None

    def validate_offer(self, offer, sources, now, *, copy_counts=None):
        if not isinstance(offer, dict) or offer.get("family") not in FAMILIES:
            raise ValueError("unsupported_family")
        ids = offer.get("sourceIds")
        by_id = {s["key"]: s for s in sources}
        if not isinstance(ids, list) or not 1 <= len(ids) <= 6 or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError("unsupported_sources")
        chosen = [by_id[i] for i in ids]
        commitment_only = all(s["tool"] == "commitment" for s in chosen)
        relevant = [s for s in sources if s in chosen or any(
            s["tool"] == c["tool"] == "gmail_get" and s["account"] == c["account"]
            and s.get("server") == c.get("server") and s["action_id"] == c["action_id"] for c in chosen)]
        if any(re.search(r"\b(?:no (?:action|review) (?:is )?needed|already (?:paid|returned|renewed)|invoice.{0,30}\bresolved)\b",
                         s["text"], re.I) for s in relevant):
            raise ValueError("resolved_source")
        anchor = offer.get("anchor", "")
        # Anchor must be an actual source subject, event summary or commitment
        # description. An arbitrary substring is insufficient for the title.
        anchors = {str(s["value"].get(k, "")) for s in chosen for k in ("subject", "summary", "description")}
        for s in chosen:
            for header in (s["value"].get("payload") or {}).get("headers", []):
                if str(header.get("name", "")).lower() == "subject":
                    anchors.add(header.get("value", ""))
        # A commitment description is the full verified subject, even when it
        # is longer than a mail subject. It must still match exactly.
        commitment_anchor = commitment_only and any(anchor == s["value"].get("description") for s in chosen)
        if (not isinstance(anchor, str) or not anchor.strip() or anchor not in anchors
                or (len(anchor) > 160 and not commitment_anchor)):
            raise ValueError("unsupported_anchor")
        facts = offer.get("facts", [])
        if not isinstance(facts, list) or not 1 <= len(facts) <= 6:
            raise ValueError("missing_facts")
        quotes = []
        for fact in facts:
            source = by_id.get(fact.get("sourceId"))
            quote = fact.get("quote")
            # Match decoded text leaves, not JSON syntax or escaped characters.
            def leaves(value):
                if isinstance(value, str):
                    return [value]
                if isinstance(value, dict):
                    return [s for v in value.values() for s in leaves(v)]
                if isinstance(value, list):
                    return [s for v in value for s in leaves(v)]
                return []
            if (source not in chosen or not isinstance(quote, str) or not 4 <= len(quote) <= 500
                    or not any(quote in text for text in leaves(source["value"]))):
                raise ValueError("unsupported_fact")
            quotes.append(quote)
        if not all(self.source_current(s, now) for s in chosen):
            raise ValueError("stale_sources")
        family = offer["family"]
        step, summary, result, optional, optional_summary, optional_result = FAMILIES[family]
        content = None
        reason = "copy_missing"
        if "content" in offer:
            if copy_counts is not None:
                copy_counts["supplied"] += 1
            try:
                # The canonical commitment deadline is evidence too, even when
                # the owner recorded it separately from the description. Do
                # not admit unrelated metadata or widen connector evidence.
                deadlines = [s["due_at"][:10] for s in chosen
                             if commitment_only and instant(s.get("due_at")) is not None]
                content = validate_copy(offer["content"], chosen, evidence=" ".join([anchor, *quotes, *deadlines]),
                                        source_texts=[anchor, *quotes])
            except ValueError as error:
                reason = str(error)
                if copy_counts is not None:
                    copy_counts["rejected"] += 1
        if content is None:
            if commitment_only:
                raise CopyUnavailable(reason)
            try:
                content = fallback_copy(family, anchor, quotes, chosen)
            except ValueError:
                raise CopyUnavailable(reason) from None
            if copy_counts is not None:
                copy_counts["fallback"] += 1
        items = [{"id": "evidence", "type": "deliverable", "selectable": False, "isSelected": True,
                  "content": {"title": step, "summary": summary,
                              "buildSummary": result}, "presentation": {"labels": [], "badges": []}},
                 {"id": "draft", "type": "deliverable", "selectable": True, "isSelected": True,
                  "content": {"title": optional, "summary": optional_summary,
                              "buildSummary": optional_result}, "presentation": {"labels": [], "badges": []}}]
        def is_anchor(source):
            v = source["value"]
            return anchor in [v.get("subject"), v.get("summary"), v.get("description")] or any(
                str(h.get("name", "")).lower() == "subject" and h.get("value") == anchor
                for h in (v.get("payload") or {}).get("headers", []))
        action_sources = [s for s in chosen if s["tool"] != "commitment"] or chosen
        primary = [s for s in action_sources if is_anchor(s)]
        identities = {(s["account"], s.get("server", ""), s["action_id"]) for s in primary}
        if len(identities) != 1:
            raise ValueError("ambiguous_action_identity")
        key = digest([self.owner, family, sorted(identities)])
        # A past commitment deadline leaves an open obligation actionable.
        # Connector-backed offers retain their existing deadline cutoff.
        expires = min([now + timedelta(hours=24)] + [d for s in chosen if (d := instant(s.get("due_at")))
                      and (not commitment_only or d > now)])
        if expires <= now:
            raise ValueError("deadline_passed")
        return dict(family=family, content=content, items=items,
                    quotes=[{"sourceId": fact["sourceId"], "quote": fact["quote"]} for fact in facts],
                    sources=[{k: v for k, v in s.items() if k not in {"text", "value"}} for s in chosen],
                    action_key=key, expires=expires.isoformat(), generated_at=now.isoformat())

    def publish(self, offers, sources, now, *, copy_failures=None):
        candidates = []
        copy_counts = dict(supplied=0, rejected=0, fallback=0)
        for offer in offers[:12]:
            try:
                payload = self.validate_offer(offer, sources, now, copy_counts=copy_counts)
            except CopyUnavailable as error:
                # Verified evidence with unusable copy gets a copy-only retry; nothing else does.
                if copy_failures is not None:
                    copy_failures.append((offer, error.reason))
                continue
            except (ValueError, TypeError, KeyError):
                continue
            due = [d for s in payload["sources"] if (d := instant(s.get("due_at")))]
            salience = max(0.65, 1 - min((min(due) - now).total_seconds(), 14 * 86400) / (40 * 86400)) if due else 0.65
            candidates.append(Candidate(type="grounded_offer", drive="duty", kind="task", title="Private owner offer",
                dedup_key=payload["action_key"], salience=salience,
                extra=payload))
        published = []
        for candidate, score in rank(candidates, feedback=getattr(self.mind, "feedback", None)):
            payload = candidate.extra
            previous = [r for r in self.rows() if r.context["idea"]["action_key"] == payload["action_key"]]
            for row in previous:
                if row.status == "proposed" and not self.valid(row, now=now):
                    self.store.transition(row.id, "expired", action="idea_expired", at=now)
            previous = [self.store.get(r.id) for r in previous]
            if any(r.status != "expired" for r in previous):
                continue  # dismissal, in-flight and completed history all suppress repeats
            revisions = digest([s["revision"] for s in payload["sources"]])
            key = f"{MARKER}:{payload['action_key']}:{revisions}:c{COPY_VERSION}"
            if self.store.get_by_dedup_key(key) is not None:
                continue
            active = self.catalog()
            weakest = min(active, key=lambda r: (r.priority, r.created_at, r.id)) if len(active) >= MAX_CARDS else None
            if weakest is not None and score <= weakest.priority:
                continue
            # Seal before superseding, so a vault failure cannot remove a card without its replacement.
            private = self.seal(payload)
            if weakest is not None:
                self.store.transition(weakest.id, "expired", action="idea_superseded", at=now)
            row, outcome = self.store.create_intention(kind="task", type="grounded_offer", title="Private owner offer",
                drive="duty", cls="owner", decision="ask", decision_reason="Passive card; only an owner Build starts work",
                status="proposed", dedup_key=key, recipient=self.owner, priority=score, created_at=now,
                expires_at=instant(payload["expires"]), context={"idea": {"generator": MARKER, "owner": self.owner,
                    "action_key": payload["action_key"], "copy": COPY_VERSION,
                    "revision": digest(payload), "private": private}})
            if outcome == "created":
                published.append(row.id)
        logger.info("Ideas copy counts: supplied=%d rejected=%d fallback=%d",
                    copy_counts["supplied"], copy_counts["rejected"], copy_counts["fallback"])
        return published

    def chunks(self, sources):
        """Disjoint source groups for the synthesis calls; messages of one mail thread stay together."""
        groups = {}
        for s in sources:
            key = ("thread", s["account"], s.get("server"), s["action_id"]) if s["tool"] == "gmail_get" else s["key"]
            groups.setdefault(key, []).append(s)
        if len(groups) <= SPLIT_GROUPS:
            return [sources]
        parts = [[] for _ in range(SYNTHESIS_CALLS)]
        for n, group in enumerate(groups.values()):
            parts[n % SYNTHESIS_CALLS] += group
        return parts

    async def refresh(self, now, generation):
        fingerprint = ""
        calls = 0
        published = []
        usage = {}
        failure = None
        retry = dict(copy_retries=0, copy_recovered=0, copy_dropped=0, copy_retry_errors=0)
        chunk_failures = 0
        shapes = []  # replies without a JSON answer, described without their text
        cost_row, cost_tokens = None, 0

        async def complete(messages, context, deadline):
            nonlocal cost_row, cost_tokens, calls
            from colonymind.usage import charged_tokens
            authority = getattr(self.mind, "authority", None)
            if authority is not None and not authority.tokens_allowed():
                raise RuntimeError("idea synthesis budget exhausted")
            if cost_row is None:
                cost_row, _ = self.store.create_intention(
                    kind="note", type="idea_synthesis", title="Idea synthesis", drive="upkeep",
                    cls="internal", decision="act", decision_reason="refresh", status="done",
                    dedup_key=None, hermes_kind="none", created_at=now)
            calls += 1
            charged = None
            try:
                charged = await asyncio.wait_for(self.mind.router.complete(messages=messages, context=context), deadline + 5)
                return charged
            except Exception as error:
                charged = error
                raise
            finally:
                cost_tokens += charged_tokens(charged)
                self.store.update(cost_row.id, cost_tokens=cost_tokens)

        try:
            sources = self.evidence(now)
            bounded, used = [], 0
            for source in sources:
                size = len(dumps(source["value"]).encode())
                if used + size <= 64_000:
                    bounded.append(source)
                    used += size
            sources = bounded
            fingerprint = digest([(s["key"], s["revision"]) for s in sources])
            if sources and self.mind.router is not None:
                deadline = self.mind.router.function_deadline_seconds(context={"task": "mind_deliberate"}) \
                    if hasattr(self.mind.router, "function_deadline_seconds") else None
                if (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)
                        or not 0 < deadline):
                    deadline = REFRESH_DEADLINE
                offers, error = [], None
                chunks = self.chunks(sources)
                per_call = "up to three" if len(chunks) > 1 else "up to six"
                for chunk in chunks:
                    try:
                        response = await complete(
                            messages=[{"role": "system", "content": named(PROMPT.replace("up to six", per_call))},
                                      {"role": "user", "content": dumps([{k: s[k] for k in ("key", "tool", "account", "value")}
                                                                         for s in chunk])}],
                            context={"task": "mind_deliberate", "max_output_tokens": 4000,
                                     "workload": "background"}, deadline=deadline)
                        usage = {"tokens": response.usage, "cost_usd": response.cost_usd, "model": response.model_id}
                        result = answer(response.content, "offers")
                        if not isinstance(result.get("offers"), list):
                            raise ValueError("invalid_batch")
                        # An offer may cite only the sources its own call saw; publish ranks across calls.
                        seen = {s["key"] for s in chunk}
                        offers += [o for o in result["offers"] if isinstance(o, dict) and isinstance(o.get("sourceIds"), list)
                                   and all(isinstance(i, str) and i in seen for i in o["sourceIds"])]
                    except asyncio.CancelledError:
                        raise
                    except Exception as chunk_error:
                        # The other groups still publish; a failed group is retried with the whole refresh later.
                        chunk_failures += 1
                        if isinstance(chunk_error, AnswerUnavailable):
                            shapes.append(chunk_error.shape)
                        error = chunk_error
                if error is not None and not offers:
                    raise error
                # Verification occurs after inference, against current revisions.
                if self.mind.enabled:
                    failures = []
                    published = self.publish(offers, sources, self.mind.clock(), copy_failures=failures)
                    while failures and retry["copy_retries"] < COPY_RETRIES:
                        retry["copy_retries"] += 1
                        try:
                            # Copy only: the offers keep their verified evidence and are revalidated in full.
                            response = await complete(
                                messages=[{"role": "system", "content": named(RETRY_PROMPT)},
                                          {"role": "user", "content": dumps([{"index": i, "family": offer.get("family"),
                                              "anchor": offer.get("anchor"), "facts": [f.get("quote") for f in offer.get("facts", [])
                                                                                       if isinstance(f, dict)],
                                              "rejected": reason} for i, (offer, reason) in enumerate(failures)])}],
                                context={"task": "mind_deliberate", "max_output_tokens": 4000,
                                         "workload": "background"}, deadline=deadline)
                            copies = answer(response.content, "copies").get("copies")
                            if not isinstance(copies, list):
                                raise ValueError("invalid_copies")
                        except asyncio.CancelledError:
                            raise
                        except Exception as retry_error:
                            retry["copy_retry_errors"] += 1
                            if isinstance(retry_error, AnswerUnavailable):
                                shapes.append(retry_error.shape)
                            break
                        answered = {c["index"]: c["content"] for c in copies
                                    if isinstance(c, dict) and type(c.get("index")) is int
                                    and 0 <= c["index"] < len(failures) and isinstance(c.get("content"), dict)}
                        rewritten = [{**failures[i][0], "content": content} for i, content in answered.items()]
                        # An offer the rewrite left out keeps failing and may be asked for again.
                        failures = [failures[i] for i in range(len(failures)) if i not in answered]
                        try:
                            recovered = self.publish(rewritten, sources, self.mind.clock(), copy_failures=failures)
                        except Exception:
                            retry["copy_retry_errors"] += 1
                            break
                        retry["copy_recovered"] += len(recovered)
                        published += recovered
                    retry["copy_dropped"] = len(failures)
        except asyncio.CancelledError as error:
            failure = "cancelled"
            raise
        except Exception as error:
            failure = type(error).__name__  # never log private model/source text
        finally:
            if failure is None and chunk_failures:
                failure = "partial"  # published what verified; the failed group retries after RETRY_AFTER
            meta = self.meta()
            failures = (int(meta.get("consecutive_failures", 0)) + 1 if failure not in (None, "cancelled") else 0)
            if failure != "cancelled":
                meta["consecutive_failures"] = failures
            day = now.astimezone(self.mind.tz).date().isoformat()
            days = meta.get("calls_per_day", {})
            days[day] = days.get(day, 0) + calls
            meta.update(calls_per_day=dict(sorted(days.items())[-32:]), last_refresh={"at": now.isoformat(),
                "model_calls": calls, "cost_tokens": cost_tokens, "input_watermark": fingerprint, "published": len(published), "failure": failure,
                "synthesis_failures": chunk_failures, "unparsed_replies": shapes, "finished_at": self.mind.clock().isoformat(), **retry, **usage})
            if failure != "cancelled" and meta.get("generation", 0) == generation:
                meta.pop("pending_at", None)
            self.save_meta(meta)
        return published

    async def tick(self, now, *, schedule=True):
        if not self.mind.enabled and self.job is not None and not self.job.done():
            self.job.cancel()
        if self.job is not None and self.job.done():
            try:
                self.job.result()
            except (Exception, asyncio.CancelledError):
                pass
            self.job = None
        retired = False
        for row in self.rows():
            if row.status == "proposed":
                if row.context["idea"].get("copy") != COPY_VERSION:
                    self.store.transition(row.id, "expired", action="idea_copy_retired", at=now)
                    retired = True
                elif not self.valid(row, now=now):
                    self.store.transition(row.id, "expired", action="idea_expired", at=now)
                elif schedule:
                    # New work, so only on the tick that forms work (the mind on, the body alive).
                    try:
                        self.start(row, now)
                    except Exception as error:
                        logger.warning("idea %s not started (%s)", row.id, type(error).__name__)
            elif row.status == "approved" and started(row) and not self.valid(row, now=now):
                # A started idea the board has not taken yet lapses with its sources, as a passive card does.
                self.mind.outcomes.record(row.id, status="expired", summary="the idea's sources are no longer current",
                                          by="mind", implicit_verdict=False)
        if retired:
            self.trigger(now)
        if self.queue is not None:
            await self.reconcile()
        if not schedule or not self.mind.enabled or not self.owner or self.job is not None:
            return
        # Commitment revisions trigger the same debounced synthesis as Google.
        commitments = self.mind.commitments
        if commitments is not None:
            watermark = digest(commitments.list(person_id=self.owner, status=["pending", "overdue"], limit=16)["commitments"])
            meta = self.meta()
            if meta.get("commitments") != watermark:
                meta["commitments"] = watermark
                self.save_meta(meta)
                if not retired:
                    self.trigger(now)
        meta = self.meta()
        pending = instant(meta.get("pending_at"))
        # A failed synthesis retries later; its inputs may never change again.
        last = meta.get("last_refresh") or {}
        # The cooldown runs from when the failed refresh finished; a long one must not rearm at once.
        failed_at = (instant(last.get("finished_at") or last.get("at"))
                     if last.get("failure") not in (None, "cancelled") else None)
        retry_seconds = RETRY_AFTER.total_seconds() * 2 ** max(0, int(meta.get("consecutive_failures", 1)) - 1)
        if pending is None and failed_at is not None and (now - failed_at).total_seconds() >= retry_seconds:
            self.trigger(now)
            meta = self.meta()
            pending = instant(meta.get("pending_at"))
        if pending is not None and now - pending >= timedelta(seconds=60):
            self.job = asyncio.create_task(self.refresh(now, meta.get("generation", 0)), name="mind-ideas-refresh")

    async def close(self):
        if self.job is not None:
            self.job.cancel()
            await asyncio.gather(self.job, return_exceptions=True)
            self.job = None

    async def build(self, row, mode, item_ids, request):
        from .ideas_build import build
        async with self.build_lock:
            return await build(self, row.id, mode, item_ids, request)

    async def reconcile(self):
        from .ideas_build import reconcile
        await reconcile(self)
