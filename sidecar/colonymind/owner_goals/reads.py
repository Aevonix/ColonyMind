"""Read-only Hermes boundary and ordered activity projection.

No scheduler imports, default profile discovery, HTTP routes or authority issuance.
The destination registry is a server dependency, not request data. Its snapshot
must resolve every delivery lane (including configured defaults and failures)
through account-scoped authority. Ordinary contact aliases cannot implement it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
import hashlib
import heapq
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Protocol

MAX_INTEGER = 9007199254740991
MAX_BYTES = 1048576
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})\Z")


class ReadError(Exception):
    def __init__(self, status: int, code: str, reason: str, *, exclusions=()):
        super().__init__(reason)
        self.status_code, self.code, self.reason = status, code, reason
        self.exclusions = tuple(exclusions)

    def response(self):
        # Internal reasons and authority evidence never cross the public boundary.
        return {"error": {"code": self.code}}


def unavailable(reason, **kwargs):
    return ReadError(503, "owner_goals_unavailable", reason, **kwargs)


def pagination(limit=50, offset=0):
    if (type(limit) is not int or not 1 <= limit <= 100 or
            type(offset) is not int or not 0 <= offset <= MAX_INTEGER):
        raise ReadError(422, "invalid_request", "pagination")
    return limit, offset


def parse_pagination(pairs):
    """Parse raw query pairs, preserving duplicate detection and ASCII syntax."""
    values = {"limit": 50, "offset": 0}
    seen = set()
    for key, value in pairs:
        if (key not in values or key in seen or not isinstance(value, str) or
                not re.fullmatch(r"[0-9]+", value) or len(value.lstrip("0")) > 16):
            raise ReadError(422, "invalid_request", "pagination")
        seen.add(key)
        values[key] = int(value.lstrip("0") or "0")
    pagination(**values)
    return values


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _reject_constant(value):
    raise ValueError("non_finite_json")


def route_version(job):
    """Pin routing inputs, never treat their strings as destination proof."""
    return digest({key: job.get(key) for key in ("deliver", "origin", "failure_deliver")})


@dataclass(frozen=True)
class Destination:
    platform: str
    account_id: str
    chat_id: str
    thread_id: str | None
    recipients: tuple[str, ...]
    direct: bool
    verified: bool


@dataclass(frozen=True)
class DeliveryEvidence:
    profile_id: str
    job_id: str
    route_version: str
    version: str
    destinations: tuple[Destination, ...]
    resolved: bool = True


class DestinationRegistry(Protocol):
    def snapshot(self, profile_id: str) -> Mapping[tuple[str, str], DeliveryEvidence]:
        """Immutable current authority keyed by (job ID, route version).

        Includes all effective lanes and proves there are no other recipients.
        Missing entries mean unresolved, never known non-owner. A resolved empty
        destination set proves disabled/missing delivery. Implementations must
        independently validate platform/account/chat/thread and owner identity.
        """


@dataclass(frozen=True)
class CronHandle:
    owner_id: str
    profile_id: str
    job_id: str
    object_version: str
    evidence: DeliveryEvidence
    name: str
    enabled: bool
    recurring: bool
    sensitivity: str = "Secret"


@dataclass(frozen=True)
class Exclusion:
    profile_id: str
    job_id: str
    reason: str
    unresolved: bool


@dataclass(frozen=True)
class CatalogSnapshot:
    binding: object
    owner_id: str
    handles: tuple[CronHandle, ...]
    exclusions: tuple[Exclusion, ...]


def _eligibility(owner_id, profile_id, job, evidence):
    if (not isinstance(evidence, DeliveryEvidence) or evidence.resolved is not True or
            evidence.profile_id != profile_id or evidence.job_id != job["id"] or
            evidence.route_version != route_version(job) or not evidence.version or
            not isinstance(evidence.destinations, tuple)):
        return "destination_unproven", True
    if not evidence.destinations:
        return "delivery_disabled", False
    for target in evidence.destinations:
        if (not isinstance(target, Destination) or target.verified is not True or
                not all(isinstance(v, str) and v for v in
                        (target.platform, target.account_id, target.chat_id)) or
                not isinstance(target.recipients, tuple) or
                (target.thread_id is not None and
                 (not isinstance(target.thread_id, str) or not target.thread_id))):
            return "destination_unproven", True
        if target.direct is not True or len(target.recipients) != 1:
            return "destination_ambiguous", True
    if any(target.recipients != (owner_id,) for target in evidence.destinations):
        return "non_owner_destination", False
    return None, False


class HermesOwnerReader:
    """One explicitly bound profile. Registry and paths are server configured.

    Catalog snapshots are ephemeral read snapshots, not the migration manifest.
    No default registry exists: profile binding is not owner attribution.
    """
    def __init__(self, home: Path, profile_id: str, registry: DestinationRegistry):
        if not home.is_absolute() or not profile_id or registry is None:
            raise ValueError("explicit_profile_and_registry_required")
        self.home, self.profile_id, self.registry = home.resolve(), profile_id, registry
        self._binding = object()

    def _jobs(self):
        try:
            # One atomic Hermes jobs document, no native load_jobs() self-healing.
            path = self.home / "cron" / "jobs.json"
            if not path.resolve().is_relative_to(self.home):
                raise ValueError
            document = json.loads(path.read_bytes(), parse_constant=_reject_constant)
            jobs = document["jobs"]
            if not isinstance(jobs, list):
                raise ValueError
            ids = [job["id"] for job in jobs]
            if (any(not isinstance(i, str) or not _ID.fullmatch(i) for i in ids) or
                    len(set(ids)) != len(ids)):
                raise ValueError
            return jobs
        except (OSError, ValueError, TypeError, KeyError):
            raise unavailable("cron_catalog_unreadable") from None

    def catalog_snapshot(self, owner_id):
        if not isinstance(owner_id, str) or not owner_id:
            raise unavailable("owner_context_missing")
        jobs = self._jobs()
        try:
            authority = MappingProxyType(dict(self.registry.snapshot(self.profile_id)))
        except Exception:
            raise unavailable("destination_registry_unavailable") from None
        handles, exclusions = [], []
        for job in sorted(jobs, key=lambda j: j["id"]):
            evidence = authority.get((job["id"], route_version(job)))
            reason, unresolved = _eligibility(owner_id, self.profile_id, job, evidence)
            if reason:
                exclusions.append(Exclusion(self.profile_id, job["id"], reason, unresolved))
                continue
            schedule = job.get("schedule")
            repeat = job.get("repeat")
            if (type(job.get("enabled")) is not bool or not isinstance(schedule, dict) or
                    schedule.get("kind") not in {"once", "cron", "interval"} or
                    not isinstance(repeat, dict) or "times" not in repeat or
                    (repeat.get("times") is not None and
                     (type(repeat["times"]) is not int or repeat["times"] < 1)) or
                    not isinstance(job.get("name"), str) or not job["name"].strip()):
                raise unavailable("cron_facts_missing")
            recurring = schedule["kind"] != "once" and repeat["times"] != 1
            handles.append(CronHandle(owner_id, self.profile_id, job["id"], digest(job),
                                      evidence, job["name"], job["enabled"], recurring))
        return CatalogSnapshot(self._binding, owner_id, tuple(handles), tuple(exclusions))

    def list_owner_cron_jobs(self, owner_id, limit=50, offset=0, snapshot=None):
        pagination(limit, offset)
        snapshot = snapshot or self.catalog_snapshot(owner_id)
        if snapshot.binding is not self._binding or snapshot.owner_id != owner_id:
            raise unavailable("catalog_snapshot_mismatch")
        # No successful exhaustion while potentially eligible jobs remain unknown.
        if any(item.unresolved for item in snapshot.exclusions):
            raise unavailable("cron_eligibility_unresolved", exclusions=snapshot.exclusions)
        rows = []
        for handle in snapshot.handles[offset:offset + limit]:
            # Reserve four bytes for terminal null, or more for a longer offset.
            candidate = {"items": [asdict(h) for h in rows + [handle]], "limit": limit,
                         "offset": offset, "next_offset": max(1000, offset + len(rows) + 1)}
            if len(json.dumps(candidate, ensure_ascii=True).encode()) > MAX_BYTES:
                if not rows:
                    raise unavailable("cron_handle_exceeds_byte_budget")
                break
            rows.append(handle)
        return {"items": list(rows), "limit": limit, "offset": offset,
                "next_offset": offset + len(rows) if offset + len(rows) < len(snapshot.handles) else None}

    def validate_handle(self, owner_id, handle, audience):
        # Audience is authenticated server context. Owner goals default to Secret;
        # this native reader has no declassification authority.
        if audience != owner_id or handle.owner_id != owner_id or handle.profile_id != self.profile_id:
            raise unavailable("owner_audience_mismatch")
        snapshot = self.catalog_snapshot(owner_id)
        fresh = next((h for h in snapshot.handles if h.job_id == handle.job_id), None)
        if fresh != handle:
            raise unavailable("cron_binding_changed")
        return fresh

    def list_owner_job_runs(self, owner_id, verified_handle, audience, limit=50, offset=0):
        pagination(limit, offset)
        self.validate_handle(owner_id, verified_handle, audience)
        # Reviewed native ledger: execution ID/outcome, no job/destination snapshot
        # or exact output reference; terminal executions and output files prune.
        # Delivery queue snapshots/content are erased at terminalization. Neither
        # timestamp filenames nor today's routing can authorize historical output.
        # Even an empty ledger cannot establish that no earlier runs were pruned.
        raise unavailable("native_run_history_evidence_unavailable")


def source_id(kind, *identity):
    """Stable namespace including profile/job/execution, bounded below SourceID limits."""
    return kind + ":" + digest(list(identity))


def _text(value, maximum, *, blank=True, multiline=False):
    return (isinstance(value, str) and len(value) <= maximum and (blank or value.strip()) and
            all(not (0xD800 <= ord(c) <= 0xDFFF or ord(c) < 32 or 127 <= ord(c) <= 159)
                or (multiline and c in "\n\t") for c in value))


def _validate_activity(item):
    fields = {"id", "source_id", "kind", "at", "title", "summary", "artifact_ref", "sensitivity"}
    if (not isinstance(item, dict) or set(item) != fields or
            not isinstance(item["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", item["id"]) or
            not _text(item["source_id"], 256, blank=False) or
            not _text(item["title"], 400, blank=False) or
            not _text(item["summary"], 1000, multiline=True) or
            item["sensitivity"] not in {"Public", "Everyday", "Personal", "Private", "Secret"} or
            item["kind"] not in {"cron_run", "progress"}):
        raise unavailable("activity_projection_invalid")
    ref = item["artifact_ref"]
    if ((item["kind"] == "progress" and ref is not None) or
            (item["kind"] == "cron_run" and (not isinstance(ref, str) or not _ID.fullmatch(ref)))):
        raise unavailable("activity_projection_invalid")


def _order(item):
    _validate_activity(item)
    value = item["at"]
    if not isinstance(value, str) or len(value) > 64 or not _TIME.fullmatch(value):
        raise unavailable("activity_timestamp_invalid")
    if int(value[17:19]) > 59 or (value[-1] != "Z" and
            (int(value[-5:-3]) > 23 or int(value[-2:]) > 59)):
        raise unavailable("activity_timestamp_invalid")
    try:
        at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Integer day/second ordering avoids floating point loss for close runs.
        seconds = ((at.toordinal() * 86400 + at.hour * 3600 + at.minute * 60 + at.second)
                   - int(at.utcoffset().total_seconds()))
        fraction = value[19:].split("Z")[0].split("+")[0].split("-")[0]
        instant = seconds + (Fraction("0" + fraction) if fraction else 0)
    except (ValueError, OverflowError):
        raise unavailable("activity_timestamp_invalid") from None
    identity = item["source_id"]
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 256:
        raise unavailable("activity_source_invalid")
    return -instant, identity


PageReader = Callable[..., dict]


def _stream(reader: PageReader, batch):
    offset, previous = 0, None
    while True:
        try:
            page = reader(limit=batch, offset=offset)
        except Exception:
            raise unavailable("activity_source_unavailable") from None
        try:
            items, next_offset = page["items"], page["next_offset"]
            if (type(page["limit"]) is not int or type(page["offset"]) is not int or
                    page["limit"] != batch or page["offset"] != offset or
                    not isinstance(items, list) or len(items) > batch or
                    (next_offset is not None and (type(next_offset) is not int or
                     not items or next_offset != offset + len(items) or next_offset > MAX_INTEGER))):
                raise ValueError
            for item in items:
                order = _order(item)
                if previous is not None and order < previous:
                    raise ValueError
                previous = order
                yield item
            if next_offset is None:
                return
            offset = next_offset
        except ReadError:
            raise
        except Exception:
            raise unavailable("activity_source_unavailable") from None


def merge_activity(readers: list[PageReader], limit=50, offset=0):
    """Bounded k-way merge of already authorized activity source pages.

    Readers project original progress/runs into Activity *after* audience checks.
    Every source starts at zero. Buffer at most one page per source plus the result;
    only stable source identities are retained for cross-source deduplication.
    Native run sources must go through list_owner_job_runs on every request.
    """
    pagination(limit, offset)
    streams = [iter(_stream(reader, min(limit + 1, 100))) for reader in readers]
    heap = []
    for index, stream in enumerate(streams):
        item = next(stream, None)
        if item is not None:
            heapq.heappush(heap, (_order(item), index, item))
    seen, skipped, result = set(), 0, []
    while heap:
        _, index, item = heapq.heappop(heap)
        identity = item["source_id"]
        if identity not in seen:
            seen.add(identity)
            if skipped < offset:
                skipped += 1
            else:
                # Reserve four bytes for terminal null, or more for a longer offset.
                candidate = {"items": result + [item], "limit": limit,
                             "offset": offset, "next_offset": max(1000, offset + len(result) + 1)}
                if len(result) == limit or len(json.dumps(candidate, ensure_ascii=True).encode()) > MAX_BYTES:
                    if not result:
                        raise unavailable("activity_item_exceeds_byte_budget")
                    return {"items": result, "limit": limit, "offset": offset,
                            "next_offset": offset + len(result)}
                result.append(item)
        following = next(streams[index], None)
        if following is not None:
            heapq.heappush(heap, (_order(following), index, following))
    return {"items": result, "limit": limit, "offset": offset, "next_offset": None}


def list_owner_activity(reader, owner_id, handles, audience, progress_reader,
                        limit=50, offset=0):
    """Current attached handles only; detaching removes that source on next read."""
    pagination(limit, offset)
    if audience != owner_id:
        raise unavailable("owner_audience_mismatch")
    sources = [progress_reader]
    for handle in handles:
        reader.validate_handle(owner_id, handle, audience)
        def runs(*, limit, offset, handle=handle):
            return reader.list_owner_job_runs(owner_id, handle, audience, limit, offset)
        sources.append(runs)
    return merge_activity(sources, limit, offset)
