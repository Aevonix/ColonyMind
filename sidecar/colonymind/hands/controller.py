"""Durable task transitions and replay, serialized by SQLite before admission.

The public functions keep the signatures the routes call. Controller instances also
provide an injectable clock and isolated database for executors and tests.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import secrets

from fastapi import HTTPException

from colonymind.config import DEFAULTS, config_path, load_config
from colonymind.redact import redact_sensitive_text
from colonymind.removed_settings import ignore_removed_settings
from colonymind.util.temporal import now_utc
from . import admission, budgets, cascade, contracts, locks, matrix, wire
from .health import EngineHealth
from .brief import BriefTooLarge, build_brief
from .evidence import evaluate_result, acceptance_outcome
from .types import Evidence, EvidenceScope
from .store import DB_FILE, HandsStore
from .obligations import ObligationOperations
from .notices import NoticeOperations
from .effects import EffectOperations
from .digest import render_digest
from .retention import RetentionOperations
from .settings import validate_limits
from .definitions import DefinitionOperations, definition_context


JSON_COLUMNS = frozenset("origin_ref needs intents resources depends_on budget usage owner_spoke_since contract families_used model epoch evidence check budgets decisions waiters owner_ref communicated_ref call targets owner_answer_ref payload response".split())
TERMINAL = frozenset("done below_minimum failed stopped superseded lost unknown_effect refused expired".split())
logger = logging.getLogger(__name__)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def plain(value):
    return value.model_dump(mode="json", by_alias=False) if hasattr(value, "model_dump") else value


def conflict(reason):
    raise HTTPException(status_code=409, detail=reason)


def merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


class Controller(DefinitionOperations, ObligationOperations, NoticeOperations, EffectOperations, RetentionOperations):
    def __init__(self, path, *, config=None, clock=None, enabled=True, faculty=None):
        self.store = HandsStore(path)
        clean = ignore_removed_settings({"hands": deepcopy(config or {})})["hands"]
        self.config = validate_limits(merge(deepcopy(DEFAULTS["hands"]), clean))
        self.clock = clock or now_utc
        self.health = EngineHealth(self.config["health"], self.clock)
        self._enabled = enabled
        # Design 8.2: turning the faculty off is a config change with no restart. A reader re-checks
        # ``mind.faculties.hands`` on each plan and sweep; without one the value given here holds.
        self._faculty = faculty
        self._lock = asyncio.Lock()
        self._writer = None
        self._maintenance = None
        self.evidence_evaluator = evaluate_result
        self.ledger = None
        self.owner_id = None
        self.recall = None
        self.parent_is_owner = lambda session: False

    @property
    def enabled(self):
        return self._enabled if self._faculty is None else self._faculty()

    @enabled.setter
    def enabled(self, value):
        self._enabled, self._faculty = value, None

    async def connect(self):
        async with self._lock:
            await self.store.connect()
            await self.publish_changes()
        await self._refresh_work()

    async def _refresh_work(self):
        from colonymind.work.hands import refresh
        try:
            await asyncio.to_thread(refresh, self.store.path)
        except Exception:
            logger.exception("Hands work refresh pending after committed transition")

    async def close(self):
        if self._maintenance:
            self._maintenance.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance
            self._maintenance = None
        async with self._lock:
            await self.store.close()

    def start_maintenance(self):
        if self._maintenance is None:
            self._maintenance = asyncio.create_task(self._maintain())

    async def _maintain(self):
        while True:
            try:
                await self.health.refresh()
                await self.sweep()
                from .runtime import expire_branches
                await expire_branches(self)
            except Exception:
                logger.exception("Hands maintenance failed; admission remains guarded")
            await asyncio.sleep(self.config["health"]["poll_seconds"])

    def now(self):
        return self.clock().astimezone(timezone.utc).isoformat()

    def day(self):
        # Local midnight is the notice and initiative reset boundary. Hosts can
        # supply an IANA timezone; the process timezone is the default.
        from zoneinfo import ZoneInfo
        zone = self.config.get("timezone")
        return self.clock().astimezone(ZoneInfo(zone) if zone else None).date().isoformat()

    @staticmethod
    def parse_time(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    async def execute(self, sql, args=()):
        cursor = await self.store.db.execute(sql, args)
        await cursor.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        for key in JSON_COLUMNS & result.keys():
            if result[key] is not None:
                try:
                    result[key] = json.loads(result[key])
                except (TypeError, ValueError):
                    pass
        return result

    async def one(self, sql, args=()):
        if self._writer is not asyncio.current_task():
            async with self._lock:
                async with self.store.db.execute(sql, args) as cursor:
                    return self.decode(await cursor.fetchone())
        async with self.store.db.execute(sql, args) as cursor:
            return self.decode(await cursor.fetchone())

    async def all(self, sql, args=()):
        if self._writer is not asyncio.current_task():
            async with self._lock:
                async with self.store.db.execute(sql, args) as cursor:
                    return [self.decode(row) for row in await cursor.fetchall()]
        async with self.store.db.execute(sql, args) as cursor:
            return [self.decode(row) for row in await cursor.fetchall()]

    async def insert(self, table, **values):
        # Identifiers are module constants, never wire inputs.
        columns = ','.join('"' + name + '"' for name in values)
        args = [encode(value) if (name == "response" or isinstance(value, (dict, list))) else value for name, value in values.items()]
        await self.execute(f'INSERT INTO {table} ({columns}) VALUES ({",".join("?" for _ in values)})', args)

    async def update(self, table, column, ident, **values):
        args = [encode(value) if (name == "response" or isinstance(value, (dict, list))) else value for name, value in values.items()]
        setters = ','.join('"' + name + '"=?' for name in values)
        await self.execute(f'UPDATE {table} SET {setters} WHERE "{column}"=?', (*args, ident))

    async def write(self, key, operation, body, callback, *, record=True, strict_body=False, skip_unchanged=False):
        """One serialized transaction. ``record=False`` is for internal
        maintenance (the sweeper), which has no caller to replay and must not
        add an idempotency row on every pass."""
        if record and (not isinstance(key, str) or not key.strip()):
            raise HTTPException(status_code=422, detail="Idempotency-Key required")
        async with self._lock:
            self._writer = asyncio.current_task()
            try:
                await self.execute("BEGIN IMMEDIATE")
                if not record:
                    before = self.store.db.total_changes
                    response = plain(await callback())
                    if skip_unchanged and self.store.db.total_changes == before:
                        await self.store.db.commit()
                        return response
                    await self.prepare_changes()
                    await self.store.db.commit()
                    await self.publish_changes()
                    await self._refresh_work()
                    return response
                prior = await self.one("SELECT * FROM idempotency WHERE key=?", (key,))
                if prior:
                    if prior["path"] != operation:
                        conflict("idempotency_scope")
                    if prior["response"] == {"erased": True}:
                        raise HTTPException(status_code=410, detail="parent_erased")
                    if prior["response"] == {"expired": True}:
                        raise HTTPException(status_code=410, detail="hands_replay_expired")
                    if strict_body and prior["request_hash"] != sha256(encode(plain(body)).encode()).hexdigest():
                        conflict("idempotency_key_reuse")
                    await self.store.db.commit()
                    await self.publish_changes()
                    await self._refresh_work()
                    return prior["response"]
                response = plain(await callback())
                await self.insert("idempotency", key=key, method="POST", path=operation,
                                  request_hash=sha256(encode(plain(body)).encode()).hexdigest(),
                                  response=response, status_code=200, created_at=self.now())
                await self.prepare_changes()
                await self.store.db.commit()
                await self.publish_changes()
                await self._refresh_work()
                return response
            except BaseException:
                await self.store.db.rollback()
                raise
            finally:
                self._writer = None

    async def require(self, table, ident):
        row = await self.one(f"SELECT * FROM {table} WHERE id=?", (ident,))
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown_{table}")
        return row

    def parent_forgotten(self, hand):
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_parent_forgotten(hand),
                                  empty=True)

    def _read_parent_forgotten(self, hand):
        from colonymind.memory import curated
        from contextlib import closing
        if self.ledger is None:
            return False
        with closing(self.ledger._connect()) as conn:
            return any(curated.is_forgotten(self.ledger, row[0], conn=conn) for row in conn.execute(
                'SELECT turn_id FROM turn_sources WHERE contact_id=? AND session_id=?',
                (self.owner_id, hand['parent_session'])))

    async def guard(self, ident, epoch=None, states=None):
        row = await self.require("hands", ident)
        if row["decisions"].get("erased") or self.parent_forgotten(row):
            raise HTTPException(status_code=410, detail="parent_erased")
        if epoch is not None and row["epoch"] != plain(epoch):
            logger.warning("Hands stale_epoch refused for %s", ident)
            conflict("stale_epoch")
        if states is not None and row["state"] not in states:
            conflict("prior_state")
        return row

    async def log(self, hand, kind, payload=None, *, key=None, source="controller", seq=None, at=None):
        def scrub(value):
            if isinstance(value, str):
                return redact_sensitive_text(value)[:280]
            if isinstance(value, dict):
                return {redact_sensitive_text(str(k))[:280]: scrub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [scrub(v) for v in value]
            return value
        payload = scrub(payload or {})
        if len(encode(payload).encode()) > 2048:
            raise HTTPException(status_code=422, detail="event_payload_limit")
        if seq is None:
            # Controller events use odd sequence numbers; plugin sequences are
            # mapped to even numbers so their independent cursors cannot collide.
            row = await self.one("SELECT coalesce(max(seq),-1) AS n FROM events WHERE hand_id=?", (hand,))
            seq = row["n"] + (2 if row["n"] % 2 else 1)
        if key and await self.one("SELECT key FROM events WHERE key=?", (key,)):
            return
        await self.insert("events", hand_id=hand, seq=seq, at=at or self.now(), type=kind,
                          payload=payload, source=source, key=key or f"ctl:{hand}:{seq}")
        await self.update("hands", "id", hand, last_event_at=self.now())

    async def get_hand(self, ident):
        return await self.require("hands", ident)

    async def get_task(self, ident):
        return await self.require("tasks", ident)

    async def list_hands(self, body):
        conditions, args = [], []
        for key, column in (("session", "h.parent_session"), ("task_id", "h.task_id"), ("state", "h.state"),
                            ("kind", "h.kind"), ("priority", "h.priority"), ("origin", "t.origin")):
            value = getattr(body, key)
            if value is not None:
                conditions.append(column + "=?")
                args.append(value.value if hasattr(value, "value") else value)
        if body.cursor:
            conditions.append("h.id>?")
            args.append(body.cursor)
        rows = await self.all("SELECT h.* FROM hands h JOIN tasks t ON t.id=h.task_id" +
                              (" WHERE " + " AND ".join(conditions) if conditions else "") + " ORDER BY h.id", args)
        rows = [row for row in rows if not self.parent_forgotten(row)]
        return {"hands": rows, "cursor": rows[-1]["id"] if rows else body.cursor}

    async def _valid_name(self, proposed, task_id=None):
        reserved = {"owner", "stop", "all", "list", "show", "log", "tasks", "obligations", "report", "revise",
                    "terminal", "execute_code", "delegate_task", *self.config.get("reserved_names", [])}
        if not re.fullmatch(r"[A-Za-z0-9-]{1,24}", proposed) or proposed.casefold() in {str(n).casefold() for n in reserved}:
            raise HTTPException(status_code=422, detail="invalid_name")
        rows = await self.all("SELECT h.task_id FROM hands h JOIN tasks t ON t.id=h.task_id "
                              "WHERE lower(h.name)=lower(?) AND t.state IN ('open','waiting')", (proposed,))
        if any(row["task_id"] != task_id for row in rows):
            conflict("name_in_use")
        return proposed

    @staticmethod
    def shell_only_kind(item, kind):
        """A coding task naming no repository, path or code need is shell work: plan it as ops.

        Coding is defined by what it changes (its workdir's repository or its touches) and
        a test command needs a tree to run in. Without either, the task is structurally ops,
        which outranks the requested kind; the plan records the change as `kind_adjusted`.
        """
        if (kind == "coding" and not item.workdir and not item.touches and set(item.needs) <= {"shell"}
                and not (item.accept and item.accept.test_command)):
            return "ops"
        return kind

    def resources(self, item, kind):
        if item.touches:
            return sorted(set(item.touches))
        if kind in {"coding", "integrate"}:
            if not item.workdir:
                raise HTTPException(status_code=422, detail="coding_requires_workdir_or_resources")
            path = Path(item.workdir).expanduser().resolve()
            for parent in (path, *path.parents):
                if (parent / ".git").exists():
                    return [str(parent)]
            return [str(path)]
        return ["ops:local"] if kind == "ops" else []

    async def plan(self, body, key):
        await self.health.refresh()
        async def apply():
            await self._expire()
            plans = []
            for item in body.tasks:
                if item.parts:
                    raise HTTPException(status_code=422, detail="parts_require_planning_unit")
                old = None
                queued_request = None
                snapshot = await self.resolve_definition(item.hand) if item.hand else None
                if item.continues:
                    old = await self.one("SELECT * FROM hands WHERE id=?", (item.continues,))
                    task = await self.get_task(old["task_id"] if old else item.continues)
                    queued_request = await self.one("SELECT * FROM definition_task_queue WHERE task_id=? AND dispatched=0",
                                                   (task['id'],))
                    if queued_request:
                        claim = await self.definition_claim(task['id'], item.queue_claim, body.parent_session)
                        if claim['hand_id']:
                            conflict('definition_claim_already_planned')
                    saved_definition = await self.task_definition(task["id"])
                    if snapshot and (not saved_definition or snapshot["id"] != saved_definition["id"]):
                        conflict("task_definition_mismatch")
                    snapshot = saved_definition
                    if snapshot:
                        await self.resolve_definition(snapshot["id"])
                    if task["state"] in {"cancelled", "done"}:
                        conflict("task_closed")
                    if old is None:
                        old = await self.one("SELECT * FROM hands WHERE task_id=? ORDER BY attempt DESC LIMIT 1", (task["id"],))
                    if old and old["decisions"].get("erased"):
                        raise HTTPException(status_code=410, detail="parent_erased")
                    if await self.one("SELECT id FROM hands WHERE task_id=? AND state IN "
                                      "('reserved','dispatched','running','checking','integrating')", (task["id"],)):
                        conflict("task_active")
                    if queued_request:
                        # Unbound pickup retries have never run a child. They
                        # are first-attempt admission retries, not cascades.
                        old = None
                    kind = item.kind.value if item.kind else task["kind"]
                    # A continuation keeps its task's kind, though it copies the request's original field.
                    adjusted = (old or {}).get("decisions", {}).get("kind_adjusted")
                    if adjusted and kind != task["kind"] and self.shell_only_kind(item, kind) == task["kind"]:
                        kind = task["kind"]
                    else:
                        adjusted = None
                else:
                    if body.origin == "completion":
                        raise HTTPException(status_code=422, detail="completion_requires_task")
                    kind = item.kind.value if item.kind else "deep"
                    planned_kind = self.shell_only_kind(item, kind)
                    adjusted = ({"from": kind, "to": planned_kind, "reason": "coding_without_workdir_or_resources"}
                                if planned_kind != kind else None)
                    kind = planned_kind
                    priority = item.priority.value if item.priority else ("auto" if body.origin == "mind" else "own" if body.origin == "own" else "owner")
                    if body.origin in {"own", "mind"}:
                        priority = "own" if body.origin == "own" else "auto"
                    task_id = "t-" + secrets.token_hex(6)
                    attempt_budget = budgets.attempt(self.config, kind)
                    resources = self.resources(item, kind)
                    risk = item.risk.value if item.risk else "normal"
                    if set(resources) & set(self.config.get("critical_resources", [])) and risk in {"low", "normal"}:
                        risk = "high"
                    await self.insert("tasks", id=task_id, origin=body.origin, origin_ref=plain(body.origin_ref),
                                      priority=priority, kind=kind, risk=risk, needs=item.needs, resources=resources,
                                      depends_on=item.depends_on, budget=budgets.task_budget(self.config, attempt_budget),
                                      intents={"1": {"goal_hash": sha256(item.goal.encode()).hexdigest(),
                                                      "acceptance": plain(item.accept) if item.accept else {}, "set_at": self.now()}},
                                      created_at=self.now())
                    task = await self.get_task(task_id)
                    if snapshot:
                        await self.bind_definition(task_id, snapshot)
                    if body.origin == "owner_turn":
                        await self._create_obligation(task)
                        task = await self.get_task(task_id)
                task_id = task["id"]
                if queued_request:
                    # A queued request's exact brief, not model-supplied replacement prose.
                    item = item.model_copy(update={"goal": queued_request["brief"]})
                cascade_count = await cascade.used(self, task) if old else 0
                cascade_reason = await cascade.reason(self, task) if old else None
                attempt_no = (await self.one("SELECT coalesce(max(attempt),0)+1 AS n FROM hands WHERE task_id=?", (task_id,)))["n"]
                hand_id = "h-" + secrets.token_hex(6)
                default_name = f"{kind}-{task_id[2:8]}" if not old else f"{old['name'].rsplit('-', 1)[0]}-{attempt_no}"
                name = await self._valid_name(item.name or default_name, task_id)
                role, effort = matrix.route(kind, self.config)
                allowance = budgets.attempt(self.config, kind, effort)
                configured = self.config["contracts"]
                chain = self.config.get("routes", {}).get(role, [role] if role in configured else [])
                if isinstance(chain, str):
                    chain = [chain]
                excluded = old["families_used"] if kind == "verify" and old else []
                valid, dropped = contracts.validate_chain(chain, configured, needs=task["needs"], kind=kind,
                    input_tokens=len((item.goal + item.context).encode()),
                    growth_tokens=self.config.get("context_growth_tokens", {}).get(kind, 0), excluded_families=excluded)
                provider = valid[0] if valid else None
                contract = {"provider": provider, "version": str(configured[provider]["version"])} if provider else None
                await self.insert("hands", id=hand_id, task_id=task_id, attempt=attempt_no, intent_version=task["intent_version"],
                                  name=name, kind=kind, effort=effort, role=role, contract=contract,
                                  parent_session=body.parent_session, origin_ref=task["origin_ref"], priority=task["priority"],
                                  executor="hermes_kanban" if task["origin"] == "mind" else "hermes_delegation",
                                  epoch={"owner_pid": 0, "owner_started_at": 0.0}, budgets=allowance,
                                  continues=old["id"] if old else None,
                                  decisions={"decision_ids": body.decision_ids, "static_effort": 2, "served_effort": 2,
                                             "control": True, "chain": valid, "dropped": dropped, "workdir": item.workdir,
                                             "acceptance": task["intents"][str(task["intent_version"])]["acceptance"],
                                             **({"definition_snapshot": snapshot} if snapshot else {}),
                                             **({'queue_return_ref': plain(body.origin_ref)} if queued_request else {}),
                                             **({"kind_adjusted": adjusted} if adjusted else {})}, created_at=self.now())
                hand = await self.get_hand(hand_id)
                if adjusted:
                    await self.log(hand_id, "kind_adjusted", adjusted)
                reason = None
                unknown = await self.all("SELECT * FROM hands WHERE task_id=? AND state='unknown_effect'", (task_id,))
                unreconciled = [row for row in unknown if not row["decisions"].get("effect_reconciled")]
                if not self.enabled:
                    reason = "faculty_off"
                elif not valid:
                    reason = "contract"
                elif unreconciled and not item.reconciled:
                    reason = "unreconciled_effect"
                elif cascade_reason:
                    reason = cascade_reason
                elif task["origin"] in self.config["open_tasks"]:
                    count = await self.one("SELECT count(*) AS n FROM tasks WHERE origin=? AND state IN ('open','waiting')", (task["origin"],))
                    if count["n"] > self.config["open_tasks"][task["origin"]]:
                        reason = "open_task_limit"
                if reason:
                    admitted, reason, token = await admission.defer(self, hand, task, reason, refuse=True)
                    if reason in {"open_task_limit", "contract", "faculty_off"} and not old and not queued_request:
                        await self.update("tasks", "id", task_id, state="failed", closed_at=self.now())
                else:
                    if item.reconciled:
                        await self.log(hand_id, "reconciled_continuation", {"previous": old["id"] if old else ""})
                        for previous in unreconciled:
                            await self.update("hands", "id", previous["id"], decisions={**previous["decisions"], "effect_reconciled": True})
                    family = configured[provider]["family"]
                    admitted, reason, token = await admission.reserve(self, hand, task, family, charge=not queued_request)
                if queued_request:
                    await self.update('definition_task_dispatch', 'task_id', task_id, hand_id=hand_id)
                    if admitted != 'start':
                        # The app request remains the queue. Do not also leave a
                        # promotable controller attempt behind on refusal.
                        claim = await self.definition_claim(task_id, item.queue_claim)
                        await self._release_definition_claim(claim, 'refused:' + reason)
                        admitted, token = 'refuse', None
                if old and admitted != "refuse":
                    await cascade.charge(self, hand, task, cascade_count)
                await self.log(hand_id, "plan", {"admit": admitted, "reason": reason})
                plans.append({"task_id": task_id, "hand_id": hand_id, "name": name, "kind": kind, "effort": effort,
                              "risk": task["risk"], "role": role, "contract": contract, "budgets": allowance,
                              "brief": await self._brief(body, item, task, name, allowance),
                              "admit": admitted, "reason": reason, "lease_token": token})
            return {"plans": plans}
        return await self.write(key, "/plan", body, apply)

    async def dispatched(self, ident, body, key):
        async def apply():
            await self.guard(ident, states={"reserved"})
            reservation = await self.one("SELECT * FROM reservations WHERE hand_id=?", (ident,))
            if (not reservation or reservation["token"] != body.lease_token or reservation["state"] != "held"
                    or reservation["expires_at"] <= self.now()):
                conflict("reservation_lapsed_or_invalid")
            await self.update("hands", "id", ident, state="dispatched", executor_ref=body.delegation_id,
                              epoch=plain(body.epoch), activity_since=self.now())
            await self.update("reservations", "hand_id", ident, state="confirmed")
            await self.log(ident, "dispatched")
            return await self.get_hand(ident)
        return await self.write(key, f"/{ident}/dispatched", body, apply)

    async def release(self, ident, body, key):
        async def apply():
            hand = await self.guard(ident, states={"reserved", "dispatched"})
            reservation = await self.one("SELECT * FROM reservations WHERE hand_id=?", (ident,))
            if not reservation or reservation["token"] != body.lease_token:
                conflict("lease_token")
            if hand["state"] == "dispatched":
                await self.guard(ident, body.epoch)
            await self._end(hand, "refused", body.reason or "released")
            return await self.get_hand(ident)
        return await self.write(key, f"/{ident}/release", body, apply)

    async def bind(self, ident, body, key):
        async def apply():
            hand = await self.guard(ident, body.epoch, {"dispatched", "running"})
            # Dispatch records Hermes' delegation id; the child's start gives its
            # subagent id, which stop and steer address. The first bind moves the
            # reference to the subagent and keeps the delegation id for the ledger;
            # a later bind (a rotated child session) must name the same subagent.
            moved = {}
            if hand["state"] == "dispatched":
                moved = {"executor_ref": body.executor_ref,
                         "decisions": {**hand["decisions"], "delegation_id": hand["executor_ref"]}}
            elif hand["executor_ref"] != body.executor_ref:
                conflict("executor_ref")
            binding = await self.one("SELECT * FROM hand_sessions WHERE session_id=?", (body.child_session,))
            if await self.one("SELECT session_id FROM hand_session_tombstones WHERE session_id=?", (body.child_session,)):
                conflict("child_session_bound")
            if binding and (binding["hand_id"] != ident or binding["epoch"] != plain(body.epoch)):
                conflict("child_session_bound")
            if not binding:
                await self.insert("hand_sessions", session_id=body.child_session, hand_id=ident,
                                  parent_session=hand["parent_session"], epoch=plain(body.epoch), registered_at=self.now())
            await self.update("hands", "id", ident, state="running", child_session=body.child_session,
                              started_at=hand["started_at"] or self.now(), activity="queued", activity_since=self.now(),
                              **moved)
            await self.log(ident, "bound")
            if hand["started_at"] is None:
                await self._notice(f"ntc:{ident}:start", ident, "routine", f"{hand['name']} started ({hand['kind']}, effort {hand['effort']}).")
            return await self.get_hand(ident)
        return await self.write(key, f"/{ident}/bind", body, apply)

    async def _end(self, hand, state, reason):
        await self.update("hands", "id", hand["id"], state=state, outcome=reason, ended_at=self.now())
        await self.update("reservations", "hand_id", hand["id"], state="released")
        await locks.release(self, hand["id"])
        await self.log(hand["id"], state, {"reason": reason})
        task = await self.get_task(hand["task_id"])
        if task["state"] not in {"done", "cancelled", "failed"}:
            await self.update("tasks", "id", task["id"], state="waiting")
        line = f"{hand['name']} {state}: {reason}"
        if not self._notice_text_allowed(line):
            line = f"{hand['name']} {state}; details in /hands."
        await self._notice(f"ntc:{hand['id']}:end", hand["id"],
                           "actionable" if state in {"failed", "unknown_effect", "below_minimum", "stopped"} else "routine",
                           line[:self.config["notices"]["max_chars"]])

    async def stop(self, ident, body, key):
        async def apply():
            hand = await self.guard(ident, body.epoch)
            if hand["state"] not in TERMINAL:
                await self._end(hand, "stopped", body.reason)
                await self._ready(hand["task_id"])
            return await self.get_hand(ident)
        return await self.write(key, f"/{ident}/stop", body, apply)

    async def name(self, ident, body, key):
        async def apply():
            hand = await self.get_hand(ident)
            name = await self._valid_name(body.name, hand["task_id"])
            await self.update("hands", "id", ident, name=name)
            return await self.get_hand(ident)
        return await self.write(key, f"/{ident}/name", body, apply)

    async def events(self, ident, body, key):
        async def apply():
            current = await self.guard(ident)
            if current["state"] == "reserved":
                # The plugin journals its dispatch intent before Hermes returns a
                # delegation id, so no executor epoch is bound yet. Only that
                # record is accepted here; effects and progress wait for dispatch.
                if body.effects or not body.events or any(e.type != "dispatch_intent" for e in body.events):
                    conflict("prior_state")
                for event in body.events:
                    if not await self.one("SELECT key FROM events WHERE key=?", (event.key,)):
                        await self.log(ident, event.type, event.payload, key=event.key, source=event.source,
                                       seq=event.seq * 2, at=self.now())
                row = await self.one("SELECT coalesce(max(seq),0) AS n FROM events WHERE hand_id=?", (ident,))
                return {"cursor": str(row["n"])}
            await self.guard(ident, body.epoch, {"dispatched", "running", "checking", "integrating"})
            from .effects import ingest
            for effect in body.effects:
                await ingest(self, ident, effect)
            for event in body.events:
                if await self.one("SELECT key FROM events WHERE key=?", (event.key,)):
                    continue
                hand = await self.get_hand(ident)
                if hand["state"] in TERMINAL:
                    conflict("prior_state")
                payload = event.payload
                if event.type == "activity":
                    if payload.get("activity") not in {"queued", "generating", "executing"} or hand["state"] != "running":
                        conflict("invalid_activity")
                    await self.update("hands", "id", ident, activity=payload["activity"], activity_since=self.now())
                elif event.type == "checking":
                    await self.guard(ident, body.epoch, {"running"})
                    await self.update("hands", "id", ident, state="checking", activity=None)
                elif event.type == "usage":
                    await self._usage(hand, payload)
                elif event.type == "progress":
                    await self._progress(hand, event.key, payload.get("text", ""))
                await self.log(ident, event.type, payload, key=event.key, source=event.source, seq=event.seq * 2, at=self.now())
            row = await self.one("SELECT coalesce(max(seq),0) AS n FROM events WHERE hand_id=?", (ident,))
            return {"cursor": str(row["n"])}
        return await self.write(key, f"/{ident}/events", body, apply)

    async def _usage(self, hand, payload):
        usage = deepcopy(hand["usage"])
        task = await self.get_task(hand["task_id"])
        task_usage = dict(task["usage"])
        for name in ("tokens", "tool_calls", "web_searches", "wall_seconds"):
            current = payload.get(name, usage.get(name, 0))
            if type(current) not in (int, float) or current < usage.get(name, 0) or not __import__("math").isfinite(current):
                raise HTTPException(status_code=422, detail="usage_must_be_monotonic")
            delta = current - usage.get(name, 0)
            if name in {"tokens", "wall_seconds"}:
                task_usage[name] = task_usage.get(name, 0) + delta
            if name == "tokens" and delta:
                family = payload.get("family")
                if not family or family not in {v["family"] for v in self.config["contracts"].values()}:
                    raise HTTPException(status_code=422, detail="usage_family_required")
                daily = usage.setdefault("daily_tokens", {}).setdefault(self.day(), {})
                daily[family] = daily.get(family, 0) + delta
            usage[name] = current
        await self.update("hands", "id", hand["id"], usage=usage)
        await self.update("tasks", "id", task["id"], usage=task_usage)
        if any(usage.get(name, 0) >= cap for name, cap in hand["budgets"].items()):
            await self._end(hand, "stopped", "attempt_budget")
            await self._ready(task["id"])
        elif any(task_usage.get(name, 0) >= task["budget"][name] for name in ("tokens", "wall_seconds")):
            await self._end(hand, "stopped", "task_budget")
            await self._ready(task["id"])
        elif hand["priority"] != "owner" and payload.get("family"):
            family = payload["family"]
            rows = await self.all("SELECT usage FROM hands WHERE priority!='owner'")
            tokens = sum(row["usage"].get("daily_tokens", {}).get(self.day(), {}).get(family, 0) for row in rows)
            limits = self.config["budgets"]["day"]
            cap = limits["tokens_per_family"].get(family, limits["default_tokens_per_family"])
            if tokens >= cap:
                await self._end(hand, "stopped", "daily_budget")
                await self._ready(task["id"])

    async def question(self, ident, body, key):
        async def apply():
            hand = await self.guard(ident, body.epoch, {"running"})
            qid = "q-" + secrets.token_hex(6)
            await self.insert("questions", id=qid, hand_id=ident, asked_at=self.now(),
                              text=redact_sensitive_text(body.question), assumption=redact_sensitive_text(body.assumption),
                              blocking=body.blocking)
            if body.blocking:
                await self.update("hands", "id", ident, activity="awaiting_input", activity_since=self.now())
                await self._end(hand, "stopped", "needs_input")
                await self._ready(hand["task_id"])
            await self.log(ident, "question", {"question_id": qid, "blocking": body.blocking})
            return await self.require("questions", qid)
        return await self.write(key, f"/{ident}/question", body, apply)

    async def answer(self, qid, body, key, *, answered_by=None):
        async def apply():
            question = await self.require("questions", qid)
            if question["state"] != "open":
                conflict("question_closed")
            await self.update("questions", "id", qid, state="answered", answer=redact_sensitive_text(body.answer),
                              answered_by=answered_by or body.answered_by, answered_at=self.now())
            await self.log(question["hand_id"], "answer", {"question_id": qid})
            return await self.require("questions", qid)
        return await self.write(key, f"/questions/{qid}/answer", body, apply, strict_body=True)

    def provenance(self):
        from .provenance import Provenance
        if self.ledger is None or not self.owner_id:
            logger.warning("Hands provenance skipped: turn ledger or owner contact unavailable")
            return None
        return Provenance(self, self.ledger, owner_id=self.owner_id)

    async def _record_accepted(self, ident):
        try:
            service = self.provenance()
            if service:
                hand = await self.get_hand(ident)
                await service.record_accepted(ident, hand["decisions"].get("result_text", ""))
        except Exception:
            logger.exception("Hands accepted provenance unavailable")

    async def _brief(self, body, item, task, name, allowance):
        from .sessions import lookup
        try:
            owner = not await lookup(self.store, body.parent_session) and self.parent_is_owner(body.parent_session) is True
        except Exception:
            logger.exception("Hands parent standing unavailable; planning without memory")
            owner = False
        memory = []
        if owner and self.recall:
            try:
                memory = list(await self.recall(query=item.goal, audience="owner", session=body.parent_session))
            except Exception:
                logger.exception("Hands recall unavailable; planning without memory")
        try:
            return build_brief(goal=item.goal, context=item.context, intent_version=task["intent_version"],
                definition_context=definition_context(await self.task_definition(task["id"])),
                acceptance=task["intents"][str(task["intent_version"])]["acceptance"], name=name,
                parent_is_owner=owner, recall=lambda **kwargs: memory, budgets=allowance,
                persona_addendum=self.config["persona_addendum"], style=self.config["style"],
                max_tokens=self.config["brief"]["max_tokens"]).text
        except BriefTooLarge:
            conflict("brief_too_large")

    async def result(self, ident, body, key):
        async def apply():
            hand = await self.guard(ident, body.epoch)
            # A budget stop still takes the attempt's partial result: the controller's own budget
            # stops, and the plugin's local token and wall budget stops (plugin budgets.stop).
            waiting_end = hand["state"] == "stopped" and hand["outcome"] in {
                "needs_input", "awaiting_approval", "attempt_budget", "task_budget", "daily_budget",
                "token_budget", "wall_budget"}
            if hand["state"] not in {"running", "checking", "integrating"} and not waiting_end:
                conflict("prior_state")
            if body.intent_version != hand["intent_version"]:
                conflict("intent_version")
            task = await self.get_task(hand["task_id"])
            evidence = self.evidence_evaluator(hand, task, body)
            state = "below_minimum"
            if task["intent_version"] != hand["intent_version"]:
                state = "superseded"
            elif body.status in {"failed", "stopped"}:
                state = body.status
            elif body.status == "done" and not waiting_end:
                criteria = task["intents"][str(hand["intent_version"])]["acceptance"]
                # The version guard above establishes currency for this rule.
                accepted = acceptance_outcome(Evidence(level=evidence["level"], scope=EvidenceScope(**evidence["scope"])),
                    kind=hand["kind"], risk=task["risk"], intent_version=hand["intent_version"],
                    current_version=hand["intent_version"], test_applies=bool(criteria.get("test_command") or criteria.get("probe")))
                state = accepted.state.value
            elif body.status in {"needs_input", "blocked"} or waiting_end:
                state = "stopped"
            if hand["started_at"]:
                wall = max(hand["usage"].get("wall_seconds", 0), (self.clock() - self.parse_time(hand["started_at"])).total_seconds())
                spent = dict(hand["usage"])
                task_usage = dict(task["usage"])
                task_usage["wall_seconds"] = task_usage.get("wall_seconds", 0) + wall - spent.get("wall_seconds", 0)
                spent["wall_seconds"] = wall
                await self.update("hands", "id", ident, usage=spent)
                await self.update("tasks", "id", task["id"], usage=task_usage)
            if body.summary_text and sha256(body.summary_text.encode()).hexdigest() != body.summary_hash:
                conflict("summary_hash")
            await self.update("hands", "id", ident, evidence=evidence, branch=body.summary_ref,
                              decisions={**hand["decisions"], "result_text": body.summary_text},
                              check=[plain_receipt(r) for r in body.receipts], families_used=sorted(set(body.families_used)),
                              model=sorted({r.model for r in body.contracts_used}),
                              contract={**(hand["contract"] or {}), "used": [plain_receipt(r) for r in body.contracts_used]})
            await self._end(hand, state, body.status)
            await budgets.outcome(self, hand, task, state == "failed")
            if state == "done":
                await self.update("tasks", "id", task["id"], state="done", closed_at=self.now())
                await self._record_accepted(ident)
            elif state == "stopped":
                await self.update("tasks", "id", task["id"], state="waiting")
            if state != "superseded":
                await self._ready(task["id"], accepted=state == "done")
            return {"outcome": state, "evidence": evidence,
                    "next_step": "none" if state == "done" else "wait" if state in {"superseded", "stopped", "failed"} else "continue"}
        return await self.write(key, f"/{ident}/result", body, apply)

    async def revise(self, tid, body, key):
        async def apply():
            task = await self.get_task(tid)
            if task["state"] == "cancelled":
                conflict("task_closed")
            version = task["intent_version"] + 1
            intents = dict(task["intents"])
            prior = intents[str(task["intent_version"])]
            intents[str(version)] = {"goal_hash": sha256(body.goal.encode()).hexdigest() if body.goal else prior["goal_hash"],
                                     "acceptance": plain(body.criteria) if body.criteria else prior["acceptance"], "set_at": self.now()}
            await self.update("tasks", "id", tid, intent_version=version, intents=intents, state="open", closed_at=None,
                              owner_spoke_since=None)
            if task["obligation_id"]:
                await self.update("obligations", "id", task["obligation_id"], state="superseded", closed_reason="revised")
                await self._create_obligation(task)
            hands = await self.all("SELECT * FROM hands WHERE task_id=?", (tid,))
            for hand in hands:
                if hand["state"] not in TERMINAL:
                    if body.stop or hand["state"] in {"planned", "reserved"}:
                        await self._end(hand, "superseded", "revised")
                    else:
                        await self.log(hand["id"], "revise", {"intent_version": version, "stop": body.stop})
            await self.update("tasks", "id", tid, state="open")
            return await self.task_view(tid)
        return await self.write(key, f"/v1/hands/tasks/{tid}/revise", body, apply, strict_body=True)

    async def cancel(self, tid, body, key):
        async def apply():
            task = await self.get_task(tid)
            await self.update("tasks", "id", tid, state="cancelled", closed_at=self.now())
            for hand in await self.all("SELECT * FROM hands WHERE task_id=?", (tid,)):
                if hand["state"] not in TERMINAL:
                    await self._end(hand, "stopped", "task_cancelled")
                await self.log(hand["id"], "cancel", {"owner_ref": plain(body.owner_ref), "reason": body.reason})
            if task["obligation_id"]:
                await self.update("obligations", "id", task["obligation_id"], state="cancelled",
                                  closed_reason=encode({"reason": body.reason, "owner_ref": plain(body.owner_ref)}))
            return await self.task_view(tid)
        return await self.write(key, f"/v1/hands/tasks/{tid}/cancel", body, apply, strict_body=True)

    async def accept(self, tid, body, key):
        async def apply():
            task = await self.get_task(tid)
            version = task["intent_version"] if body.for_version is None else body.for_version
            if version != task["intent_version"]:
                conflict("stale_result")
            hand = (await self.get_hand(body.hand) if body.hand else await self.one(
                "SELECT * FROM hands WHERE task_id=? AND intent_version=? ORDER BY attempt DESC LIMIT 1",
                (tid, version)))
            if (not hand or hand["task_id"] != tid or task["state"] == "cancelled"
                    or hand["state"] not in {"done", "below_minimum", "superseded"} or not hand["evidence"]):
                conflict("acceptance_guard")
            await self.log(hand["id"], "accepted_judgment", {"reason": body.reason, "for_version": version,
                                                           "evidence": hand["evidence"]["level"]})
            await self.update("tasks", "id", tid, state="done", closed_at=self.now())
            if task["obligation_id"]:
                obligation = await self.obligation(task["obligation_id"])
                await self.update("obligations", "id", obligation["id"], accepted_at=self.now(),
                                  reviewed_at=obligation["reviewed_at"] or self.now(),
                                  state="accepted" if obligation["state"] in {"result_ready", "reviewed"}
                                  else obligation["state"])
            await self._record_accepted(hand["id"])
            return await self.task_view(tid)
        return await self.write(key, f"/v1/hands/tasks/{tid}/accept", body, apply, strict_body=True)

    async def _expire(self):
        rows = await self.all("SELECT h.* FROM hands h JOIN reservations r ON r.hand_id=h.id "
                              "WHERE r.state='held' AND r.expires_at<=? AND h.state='reserved'", (self.now(),))
        for hand in rows:
            await self._end(hand, "expired", "reservation_lapsed")
            await self.update("reservations", "hand_id", hand["id"], state="expired")
        return [row["id"] for row in rows]

    async def sweep(self):
        async def apply():
            expired = await self._expire()
            reserved, lookup = [], []
            rows = await self.all("SELECT * FROM hands WHERE state IN ('planned','dispatched','running','checking','integrating') "
                                  "ORDER BY CASE priority WHEN 'owner' THEN 0 ELSE 1 END, created_at, rowid")
            for hand in rows:
                if hand["state"] == "dispatched":
                    if (self.clock() - self.parse_time(hand["activity_since"])).total_seconds() >= self.config["dispatch_start_seconds"]:
                        lookup.append(hand["id"])
                    continue
                if hand["state"] in {"running", "checking", "integrating"}:
                    if hand["started_at"]:
                        wall = (self.clock() - self.parse_time(hand["started_at"])).total_seconds()
                        await self._usage(hand, {"wall_seconds": max(wall, hand["usage"].get("wall_seconds", 0))})
                        if (wall >= hand["budgets"]["wall_seconds"] * self.config["budgets"]["wall_warning_fraction"]
                                and wall < hand["budgets"]["wall_seconds"] and not hand["decisions"].get("wall_warned")):
                            await self.log(hand["id"], "wrap_up", {"reason": "wall_budget"})
                            await self.update("hands", "id", hand["id"], decisions={**hand["decisions"], "wall_warned": True})
                    continue
                task = await self.get_task(hand["task_id"])
                if task["state"] in {"cancelled", "done"}:
                    await self._end(hand, "stopped", "task_closed")
                    continue
                deadline = self.config["queue"]["deadline_seconds"][hand["priority"]]
                if hand["priority"] == "owner" and (self.clock() - self.parse_time(hand["created_at"])).total_seconds() >= self.config["notices"]["queued_actionable_seconds"]:
                    await self._notice(f"ntc:{hand['id']}:queue-wait", hand["id"], "actionable", f"{hand['name']} is still queued; /hands.")
                if (self.clock() - self.parse_time(hand["created_at"])).total_seconds() >= deadline:
                    await self._end(hand, "expired", "queue_deadline")
                    continue
                provider = (hand["contract"] or {}).get("provider")
                if provider not in self.config["contracts"] or not self.enabled:
                    continue
                admitted, _, _ = await admission.reserve(self, hand, task, self.config["contracts"][provider]["family"])
                if admitted == "start":
                    reserved.append(hand["id"])
            overdue, retention = [], {"events": 0, "tasks": 0, "hands": 0, "replays": 0}
            for name, operation in (("overdue", self._overdue_obligations), ("retention", self._prune_retained)):
                await self.execute("SAVEPOINT hands_maintenance")
                try:
                    value = await operation()
                except Exception:
                    await self.execute("ROLLBACK TO hands_maintenance")
                    logger.exception("Hands %s maintenance failed; core sweep continues", name)
                else:
                    if name == "overdue":
                        overdue = value
                    else:
                        retention = value
                finally:
                    await self.execute("RELEASE hands_maintenance")
            return {"expired": expired, "reserved": reserved, "lookup": lookup,
                    "overdue": overdue, "retention": retention}
        return await self.write(None, "/sweep", {}, apply, record=False)

    async def erase_parent(self, session, key):
        async def apply():
            service = self.provenance()
            if service:
                await service.erase_parent(session)
            hands = await self.all("SELECT * FROM hands WHERE parent_session=?", (session,))
            ids = {hand["id"] for hand in hands}
            tasks = {hand["task_id"] for hand in hands}
            questions = set()
            for hand in hands:
                questions.update(row["id"] for row in await self.all("SELECT id FROM questions WHERE hand_id=?", (hand["id"],)))
                if hand["state"] not in TERMINAL:
                    await self._end(hand, "stopped", "parent_erased")
                await self.update("hands", "id", hand["id"], branch=None, evidence=None, check=[],
                                  decisions={"erased": True})
                await self.execute("UPDATE questions SET text=NULL, assumption=NULL, answer=NULL, state='erased' WHERE hand_id=?", (hand["id"],))
                await self.execute("DELETE FROM derivations WHERE hand_id=? AND derived_kind='relay'", (hand["id"],))
                await self.execute("UPDATE derivations SET state='retracted', evidence_ref='' WHERE hand_id=?", (hand["id"],))
                await self.execute("DELETE FROM events WHERE hand_id=?", (hand["id"],))
                await self.execute("DELETE FROM notices WHERE hand_id=?", (hand["id"],))
            # Replay must not resurrect erased content. Keep the keys as
            # tombstones, so an old plan cannot recreate the task either.
            for row in await self.all("SELECT key, path, response FROM idempotency"):
                serialized = encode(row["response"])
                if session in serialized or any(ident in serialized or ident in row["path"] for ident in ids | tasks | questions):
                    await self.update("idempotency", "key", row["key"], response={"erased": True})
            return {"erased": len(hands)}
        return await self.write(key, "/erase_parent", {"session": session}, apply)

    async def probe(self, ident, body, key):
        async def apply():
            hand = await self.guard(ident, body.epoch, {"unknown_effect"})
            receipt = plain_receipt(body.receipt)
            await self.update("hands", "id", ident, decisions={**hand["decisions"], "probe": receipt})
            await self.log(ident, "probe", {"passed": receipt["passed"]})
            return await self.get_hand(ident)
        return await self.write(key, f"/{ident}/probe", body, apply)

    async def relay(self, body, key):
        async def apply():
            from .sessions import lookup
            if await lookup(self.store, body.session):
                conflict("hand_session")
            for ident in body.hand_ids:
                hand = await self.guard(ident)
                if not hand.get("evidence"):
                    conflict("result_required")
            # A turn id cannot be rebound to a different session, even with a new key.
            for prior in await self.all("SELECT response FROM idempotency WHERE path='/relay'"):
                response = prior["response"]
                if response.get("turn") == body.turn:
                    if response.get("session") != body.session:
                        conflict("relay_turn_session")
                    return response
            service = self.provenance()
            refs = await service.record_relay(body.turn, body.hand_ids) if service else []
            return {"session": body.session, "turn": body.turn, "hand_refs": refs}
        return await self.write(key, "/relay", body, apply)

    async def owner_message(self, reference, key):
        async def apply():
            for task in await self.all("SELECT * FROM tasks WHERE state IN ('open','waiting')"):
                origin = task["origin_ref"]
                if isinstance(origin, dict) and all(origin.get(k) == reference.get(k) for k in ("platform", "chat")) and origin != reference:
                    await self.update("tasks", "id", task["id"], owner_spoke_since=reference)
            return {"recorded": True}
        return await self.write(key, "/owner_message", reference, apply)


def plain_receipt(value):
    from dataclasses import asdict, is_dataclass
    return asdict(value) if is_dataclass(value) else plain(value)


# Digest reads the same snapshots but does not mutate them.
Controller.digest = render_digest


_runtime = None
_runtime_lock = None


def faculty_reader(home=None):
    """``mind.faculties.hands`` as the instance config says now, re-read only when the file changes.

    A file that cannot be read or validated (for example caught mid-write) keeps the last value read, so a failed
    read never turns the faculty on or off by itself; before any good read the faculty counts as on (its default).
    """
    state = {"stamp": None, "value": True}

    def read():
        try:
            stat = config_path(home).stat()
            stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            stamp = None
        if stamp != state["stamp"]:
            try:
                state["value"] = load_config(home).data["mind"]["faculties"].get("hands", True) is not False
                state["stamp"] = stamp
            except Exception:
                logger.warning("hands faculty flag unreadable; keeping the last value")
        return state["value"]

    return read


async def get_controller():
    global _runtime, _runtime_lock
    if _runtime is None:
        if _runtime_lock is None:
            _runtime_lock = asyncio.Lock()
        async with _runtime_lock:
            if _runtime is None:
                config = load_config()
                _runtime = Controller(config.home / DB_FILE, config=config.data["hands"],
                                      enabled=config.data["mind"]["faculties"].get("hands", True),
                                      faculty=faculty_reader(config.home))
                await _runtime.connect()
                from .runtime import configure
                configure(_runtime)
                _runtime.start_maintenance()
    return _runtime


async def plan(body, idempotency_key):
    return await (await get_controller()).plan(body, idempotency_key)


async def dispatched(id, body, idempotency_key):
    return await (await get_controller()).dispatched(id, body, idempotency_key)


async def release(id, body, idempotency_key):
    return await (await get_controller()).release(id, body, idempotency_key)


async def bind(id, body, idempotency_key):
    return await (await get_controller()).bind(id, body, idempotency_key)


async def events(id, body, idempotency_key):
    return await (await get_controller()).events(id, body, idempotency_key)


async def question(id, body, idempotency_key):
    return await (await get_controller()).question(id, body, idempotency_key)


async def answer(qid, body, idempotency_key):
    return await (await get_controller()).answer(qid, body, idempotency_key)


async def result(id, body, idempotency_key):
    return await (await get_controller()).result(id, body, idempotency_key)


async def revise(tid, body, idempotency_key):
    return await (await get_controller()).revise(tid, body, idempotency_key)


async def cancel(tid, body, idempotency_key):
    return await (await get_controller()).cancel(tid, body, idempotency_key)


async def accept(tid, body, idempotency_key):
    return await (await get_controller()).accept(tid, body, idempotency_key)


async def list_hands(body):
    return await (await get_controller()).list_hands(body)


async def get_task(tid):
    return await (await get_controller()).task_view(tid)


async def stop(id, body, idempotency_key):
    return await (await get_controller()).stop(id, body, idempotency_key)


async def name(id, body, idempotency_key):
    return await (await get_controller()).name(id, body, idempotency_key)


async def get_hand(id):
    return await (await get_controller()).get_hand(id)
