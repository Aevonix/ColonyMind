"""Transactional owner-goal intentions, immutable progress, proposals and durable receipts."""
from __future__ import annotations

from contextlib import contextmanager
from colonymind.util.temporal import now_utc
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import uuid

from .artifacts import ArtifactHidden, ArtifactResolver
from .authority import ActionContext, Audience, VerifiedAction
from .sealing import Codec
from .schema import save_visibility
from .validation import (GoalError, RESPONSE_BYTES, TIERS, dumps, identifier, integer, operation,
                         timestamp, validate)

GOAL_KINDS = ("owner_goal", "owner_subgoal")
KINDS = (*GOAL_KINDS, "owner_goal_progress")


def now():
    return now_utc().isoformat().replace("+00:00", "Z")


def new_id():
    return uuid.uuid4().hex


def stricter(*tiers):
    return max(tiers, key=TIERS.index)


def page(values, limit, offset, *, key="items"):
    """Consume at most limit + 1 projections; trim at the byte boundary with real continuation."""
    iterator = iter(values)
    items = []
    more = False
    for value in iterator:
        if len(items) == limit:
            more = True
            break
        candidate = {key: [*items, value], "limit": limit, "offset": offset,
                     "next_offset": offset + len(items) + 1}
        # Reserve space for a wrapping single-goal response and enclosing page metadata.
        if len(dumps(candidate).encode("utf-8")) > RESPONSE_BYTES - 4096:
            if not items:
                raise GoalError(503, "owner_goals_unavailable")
            more = True
            break
        items.append(value)
    return {key: items, "limit": limit, "offset": offset,
            "next_offset": offset + len(items) if more else None}


class OwnerGoalsStore:
    """Owns separate record, history and join tables in the intentions file.

    Context status is active/completed. Engine status is recorded, with no horizon,
    budgets, success checks or dispatch state. All owner reads go through this store.
    """
    def __init__(self, intentions, *, resolver: ArtifactResolver | None = None, vault=None):
        self.db_path = Path(getattr(intentions, "_db_path", intentions))
        self.resolver = resolver or ArtifactResolver()
        self.codec = Codec(self.db_path.parent, vault)

    def _encode(self, value, owner):
        return self.codec.seal(dumps(value), owner)

    def _plaintext(self, value):
        from colonymind.secrets.vault import VaultUnavailable
        try:
            return self.codec.plaintext(value)
        except VaultUnavailable:
            raise GoalError(503, "owner_goal_content_unavailable") from None

    def _decode(self, value):
        try:
            return json.loads(self._plaintext(value))
        except (ValueError, TypeError):
            raise GoalError(503, "owner_goal_content_unavailable") from None

    @staticmethod
    def seal_existing(path, *, vault=None, dry_run=False):
        from .sealing import seal_existing
        return seal_existing(path, vault=vault, dry_run=dry_run)

    @contextmanager
    def connection(self, *, write=False):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA secure_delete=ON")
            db.create_function("owner_goal_storage_rewrite", 2, lambda old, new: False)
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _record_tier(self, db, owner, id, seen=None):
        """Resolve inheritance using structural metadata only, never content."""
        seen = set() if seen is None else seen
        if id in seen:
            return None
        seen.add(id)
        row = db.execute("""SELECT r.parent_goal_id,v.sensitivity,v.tier_override
            FROM owner_goal_records r LEFT JOIN owner_goal_visibility v ON v.record_id=r.id
            WHERE r.id=? AND r.entity_id=?""", (id, owner)).fetchone()
        if row is None or row["sensitivity"] is None:
            return None
        if row["tier_override"] or row["parent_goal_id"] is None:
            return row["sensitivity"]
        return self._record_tier(db, owner, row["parent_goal_id"], seen)

    def _admitted(self, db, owner, id, audience):
        if audience.owner_only:
            return True
        tier = self._record_tier(db, owner, id)
        return tier is not None and audience.allows(tier)

    def _row(self, db, owner, id, *, live=True, progress=False, audience=None):
        row = db.execute("SELECT * FROM owner_goal_records WHERE id=? AND entity_id=? AND kind IN (?,?,?)",
                         (id, owner, *KINDS)).fetchone()
        if row is None or (not progress and row["kind"] not in GOAL_KINDS):
            raise GoalError(404, "owner_goal_not_found")
        if audience is not None and not self._admitted(db, owner, id, audience):
            raise GoalError(404, "owner_goal_not_found")
        visibility = db.execute("SELECT deleted FROM owner_goal_visibility WHERE record_id=?", (id,)).fetchone()
        if live and visibility is not None and visibility[0]:
            raise GoalError(404, "owner_goal_not_found")
        data = self._decode(row["context"])
        if live and data.get("_deleted_at"):
            raise GoalError(404, "owner_goal_not_found")
        return data

    def _save(self, db, data):
        owner = db.execute("SELECT entity_id FROM owner_goal_records WHERE id=?", (data["id"],)).fetchone()[0]
        db.execute("UPDATE owner_goal_records SET context=? WHERE id=?", (self._encode(data, owner), data["id"]))
        save_visibility(db, data)

    def _insert(self, db, owner, principal, kind, data, source_id):
        db.execute("""INSERT INTO owner_goal_records
            (id,type,kind,description,status,entity_id,created_by,source_id,source_type,
             parent_goal_id,context,created_at,hermes_kind)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (data["id"], "owner_record", kind, "", "recorded", owner, principal,
                    self.codec.seal(source_id, owner), "owner_goal_action", data.get("parent_id"),
                    self._encode(data, owner),
                    data.get("created_at", now()), "none"))
        save_visibility(db, data)

    def _audit(self, db, id, principal, action, evidence, details=None):
        owner = db.execute("""SELECT entity_id FROM owner_goal_records WHERE id=?
            UNION ALL SELECT owner_id FROM owner_goal_proposals WHERE id=?""", (id, id)).fetchone()[0]
        db.execute("""INSERT INTO owner_goal_history (initiative_id,agent_id,action,timestamp,details)
                    VALUES (?,?,?,?,?)""",
                   (id, principal, "owner_goal_" + action, now(),
                    self._encode({"evidence_id": evidence, **(details or {})}, owner)))

    def _effective(self, db, owner, data):
        if data.get("_tier_override") or data.get("parent_id") is None:
            return data["sensitivity"]
        tier = self._record_tier(db, owner, data["parent_id"])
        if tier is not None:
            return tier
        parent = self._row(db, owner, data["parent_id"], live=False)
        return self._effective(db, owner, parent)

    def _goal(self, db, owner, id, audience):
        return self._row(db, owner, id, audience=audience)

    def _bump(self, db, owner, data, *, parent=True):
        data["revision"] += 1
        if data["revision"] > 9007199254740991:
            raise GoalError(503, "owner_goals_unavailable")
        data["updated_at"] = now()
        self._save(db, data)
        if parent and data.get("parent_id"):
            ancestor = self._row(db, owner, data["parent_id"])
            self._bump(db, owner, ancestor, parent=False)

    def _revision(self, data, expected):
        if data["revision"] != expected:
            raise GoalError(409, "revision_conflict")

    def _parent(self, db, owner, id):
        if id is None:
            return None
        try:
            parent = self._row(db, owner, id)
        except GoalError as exc:
            raise GoalError(409, "invalid_parent") from exc
        if parent["parent_id"] is not None:
            raise GoalError(409, "invalid_parent")
        return parent

    def _materialize(self, db, owner, kind, payload, target=None):
        result = validate(kind, payload)
        parent = self._parent(db, owner, result["parent_id"]) if kind == "create" else target
        result.setdefault("sensitivity", self._effective(db, owner, parent) if parent else "Secret")
        return result, parent

    def _create(self, db, context, payload, override, evidence):
        at = now()
        data = {**payload, "id": new_id(), "revision": 1, "status": "active",
                "created_at": at, "updated_at": at, "completed_at": None,
                "_tier_override": override}
        self._insert(db, context.owner_id, context.principal_id,
                     "owner_subgoal" if data["parent_id"] else "owner_goal", data, context.body.get("source_id"))
        if data["parent_id"]:
            parent = self._row(db, context.owner_id, data["parent_id"])
            self._bump(db, context.owner_id, parent)
        self._audit(db, data["id"], context.principal_id, "create", evidence)
        return data

    def _progress(self, db, context, target, payload, override, evidence):
        data = {**payload, "id": new_id(), "parent_id": target["id"],
                "_tier_override": override, "_sort_at": timestamp(payload["at"])}
        self._insert(db, context.owner_id, context.principal_id, "owner_goal_progress", data,
                     context.body.get("source_id"))
        self._bump(db, context.owner_id, target)
        self._audit(db, target["id"], context.principal_id, "progress", evidence, {"progress_id": data["id"]})
        return target

    def _proposal(self, db, owner, id, audience):
        if audience.owner_id != owner or not audience.owner_only:
            raise GoalError(404, "owner_goal_proposal_not_found")
        row = db.execute("SELECT * FROM owner_goal_proposals WHERE id=? AND owner_id=?", (id, owner)).fetchone()
        if row is None:
            raise GoalError(404, "owner_goal_proposal_not_found")
        data = self._decode(row["data"])
        data["sensitivity"] = self._proposal_tier(db, owner, data, bool(row["tier_override"]))
        if not audience.allows(data["sensitivity"]):
            raise GoalError(404, "owner_goal_proposal_not_found")
        return data, bool(row["tier_override"])

    def _proposal_tier(self, db, owner, data, override):
        target = data.get("target_id") or data.get("payload", {}).get("parent_id")
        if target and not override:
            return self._effective(db, owner, self._row(db, owner, target, live=False))
        return data["sensitivity"]

    def mutate(self, context: ActionContext, action: str, verifier, *, id=None, type=None, ref=None):
        """Authorize before lookups, then verify resolved effects or retained receipt evidence."""
        operation(context.operation_id)
        body = dict(context.body)
        expected_hash = hashlib.sha256(dumps([context.method, context.path, body]).encode()).hexdigest()
        if (context.request_hash != expected_hash or not isinstance(context.audience, Audience)
                or context.audience.owner_id != context.owner_id or not context.audience.owner_only):
            raise GoalError(403, "not_owner")
        with self.connection(write=True) as db:
            # Request-only authorization must precede receipt, target, parent,
            # proposal, artifact and revision lookups, including validation errors
            # whose outcome might otherwise depend on a target's existence.
            authorization = {"phase": "authorize", "action": action, "id": id,
                             "type": type, "ref": ref, "body": body,
                             "tier_instruction": "sensitivity" in body or (
                                 action == "propose" and isinstance(body.get("payload"), dict)
                                 and "sensitivity" in body["payload"])}
            self._verify(context, action, verifier, authorization, None, db)
            row = db.execute("""SELECT * FROM owner_goal_receipts
                WHERE principal_id=? AND owner_id=? AND operation_id=?""",
                             (context.principal_id, context.owner_id, context.operation_id)).fetchone()
            receipt = dict(row) if row is not None else None
            if receipt is not None:
                if receipt["request_hash"] != context.request_hash:
                    raise GoalError(409, "idempotency_conflict")
                # Verification sees the original logical bytes, including requests that
                # already contained references. Never expand arbitrary secret references.
                for column in ("result", "resolved_action"):
                    receipt[column] = self._plaintext(receipt[column])
                resolved = json.loads(receipt["resolved_action"])
            else:
                validated = validate(action, body)
                resolved = self._resolve(db, context, action, validated, id, type, ref)
            grant = self._verify(context, action, verifier, resolved, receipt, db)
            if receipt is not None:
                if grant.evidence_id != receipt["evidence_id"]:
                    raise GoalError(403, "not_owner")
                return receipt["status"], json.loads(receipt["result"])
            status, result = self._apply(db, context, resolved, grant.evidence_id)
            if len(dumps(result).encode()) > RESPONSE_BYTES:
                raise GoalError(503, "owner_goals_unavailable")
            db.execute("""INSERT INTO owner_goal_receipts VALUES (?,?,?,?,?,?,?,?,?)""",
                       (context.principal_id, context.owner_id, context.operation_id, context.request_hash,
                        status, self._encode(result, context.owner_id), grant.evidence_id,
                        self._encode(resolved, context.owner_id), now()))
            return status, result

    @staticmethod
    def _verify(context, action, verifier, resolved, receipt, db):
        grant = verifier(context, resolved_action=resolved, receipt=receipt, connection=db)
        purpose = {"propose": "proposal_only", "confirm": "confirmation"}.get(action, "requested_write")
        if (not isinstance(grant, VerifiedAction) or grant.owner_id != context.owner_id
                or grant.principal_id != context.principal_id or not grant.evidence_id
                or grant.purpose != purpose or (resolved.get("tier_instruction") and not grant.explicit_tier)):
            raise GoalError(403, "not_owner")
        return grant

    def _resolve(self, db, ctx, action, body, id, type, ref):
        owner, audience = ctx.owner_id, ctx.audience
        resolved = {"action": action, "id": id, "type": type, "ref": ref, "body": body}
        target = None
        if action in {"patch", "progress", "attach", "detach", "delete"}:
            target = self._goal(db, owner, id, audience)
            if "expected_revision" in body:
                self._revision(target, body["expected_revision"])
            resolved["target_revision"] = target["revision"]
            resolved["target_sensitivity"] = self._effective(db, owner, target)
        if action in {"create", "progress"}:
            payload, parent = self._materialize(db, owner, action, dict(ctx.body), target)
            resolved.update(payload=payload, tier_override="sensitivity" in ctx.body,
                            parent_revision=parent["revision"] if parent else None)
        elif action == "propose":
            kind = "create" if body["kind"] == "goal" else "progress"
            target = self._goal(db, owner, body["target_id"], audience) if kind == "progress" else None
            payload, parent = self._materialize(db, owner, kind, body["payload"], target)
            resolved.update(payload=payload, target_revision=parent["revision"] if parent else None,
                            tier_override="sensitivity" in body["payload"])
        elif action == "confirm":
            proposal, override = self._proposal(db, owner, id, audience)
            if proposal["state"] != "proposed":
                raise GoalError(409, "proposal_already_confirmed")
            self._revision(proposal, body["expected_revision"])
            target_id = proposal["target_id"] or proposal["payload"].get("parent_id")
            if target_id is not None:
                parent = self._goal(db, owner, target_id, audience)
                self._revision(parent, proposal["target_revision"])
            resolved.update(proposal=proposal, tier_override=override, payload=proposal["payload"])
        elif action == "attach":
            handle = self.resolver.resolve_owner_artifact(owner, body["type"], body["ref"], audience)
            resolved["artifact_binding"] = handle.binding()
        elif action == "detach":
            relation = db.execute("""SELECT 1 FROM owner_goal_artifacts
                WHERE goal_id=? AND type=? AND ref=? AND detached_at IS NULL""", (id, type, ref)).fetchone()
            if relation is None:
                raise GoalError(404, "artifact_not_found")
        # Every explicit tier override must be owner-authorized; the verifier sees all inherited defaults.
        resolved["tier_instruction"] = ("sensitivity" in ctx.body or bool(resolved.get("tier_override")))
        return resolved

    def _apply(self, db, ctx, resolved, evidence):
        action, id, body = resolved["action"], resolved["id"], resolved["body"]
        owner, principal, audience = ctx.owner_id, ctx.principal_id, ctx.audience
        if action == "propose":
            at = now()
            proposal = {"id": new_id(), "revision": 1, "state": "proposed", "kind": body["kind"],
                        "target_id": body["target_id"], "target_revision": resolved["target_revision"],
                        "payload": resolved["payload"], "source_id": ctx.body.get("source_id"),
                        "sensitivity": resolved["payload"]["sensitivity"], "created_at": at,
                        "updated_at": at, "confirmed_at": None, "result_goal_id": None}
            db.execute("INSERT INTO owner_goal_proposals VALUES (?,?,?,?,?,?)",
                       (proposal["id"], owner, self._encode(proposal, owner), int(resolved["tier_override"]),
                        body["target_id"] or proposal["payload"].get("parent_id"), at))
            self._audit(db, proposal["id"], principal, "propose", evidence,
                        {"payload_digest": hashlib.sha256(dumps(proposal["payload"]).encode()).hexdigest()})
            return 201, {"proposal": proposal}
        if action == "confirm":
            # Preserve the exact pre-confirmation effects verified above for the receipt.
            proposal = dict(resolved["proposal"])
            if proposal["kind"] == "goal":
                goal = self._create(db, ctx, proposal["payload"], resolved["tier_override"], evidence)
            else:
                target = self._row(db, owner, proposal["target_id"])
                goal = self._progress(db, ctx, target, proposal["payload"], resolved["tier_override"], evidence)
            proposal.update(state="confirmed", revision=proposal["revision"] + 1,
                            updated_at=now(), confirmed_at=now(), result_goal_id=goal["id"])
            db.execute("UPDATE owner_goal_proposals SET data=? WHERE id=?", (self._encode(proposal, owner), id))
            self._audit(db, id, principal, "confirm", evidence)
            return 200, {"proposal": proposal, "goal": self._project(db, owner, goal, audience)}
        if action == "create":
            goal = self._create(db, ctx, resolved["payload"], resolved["tier_override"], evidence)
            return 201, {"goal": self._project(db, owner, goal, audience)}
        goal = self._row(db, owner, id)
        if action == "delete":
            rows = db.execute("""SELECT context FROM owner_goal_records WHERE entity_id=? AND kind IN (?,?)
                AND (id=? OR parent_goal_id=?)""", (owner, *GOAL_KINDS, id, id))
            for row in rows.fetchall():
                data = self._decode(row["context"])
                if not data.get("_deleted_at"):
                    data.update(_deleted_at=now(), _deleted_by=principal)
                    self._bump(db, owner, data, parent=False)
                    self._audit(db, data["id"], principal, "delete", evidence)
            if goal["parent_id"]:
                self._bump(db, owner, self._row(db, owner, goal["parent_id"]))
            return 200, {"deleted_id": id}
        if action == "progress":
            goal = self._progress(db, ctx, goal, resolved["payload"], resolved["tier_override"], evidence)
        else:
            if action == "patch":
                old_tier = self._effective(db, owner, goal)
                for key, value in body.items():
                    if key != "expected_revision":
                        goal[key] = value
                if "sensitivity" in body:
                    goal["_tier_override"] = True
                if goal["status"] == "active":
                    goal["completed_at"] = None
                elif goal["completed_at"] is None:
                    goal["completed_at"] = now()
                if old_tier != self._effective(db, owner, goal):
                    self._audit(db, id, principal, "tier_propagation", evidence,
                                {"from": old_tier, "to": self._effective(db, owner, goal)})
                    # Inherited child projections changed too. Their revision invalidates any
                    # pending display/confirmation; original progress records stay untouched.
                    for row in db.execute("""SELECT r.context FROM owner_goal_records r
                        LEFT JOIN owner_goal_visibility v ON v.record_id=r.id WHERE r.entity_id=?
                        AND r.kind='owner_subgoal' AND r.parent_goal_id=?
                        AND COALESCE(v.deleted,0)=0""", (owner, id)).fetchall():
                        child = self._decode(row["context"])
                        if not child.get("_tier_override"):
                            self._bump(db, owner, child, parent=False)
                            self._audit(db, child["id"], principal, "tier_propagation", evidence,
                                        {"ancestor_id": id, "to": goal["sensitivity"]})
            elif action == "attach":
                current = self.resolver.resolve_owner_artifact(owner, body["type"], body["ref"], audience)
                if current.binding() != resolved["artifact_binding"]:
                    raise GoalError(503, "owner_goals_unavailable")
                db.execute("""INSERT INTO owner_goal_artifacts VALUES (?,?,?,?,?,?,NULL)
                    ON CONFLICT(goal_id,type,ref) DO UPDATE SET label=excluded.label,
                    added_at=CASE WHEN detached_at IS NULL THEN added_at ELSE excluded.added_at END,
                    binding=excluded.binding,detached_at=NULL""",
                           (id, body["type"], body["ref"], self.codec.seal(body["label"], owner), now(),
                            self._encode(resolved["artifact_binding"], owner)))
            elif action == "detach":
                db.execute("""UPDATE owner_goal_artifacts SET detached_at=?
                    WHERE goal_id=? AND type=? AND ref=?""", (now(), id, resolved["type"], resolved["ref"]))
            self._bump(db, owner, goal)
            self._audit(db, id, principal, action, evidence,
                        {"resolved_action": resolved})
        return 200 if action != "progress" else 201, {"goal": self._project(db, owner, goal, audience)}

    def _progress_rows(self, db, owner, id, audience, offset=0):
        # Audience filtering occurs before offsets; all inherited tiers are resolved at read time.
        rows = db.execute("""SELECT r.id,r.context FROM owner_goal_records r
            LEFT JOIN owner_goal_visibility v ON v.record_id=r.id
            WHERE r.entity_id=? AND r.kind='owner_goal_progress'
            AND r.parent_goal_id=? ORDER BY v.sort_at DESC,r.id ASC""", (owner, id))
        skipped = 0
        for row in rows:
            if not self._admitted(db, owner, row["id"], audience):
                continue
            data = self._decode(row["context"])
            if skipped < offset:
                skipped += 1
                continue
            yield {key: data[key] for key in ("id", "at", "text", "value", "unit")} | {
                "sensitivity": self._effective(db, owner, data)}

    def _children(self, db, owner, id, audience, offset=0):
        rows = db.execute("""SELECT r.id,r.context FROM owner_goal_records r
            LEFT JOIN owner_goal_visibility v ON v.record_id=r.id
            WHERE r.entity_id=? AND r.kind='owner_subgoal'
            AND r.parent_goal_id=? AND COALESCE(v.deleted,0)=0
            ORDER BY r.created_at ASC,r.id ASC""", (owner, id))
        skipped = 0
        for row in rows:
            if not self._admitted(db, owner, row["id"], audience):
                continue
            data = self._decode(row["context"])
            if data.get("_deleted_at"):
                continue
            if skipped < offset:
                skipped += 1
                continue
            yield self._project(db, owner, data, audience)

    def _handles(self, db, owner, goal, audience):
        rows = db.execute("""SELECT * FROM owner_goal_artifacts WHERE goal_id=? AND detached_at IS NULL
            ORDER BY added_at ASC,type ASC,ref ASC""", (goal["id"],))
        for relation in rows:
            try:
                handle = self.resolver.resolve_owner_artifact(owner, relation["type"], relation["ref"], audience)
                pinned = self._decode(relation["binding"])
                if handle.namespace_id != pinned["namespace_id"]:
                    raise GoalError(503, "owner_goals_unavailable")
            except ArtifactHidden:
                continue
            except GoalError as exc:
                # Existing relations with lost eligibility cannot look like exhausted history.
                raise GoalError(503, "owner_goals_unavailable") from exc
            yield relation, handle

    def _artifacts(self, db, owner, goal, audience, offset=0):
        # Revalidate even relations preceding offset: an invalid relation cannot disappear behind pagination.
        for index, (relation, handle) in enumerate(self._handles(db, owner, goal, audience)):
            if index < offset:
                continue
            yield {"type": relation["type"], "ref": relation["ref"], "label": self._plaintext(relation["label"]),
                   "added_at": relation["added_at"],
                   "sensitivity": stricter(self._effective(db, owner, goal), handle.sensitivity),
                   "cron": {"enabled": handle.enabled, "recurring": handle.recurring}
                   if handle.type == "cron_job" else None}

    def _runs(self, handle, owner, audience, sensitivity):
        offset = 0
        previous = None
        while True:
            result = self.resolver.list_owner_job_runs(owner, handle, audience, 100, offset)
            try:
                if (set(result) != {"items", "limit", "offset", "next_offset"}
                        or result["limit"] != 100 or result["offset"] != offset
                        or not isinstance(result["items"], list) or len(result["items"]) > 100):
                    raise ValueError()
                next_offset = result["next_offset"]
                if next_offset is not None:
                    integer(next_offset)
                    if not result["items"] or next_offset != offset + len(result["items"]):
                        raise ValueError()
                for run in result["items"]:
                    # Internal adapters return the contract's public activity projection after verifying
                    # original output, immutable run identity and per-run destination evidence.
                    from .validation import text
                    if set(run) != {"id", "source_id", "kind", "at", "title", "summary", "artifact_ref", "sensitivity"}:
                        raise ValueError()
                    identifier(run["id"])
                    text(run["source_id"], 256, blank=False)
                    text(run["title"], 400)
                    text(run["summary"], 1000, multiline=True)
                    if run["kind"] != "cron_run" or run["artifact_ref"] != handle.ref or run["sensitivity"] not in TIERS:
                        raise ValueError()
                    key = self._activity_key(run)
                    if previous is not None and key < previous:
                        raise ValueError()
                    previous = key
                    if audience.allows(run["sensitivity"]):
                        yield {**run, "sensitivity": stricter(sensitivity, handle.sensitivity, run["sensitivity"])}
            except (KeyError, TypeError, ValueError, GoalError) as exc:
                raise GoalError(503, "owner_goals_unavailable") from exc
            if next_offset is None:
                break
            offset = next_offset

    @staticmethod
    def _activity_key(item):
        # Reverse the exact sortable UTC timestamp, while source identity remains ascending.
        instant = timestamp(item["at"])
        whole, fraction = instant.split(".")
        from decimal import Decimal
        return Decimal(whole + "." + (fraction or "0")).copy_negate(), item["source_id"]

    def _activity(self, db, owner, goal, audience, offset=0):
        def progress():
            for item in self._progress_rows(db, owner, goal["id"], audience):
                title = next((line.strip().replace("\t", " ") for line in item["text"].splitlines()
                              if line.strip()), "")[:400]
                yield {"id": item["id"], "source_id": "progress:" + item["id"], "kind": "progress",
                       "at": item["at"], "title": title, "summary": item["text"][:1000],
                       "artifact_ref": None, "sensitivity": item["sensitivity"]}
        streams = [progress()]
        for _, handle in self._handles(db, owner, goal, audience):
            if handle.type == "cron_job":
                streams.append(self._runs(handle, owner, audience, self._effective(db, owner, goal)))
        seen = set()
        skipped = 0
        for item in heapq.merge(*streams, key=self._activity_key):
            if item["source_id"] in seen:
                continue
            seen.add(item["source_id"])
            if skipped < offset:
                skipped += 1
            else:
                yield item

    def _project(self, db, owner, goal, audience):
        result = {key: value for key, value in goal.items() if not key.startswith("_")}
        result["sensitivity"] = self._effective(db, owner, goal)
        result["subgoals"] = page(self._children(db, owner, goal["id"], audience), 1, 0)
        result["progress"] = page(self._progress_rows(db, owner, goal["id"], audience), 1, 0)
        result["artifacts"] = page(self._artifacts(db, owner, goal, audience), 1, 0)
        result["activity"] = page(self._activity(db, owner, goal, audience), 1, 0)
        return result

    def read(self, owner, audience, *, id=None, collection=None, proposals=False, limit=50, offset=0):
        integer(offset)
        if type(limit) is not int or not 1 <= limit <= 100 or audience.owner_id != owner:
            raise GoalError(422, "invalid_request")
        if proposals and not audience.owner_only:
            if id is not None:
                raise GoalError(404, "owner_goal_proposal_not_found")
            return page((), limit, offset, key="proposals")
        with self.connection() as db:
            if proposals:
                if id:
                    return {"proposal": self._proposal(db, owner, id, audience)[0]}
                def proposed():
                    rows = db.execute("SELECT data,tier_override FROM owner_goal_proposals WHERE owner_id=? ORDER BY created_at DESC,id ASC", (owner,))
                    skipped = 0
                    for row in rows:
                        data = self._decode(row["data"])
                        data["sensitivity"] = self._proposal_tier(db, owner, data, bool(row["tier_override"]))
                        if not audience.allows(data["sensitivity"]):
                            continue
                        if skipped < offset:
                            skipped += 1
                        else:
                            yield data
                return page(proposed(), limit, offset, key="proposals")
            if id:
                goal = self._goal(db, owner, id, audience)
                if collection is None:
                    return {"goal": self._project(db, owner, goal, audience)}
                readers = {"subgoals": lambda: self._children(db, owner, id, audience, offset),
                           "progress": lambda: self._progress_rows(db, owner, id, audience, offset),
                           "artifacts": lambda: self._artifacts(db, owner, goal, audience, offset),
                           "activity": lambda: self._activity(db, owner, goal, audience, offset)}
                if collection not in readers:
                    raise GoalError(422, "invalid_request")
                return page(readers[collection](), limit, offset)
            def roots():
                rows = db.execute("""SELECT r.id,r.context FROM owner_goal_records r
                    LEFT JOIN owner_goal_visibility v ON v.record_id=r.id
                    WHERE r.entity_id=? AND r.kind='owner_goal' AND COALESCE(v.deleted,0)=0
                    ORDER BY v.status ASC,r.created_at DESC,r.id ASC""", (owner,))
                skipped = 0
                for row in rows:
                    if not self._admitted(db, owner, row["id"], audience):
                        continue
                    data = self._decode(row["context"])
                    if data.get("_deleted_at"):
                        continue
                    if skipped < offset:
                        skipped += 1
                    else:
                        yield self._project(db, owner, data, audience)
            return page(roots(), limit, offset, key="goals")

    def recall(self, owner, audience, *, limit=50, offset=0):
        """Audience-bound recall. Secret never enters mixed or non-owner context."""
        return self.read(owner, audience, limit=limit, offset=offset)

    def outreach_eligible(self, owner, id):
        with self.connection() as db:
            try:
                goal = self._row(db, owner, id)
                while True:
                    if goal["status"] != "active" or goal["muted"]:
                        return False
                    if not goal["parent_id"]:
                        return True
                    goal = self._row(db, owner, goal["parent_id"])
            except GoalError:
                return False
