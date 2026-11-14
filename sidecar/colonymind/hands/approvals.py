"""Exact-call holds through the mind's owner-answer path, with one-use permits."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import hashlib
import json
import uuid

from fastapi import HTTPException

from colonymind.mind.authority import effect_of
from colonymind.mind.outcomes import owner_answer
from . import sessions, wire


HAND_REVIEW_WAIT_SECONDS = 180  # Maximum hold-to-notice wait, also bounded by the hand's wall deadline.


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def call_digest(call: wire.ExactCall, workdir: str) -> str:
    return hashlib.sha256(canonical({"call": call.model_dump(), "workdir": workdir}).encode()).hexdigest()


def target_values(targets) -> list[dict]:
    values = [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in targets]
    keys = [(item["kind"], item["target"]) for item in values]
    if len(set(keys)) != len(keys):
        raise HTTPException(422, detail="duplicate target bindings")
    return sorted(values, key=lambda item: (item["kind"], item["target"]))


def bound_call(body: wire.HoldRequest) -> str:
    return canonical({"tool": body.call.tool, "args": body.call.arguments, "cwd": body.workdir,
                      "targets": target_values(body.targets)})


def _time(value) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value


class Approvals:
    def __init__(self, store, mind):
        self.store, self.mind = store, mind

    @asynccontextmanager
    async def _transaction(self):
        async with sessions.transaction_lock(self.store):
            await self.store.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                await self.store.db.commit()
                from .definitions import publish_store_changes
                await publish_store_changes(self.store)
            except BaseException:
                await self.store.db.rollback()
                raise

    async def _row(self, sql, args=()):
        rows = await self.store.db.execute_fetchall(sql, args)
        return dict(rows[0]) if rows else None

    async def _hand(self, ident):
        row = await self._row("SELECT h.*, t.intent_version AS current_version, t.state AS task_state, "
                              "t.origin AS task_origin, t.origin_ref AS task_origin_ref "
                              "FROM hands h JOIN tasks t ON t.id=h.task_id WHERE h.id=?", (ident,))
        if not row:
            raise HTTPException(404, detail="unknown hand")
        return row

    async def held_permits(self, session: str) -> wire.HeldPermitsResponse:
        """Read held targets for a front session without refreshing or consuming.

        A terminal hand can still own a permit for its continuation, so the
        permit state determines whether the front must leave its targets alone.
        """
        rows = await self.store.db.execute_fetchall(
            "SELECT p.id AS permit_id,p.ask_id,p.hand_id,h.parent_session,p.state,p.call,p.workdir,"
            "p.call_digest,p.targets "
            "FROM permits p JOIN hands h ON h.id=p.hand_id "
            "WHERE h.parent_session=? AND p.state IN ('waiting','granted') ORDER BY p.rowid", (session,))
        return wire.HeldPermitsResponse(permits=[wire.HeldPermit(
            **{**dict(row), "call": json.loads(row["call"]), "targets": json.loads(row["targets"])})
            for row in rows])

    async def granted_hands(self, approved: list[str]) -> list[dict]:
        """Return current, unused grants for asks the answer reader approved."""
        if not approved:
            return []
        placeholders = ",".join("?" for _ in approved)
        rows = await self.store.db.execute_fetchall(
            "SELECT p.id AS permit_id,p.ask_id,p.hand_id,h.parent_session,p.call,p.workdir "
            "FROM permits p JOIN hands h ON h.id=p.hand_id "
            f"WHERE p.state='granted' AND p.ask_id IN ({placeholders}) ORDER BY p.rowid", approved)
        return [{**dict(row), "call": json.loads(row["call"])} for row in rows]

    async def hold(self, id: str, body: wire.HoldRequest, idempotency_key: str) -> wire.HoldResponse:
        if not idempotency_key.strip():
            raise HTTPException(422, detail="Idempotency-Key must not be blank")
        request_hash = hashlib.sha256(canonical(body.model_dump(mode="json")).encode()).hexdigest()
        path = f"/v1/hands/{id}/hold"
        async with self._transaction():
            prior = await self._row("SELECT * FROM idempotency WHERE key=?", (idempotency_key,))
            if prior:
                if (prior["method"], prior["path"], prior["request_hash"]) != ("POST", path, request_hash):
                    raise HTTPException(409, detail="idempotency key names a different request")
                return wire.HoldResponse.model_validate_json(prior["response"])
            hand = await self._hand(id)
            if json.loads(hand["epoch"]) != body.epoch.model_dump():
                raise HTTPException(409, detail="executor epoch does not match")
            if hand["intent_version"] != hand["current_version"] or hand["task_state"] not in {"open", "waiting"}:
                raise HTTPException(409, detail="task intent is no longer current")
            answer = await self._hold(hand, body)
            await self.store.db.execute(
                "INSERT INTO idempotency(key,method,path,request_hash,response,status_code,created_at) "
                "VALUES (?,?,?,?,?,?,?)", (idempotency_key, "POST", path, request_hash,
                    answer.model_dump_json(), 200, self.mind.clock().isoformat()))
            return answer

    async def _hold(self, hand, body):
        digest, targets = call_digest(body.call, body.workdir), canonical(target_values(body.targets))
        await self._supersede(hand, digest, targets, body.workdir)
        prior = await self._row("SELECT * FROM permits WHERE hand_id=? AND call_digest=? AND targets=? "
            "AND state IN ('waiting','granted') ORDER BY rowid DESC LIMIT 1", (hand["id"], digest, targets))
        if prior:
            prior = await self._refresh(prior)
            if prior["state"] in {"waiting", "granted"}:
                return wire.HoldResponse(permit_id=prior["id"], ask_id=prior["ask_id"])
        call = bound_call(body)
        binding_digest = hashlib.sha256(call.encode()).hexdigest()
        effect = effect_of(body.call.tool, body.call.arguments, body.workdir)
        described = (effect.cls.replace("_", " ") if effect else "effect") + ": " + body.call.tool
        delivery_ref = None
        if hand["task_origin"] == "owner_turn":
            try:
                reference = wire.MessageRef.model_validate_json(hand["task_origin_ref"])
            except ValueError:
                pass  # Older tasks may have only a textual origin reference.
            else:
                if reference.platform == "rcs":
                    delivery_ref = reference.model_dump()
        # No summary, brief, claimed approval or rationale enters this binding.
        held = self.mind._hold(None, {"effect": {"cls": effect.cls if effect else "effect",
            "tool": body.call.tool, "call": call, "digest": binding_digest, "text": described + "; " + call},
            "reason": "this exact call and its target values need the owner's yes"},
            task_id=hand["id"], now=self.mind.clock(), origin="hand", delivery_ref=delivery_ref)
        ask = self.mind.store.get(held["ask"])
        if "hand_approval_wait" not in ask.context:
            now = self.mind.clock()
            # The wait includes the review and its reply gap. Neither a hung
            # hand nor a lost native completion can extend the hand's lifetime.
            wall = json.loads(hand["budgets"]).get("wall_seconds", HAND_REVIEW_WAIT_SECONDS)
            started = _time(hand.get("started_at") or hand["created_at"])
            deadline = max(now, min(now + timedelta(seconds=HAND_REVIEW_WAIT_SECONDS),
                                    started + timedelta(seconds=wall)))
            self.mind.store.update(ask.id, context={**ask.context, "hand_approval_wait": {
                "hand_id": hand["id"], "held_at": now.isoformat(), "deadline": deadline.isoformat()}})
        ident = "permit-" + uuid.uuid4().hex
        await self.store.db.execute(
            "INSERT INTO permits(id,ask_id,hand_id,task_id,call,call_digest,workdir,targets,state) "
            "VALUES (?,?,?,?,?,?,?,?, 'waiting')", (ident, held["ask"], hand["id"], hand["task_id"],
                canonical(body.call.model_dump()), digest, body.workdir, targets))
        await self.store.db.execute("UPDATE hands SET activity='awaiting_approval', activity_since=? WHERE id=?",
                                    (self.mind.clock().isoformat(), hand["id"]))
        return wire.HoldResponse(permit_id=ident, ask_id=held["ask"])

    async def _supersede(self, hand, digest, targets, workdir):
        """One parent has one open ask for an exact call, even after another delegation.

        Runs inside the same write transaction as hold and check. Replacing a
        hand's ask revokes its permit, including for its continuations. The new
        hand gets a fresh ask and never inherits another hand's owner answer.
        """
        rows = await self.store.db.execute_fetchall(
            "SELECT p.* FROM permits p JOIN hands h ON h.id=p.hand_id "
            "WHERE h.parent_session=? AND p.hand_id<>? AND p.call_digest=? AND p.targets=? "
            "AND p.workdir=? AND p.state IN ('waiting','granted')",
            (hand["parent_session"], hand["id"], digest, targets, workdir))
        for row in rows:
            # Withdraw the ask first. If the hands transaction rolls back, its
            # next refresh still refuses the cancelled ask rather than grants it.
            self.mind.outcomes.record(row["ask_id"], status="cancelled", by="mind", implicit_verdict=False,
                summary="a newer hand replaced this identical held call",
                details={"superseded_by_hand": hand["id"]})
            await self.store.db.execute("UPDATE permits SET state='stale' WHERE id=?", (row["id"],))

    async def _refresh(self, permit):
        """A credential or caller cannot grant. Only the stored answer reader can."""
        if permit["state"] not in {"waiting", "granted"}:
            return permit
        ask = self.mind.store.get(permit["ask_id"])
        now = self.mind.clock()
        state = permit["state"]
        if ask is None or ask.status in {"denied", "cancelled", "failed"}:
            state = "refused"
        elif ask.status == "expired" or (ask.status == "asked" and ask.expires_at and ask.expires_at <= now):
            state = "lapsed"
        elif ask.status == "asked":
            state = "waiting"  # Includes the owner's undo of an earlier approval.
        elif ask.status in {"approved", "dispatched"}:
            history = self.mind.store.get_history(ask.id, limit=200)  # Same bound as mind permit history.
            approved = [h for h in history if h.action == "queued" and
                owner_answer((h.details or {}).get("by"), h.details) is True and
                (h.details or {}).get("answer", {}).get("message")]
            if not approved:
                state = "refused"
            else:
                answer = max(approved, key=lambda h: h.timestamp)
                granted = _time(answer.timestamp)
                expires = granted + timedelta(hours=self.mind.authority.policy.permit_hours)
                state = "granted" if now < expires else "lapsed"
                permit.update(owner_answer_ref=answer.details["answer"]["message"],
                              granted_at=granted.isoformat(), expires_at=expires.isoformat())
        else:
            state = "refused"
        permit["state"] = state
        await self.store.db.execute("UPDATE permits SET state=?,owner_answer_ref=?,granted_at=?,expires_at=? WHERE id=?",
            (state, permit.get("owner_answer_ref"), permit.get("granted_at"), permit.get("expires_at"), permit["id"]))
        return permit

    async def refresh_pending(self):
        """Persist the owner's answer without consuming execution authority."""
        async with self._transaction():
            rows = await self.store.db.execute_fetchall("SELECT * FROM permits WHERE state IN ('waiting','granted')")
            for row in rows:
                await self._refresh(dict(row))

    async def check(self, body: wire.PermitCheckRequest, idempotency_key: str) -> wire.PermitCheckResponse:
        if not idempotency_key.strip():
            raise HTTPException(422, detail="Idempotency-Key must not be blank")
        fingerprint = hashlib.sha256(canonical({"hand": body.hand, "call_digest": body.call_digest,
            "targets_now": target_values(body.targets_now)}).encode()).hexdigest()
        path = "/v1/hands/permits/check"
        async with self._transaction():
            prior = await self._row("SELECT * FROM idempotency WHERE key=?", (idempotency_key,))
            if prior:
                if (prior["method"], prior["path"], prior["request_hash"]) != ("GET", path, fingerprint):
                    raise HTTPException(409, detail="idempotency key names a different request")
                response = wire.PermitCheckResponse.model_validate_json(prior["response"])
                if response.decision == "allow":
                    return wire.PermitCheckResponse(decision="hold", reason="this execution check was already consumed",
                                                   permit_id=response.permit_id)
                return response
            response = await self._check(body)
            await self.store.db.execute(
                "INSERT INTO idempotency(key,method,path,request_hash,response,status_code,created_at) "
                "VALUES (?,?,?,?,?,?,?)", (idempotency_key, "GET", path, fingerprint,
                    response.model_dump_json(), 200, self.mind.clock().isoformat()))
            return response

    async def _check(self, body):
        hand = await self._hand(body.hand)
        if hand["intent_version"] != hand["current_version"] or hand["task_state"] not in {"open", "waiting"}:
            return wire.PermitCheckResponse(decision="hold", reason="task intent is no longer current")
        # A continuation follows its own ancestry, never a sibling's grant.
        allowed, current = {hand["id"]}, hand
        while current.get("continues") and current["continues"] not in allowed:
            current = await self._hand(current["continues"])
            if current["task_id"] != hand["task_id"] or current["intent_version"] != hand["intent_version"]:
                break
            allowed.add(current["id"])
        rows = await self.store.db.execute_fetchall("SELECT * FROM permits WHERE task_id=? AND call_digest=? "
            "AND state IN ('waiting','granted') ORDER BY rowid DESC", (hand["task_id"], body.call_digest))
        targets = target_values(body.targets_now)
        for row in rows:
            if row["hand_id"] not in allowed:
                continue
            permit = await self._refresh(dict(row))
            if permit["state"] != "granted":
                continue
            bound = json.loads(permit["targets"])
            if bound != targets:
                await self.store.db.execute("UPDATE permits SET state='stale' WHERE id=?", (permit["id"],))
                keys = lambda values: {(v["kind"], v["target"]) for v in values}
                if keys(bound) == keys(targets):
                    renewed = wire.HoldRequest(epoch=json.loads(hand["epoch"]), call=json.loads(permit["call"]),
                                              workdir=permit["workdir"], targets=targets)
                    await self._hold(hand, renewed)
                return wire.PermitCheckResponse(decision="hold", reason="target values changed since approval",
                                                 permit_id=permit["id"])
            cursor = await self.store.db.execute("UPDATE permits SET state='consumed' WHERE id=? AND state='granted'",
                                                  (permit["id"],))
            if cursor.rowcount == 1:
                # Never replay an allow response, even for a repeated check key.
                return wire.PermitCheckResponse(decision="allow", permit_id=permit["id"])
        return wire.PermitCheckResponse(decision="hold", reason="no current unused owner permit")


@asynccontextmanager
async def service():
    from colonymind.api.routers.mind import get_mind
    mind = get_mind()
    if mind is None:
        raise HTTPException(503, detail="owner ask service is unavailable")
    async with sessions.open_store() as store:
        yield Approvals(store, mind)


async def hold(id: str, body: wire.HoldRequest, idempotency_key: str) -> wire.HoldResponse:
    async with service() as current:
        return await current.hold(id, body, idempotency_key)


async def held_permits(session: str) -> wire.HeldPermitsResponse:
    # Durable bindings remain readable before the mind service has restarted.
    from colonymind.api.routers.mind import get_mind
    async with sessions.open_store() as store:
        current = Approvals(store, get_mind())
        if current.mind is not None:
            await current.refresh_pending()
        response = await current.held_permits(session)
        return response.model_copy(update={"refreshed": current.mind is not None})


async def answer_grants(mind, approved: list[str], *, store=None) -> list[dict]:
    """Synchronize before returning an answer so the runtime can resume its hand."""
    if store is not None:
        current = Approvals(store, mind)
        await current.refresh_pending()
        return await current.granted_hands(approved)
    from colonymind.config import instance_home
    from .store import DB_FILE
    # Ordinary answers do not create a hands store.
    if not (instance_home() / DB_FILE).exists():
        return []
    async with sessions.open_store() as opened:
        return await answer_grants(mind, approved, store=opened)


async def check(body: wire.PermitCheckRequest, idempotency_key: str) -> wire.PermitCheckResponse:
    async with service() as current:
        return await current.check(body, idempotency_key)
