"""Journal ingestion and restart reconciliation never execute or retry an effect."""

from fastapi import HTTPException


async def ingest(c, hand, effect):
    if not effect.key.startswith(f"eff:{hand}:"):
        raise HTTPException(status_code=422, detail="effect_key_scope")
    existing = await c.one("SELECT * FROM effects WHERE key=?", (effect.key,))
    if existing:
        if any(existing[name] != getattr(effect, name) for name in ("tool", "call_hash", "workdir", "intent_at")) or existing["hand_id"] != hand:
            raise HTTPException(status_code=409, detail="effect_identity_changed")
        if existing["done_at"]:
            if any(existing[name] != getattr(effect, name) for name in ("outcome", "exit_code", "result_hash", "done_at")):
                raise HTTPException(status_code=409, detail="effect_already_finished")
            return
        if effect.done_at:
            if effect.outcome == "unknown":
                raise HTTPException(status_code=422, detail="effect_outcome_required")
            await c.update("effects", "key", effect.key, outcome=effect.outcome, exit_code=effect.exit_code,
                           result_hash=effect.result_hash, done_at=effect.done_at)
    else:
        if effect.done_at:
            raise HTTPException(status_code=409, detail="effect_intent_required")
        await c.insert("effects", hand_id=hand, **effect.model_dump(mode="json"))


class EffectOperations:
    async def reconcile(self, body, key):
        async def apply():
            current = body.epoch.model_dump(mode="json")
            live = {row.hand_id: row for row in body.live}
            ledger = {row.hand_id: row for row in body.ledger}
            changed = []
            rows = await self.all("SELECT * FROM hands WHERE state IN ('reserved','dispatched','running','checking','integrating')")
            for hand in rows:
                record = live.get(hand["id"])
                if record and (record.epoch.model_dump(mode="json") != hand["epoch"] or record.executor_ref != hand["executor_ref"]):
                    raise HTTPException(status_code=409, detail="executor_identity_mismatch")
                if record and record.status in {"running", "dispatched", "checking", "integrating"}:
                    continue
                known = ledger.get(hand["id"])
                if known and (known.epoch.model_dump(mode="json") != hand["epoch"] or known.executor_ref != hand["executor_ref"]):
                    raise HTTPException(status_code=409, detail="executor_identity_mismatch")
                if hand["epoch"] == current and not known:
                    continue
                # A ledger's completed status is not an acceptance receipt. The
                # plugin may replay its result before reconciling, otherwise the
                # result is lost and must be inspected, never rerun automatically.
                open_effect = await self.one("SELECT key FROM effects WHERE hand_id=? AND done_at IS NULL", (hand["id"],))
                state = "unknown_effect" if open_effect else "lost"
                if hand["state"] == "reserved":
                    state = "expired"
                await self._end(hand, state, "executor_epoch_ended")
                await self._ready(hand["task_id"])
                changed.append(await self.get_hand(hand["id"]))
            return {"hands": changed}
        return await self.write(key, "/reconcile", body, apply)


async def reconcile(body, idempotency_key):
    from .controller import get_controller
    return await (await get_controller()).reconcile(body, idempotency_key)
