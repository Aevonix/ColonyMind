"""A nightly learned revision carries its parent's live Instinct tier through the configured runtime."""

from types import SimpleNamespace

from colonymind.instinct import REGISTRY, state
from colonymind.plasticity.nightly import Runtime
from colonymind.plasticity.synthetic import Cycle, SyntheticSource, make_rows
from test_instinct_fit import NOW
from test_instinct_lineage import Live
from test_plasticity_wiring_runtime import config, declaration


async def test_nightly_child_promotion_carries_parent_live_before_its_first_decision(tmp_path, monkeypatch):
    live = await Live(tmp_path).promoted()
    monkeypatch.setitem(REGISTRY, live.point.name, live.point)
    [parent_record] = live.store.records()
    calibration = [parent_record.id]
    source = SyntheticSource(make_rows(60))

    def factory(*, store, controller, source, declaration, clock):
        return Cycle(store, controller, source=source, clock=clock, refit=lambda digest: tuple(calibration))

    service = Runtime.from_config(
        config(tmp_path, {"probe": declaration()}),
        check_learner=lambda name, declared: SimpleNamespace(name=name),
        factories={"synthetic": factory}, sources={"probe": source},
        instinct_store=live.store, clock=lambda: NOW)
    try:
        [first] = service.run()
        assert first["result"] == "promoted"
        parent = service.pin("probe", "owner").closure
        assert parent.calibration == (parent_record.id,)
        assert not any(change["source"].startswith(state.CARRIED) for change in live.store.transitions())

        child_record = live.child("rev-2", "rev-1")
        live.frozen("rev-2")
        calibration[:] = [child_record.id]
        source.add(make_rows(20, prefix="more"))
        assert live.rows_under("rev-2") == []
        [second] = service.run()

        assert second["result"] == "promoted", second
        child = service.pin("probe", "owner").closure
        assert child.parent == parent.id and child.id != parent.id
        assert child.calibration == (child_record.id,)
        carry = live.store.transitions()[0]
        assert carry["source"] == f"{state.CARRIED} (plasticity)"
        assert carry["from_state"] == carry["to_state"] == "live"
        assert carry["evidence"]["parent_closure"] == parent.id
        assert carry["evidence"]["closure"] == child.id
        assert carry["evidence"]["reached"] == parent_record.fingerprint
        assert carry["evidence"]["fingerprint"] == child_record.fingerprint
        assert live.rows_under("rev-2") == []

        await live.serves("rev-2")
        [decision] = live.rows_under("rev-2")
        assert decision["tiers"]["fast"]["state"] == "live"
        assert decision["answered_by"] == "fast" and decision["applied"]
        assert live.effective() == "live"
    finally:
        service.close()
