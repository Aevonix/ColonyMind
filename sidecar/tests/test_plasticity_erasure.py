"""Erasure for weights (docs/PLASTICITY.md, Erasure): unserve at once, delete, then rebuild or discard, with no
contaminated file left. The guarantee: the erased item is removed from every retained training artifact, and no
served revision was trained with it."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from colonymind.plasticity import _io, builds
from colonymind.plasticity.erasure import Chain, Erasure, JointLedger, scan
from colonymind.plasticity.revisions import RevisionStore
from colonymind.plasticity.states import Controller
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import EPOCH, Cycle, SyntheticSource, make_rows, state_hash

NOW = EPOCH + timedelta(days=10)


def _cycle(root, name, rows=(), *, mode="rebuild"):
    store = LearnerStore(root, name, base="synthetic-base@1", mode=mode, declared=("threshold",), clock=lambda: NOW)
    return Cycle(store, Controller(store, live=False, clock=lambda: NOW), source=SyntheticSource(list(rows)),
                 clock=lambda: NOW)


def _files_containing(root, needles):
    """Files holding a needle as a JSON string (so ``late-2`` is not found inside ``late-20``)."""
    found = []
    for path in root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            found += [str(path) for needle in needles if json.dumps(needle).encode() in data]
    return found


def test_erasure_withdraws_instinct_records_before_closure_deletion_and_retries(tmp_path, monkeypatch):
    from colonymind.instinct.log import Store
    from test_instinct_fit import record, yes_no

    cycle = _cycle(tmp_path / "weights", "probe", make_rows(60))
    closure, _, _ = cycle.step()
    log = Store(tmp_path / "instinct.db")
    removed = record(yes_no(), revision=closure.state)
    clean = record(yes_no(), revision="clean-revision")
    log.put_record(removed)
    log.put_record(clean)
    erase = Erasure({"probe": cycle.store}, calibrations=log)
    original = cycle.store.revisions.delete

    def interrupted(ident):
        assert cycle.store.revisions.serving().id == cycle.store.revisions.base.id
        assert removed.id not in {item.id for item in log.records()}
        with pytest.raises(KeyError):
            log.restore_record(removed.id)
        raise RuntimeError("interrupted after withdrawal")

    monkeypatch.setattr(cycle.store.revisions, "delete", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        erase.erase({"case-7"})
    monkeypatch.setattr(cycle.store.revisions, "delete", original)
    assert erase.erase({"case-7"}).clean
    assert [item.id for item in log.records()] == [clean.id]
    assert not _files_containing(cycle.store.root, ["case-7"])


@pytest.mark.parametrize("external", [False, True])
def test_erasure_invalidates_cached_calibrations_without_waiting(tmp_path, external):
    from types import SimpleNamespace
    import subprocess
    import sys

    from colonymind.instinct.log import Store
    from colonymind.instinct.tiers import Instinct
    from test_instinct_fit import record, yes_no

    # Fixture assumption: the existing synthetic cycle's row count is enough to promote.
    cycle = _cycle(tmp_path / "weights", "probe", make_rows(60))
    closure, _, _ = cycle.step()
    point = yes_no()
    removed = record(point, revision=closure.state)
    clean = record(point, revision="clean-revision")
    log = Store(tmp_path / "instinct.db")
    log.put_record(removed)
    log.put_record(clean)
    provider = SimpleNamespace(kind=removed.provider_kind, model=removed.model)
    # A fixed clock cannot reach the refresh deadline. Separate Store objects share the same database.
    engines = [Instinct(store=store, records=(), monotonic=lambda: NOW.timestamp())
               for store in (log, Store(log.path))]
    for engine in engines:
        assert engine.record_for(point, "fast", provider, closure.state) == removed
        assert engine.record_for(point, "fast", provider, clean.revision) == clean
    if external:
        subprocess.run([sys.executable, "-c",
                        "from pathlib import Path; import sys; "
                        "from colonymind.instinct.log import Store; "
                        "Store(Path(sys.argv[1])).withdraw_revisions({sys.argv[2]})",
                        str(log.path), closure.state], check=True)
    else:
        Erasure({"probe": cycle.store}, calibrations=Store(log.path)).erase({"case-7"})
    for engine in engines:
        assert engine.record_for(point, "fast", provider, closure.state) is None
        assert engine.record_for(point, "fast", provider, clean.revision) == clean


def test_retry_rebuilds_after_a_crash_after_the_manifests_were_rewritten(tmp_path):
    cycle = _cycle(tmp_path, "probe", make_rows(60))
    contaminated, _, _ = cycle.step()
    cycle.source.erase({"case-7"})

    def interrupted():
        raise RuntimeError("interrupted before rebuild")

    with pytest.raises(RuntimeError, match="interrupted"):
        Erasure({"probe": cycle.store}).erase({"case-7"}, rebuild={"probe": interrupted})
    assert cycle.store.revisions.serving().id == cycle.store.revisions.base.id
    resumed = Erasure({"probe": cycle.store}).erase({"case-7"}, rebuild={"probe": cycle.rebuild})
    assert resumed.clean and ("rebuild", "probe") in resumed.events
    assert cycle.store.revisions.serving().id not in {cycle.store.revisions.base.id, contaminated.id}
    assert not _files_containing(cycle.store.root, ["case-7"])


def test_unserve_happens_before_the_rebuild(tmp_path):
    cycle = _cycle(tmp_path, "probe", make_rows(60))
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(30, prefix="late"))
    second, _, _ = cycle.step()
    seen = []

    def rebuild():
        seen.append(cycle.store.revisions.serving().id)
        return cycle.rebuild()

    cycle.source.erase({"late-2"})
    result = Erasure({"probe": cycle.store}).erase({"late-2"}, readmit={"probe": (cycle.admission, cycle.source)},
                                                   rebuild={"probe": rebuild})
    assert seen == [first.id]  # the contaminated revision stopped serving before any rebuild began
    steps = [step for step, _ in result.events]
    assert steps.index("unserve") < steps.index("delete") < steps.index("rebuild")
    assert result.fallbacks == {"probe": first.id}
    assert result.contaminated == {"probe": (second.id,)}
    assert cycle.store.revisions.serving().id not in (first.id, second.id)  # the clean rebuild now serves


def test_with_no_retained_closure_free_of_the_row_the_base_serves(tmp_path):
    cycle = _cycle(tmp_path, "probe", make_rows(60))
    cycle.step()
    cycle.source.erase({"case-7"})
    result = Erasure({"probe": cycle.store}).erase({"case-7"})
    assert cycle.store.revisions.serving().id == cycle.store.revisions.base.id
    assert result.fallbacks == {"probe": cycle.store.revisions.base.id}


def test_a_withdrawn_closure_is_never_a_fallback(tmp_path):
    cycle = _cycle(tmp_path, "probe", make_rows(60))
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(20, prefix="failed"))
    dropped, _, outcome = cycle.step(calibration=())
    assert outcome == "dropped"
    cycle.source.add(make_rows(20, prefix="later"))
    third, _, outcome = cycle.step()
    assert outcome == "promoted"
    cycle.source.erase({"later-3"})
    result = Erasure({"probe": cycle.store}).erase({"later-3"})
    # The dropped candidate is newer and clean of the row, but it failed its gates: it never serves.
    assert result.fallbacks == {"probe": first.id}
    assert cycle.store.revisions.serving().id == first.id


def test_a_chain_falls_back_as_one(tmp_path):
    joints = JointLedger(tmp_path / "chains")
    first = _cycle(tmp_path, "first", make_rows(60, prefix="a"))
    second = _cycle(tmp_path, "second", make_rows(60, prefix="c"))
    one_first, _, _ = first.step()
    one_second, _, _ = second.step()
    joints.record("pair", {"first": one_first.id, "second": one_second.id})
    first.source.add(make_rows(30, prefix="b"))
    second.source.add(make_rows(30, prefix="d"))
    two_first, _, _ = first.step()
    two_second, _, _ = second.step()  # clean: it never saw the erased row
    joints.record("pair", {"first": two_first.id, "second": two_second.id})

    first.source.erase({"b-4"})
    result = Erasure({"first": first.store, "second": second.store}, chains=[Chain("pair", ("first", "second"))],
                     joints=joints).erase({"b-4"})
    assert first.store.revisions.serving().id == one_first.id
    assert second.store.revisions.serving().id == one_second.id  # never a mix of the two joint closures
    assert result.fallbacks == {"first": one_first.id, "second": one_second.id}
    assert [record["members"]["first"] for record in joints.records("pair")] == [one_first.id]


def test_a_chain_with_no_clean_joint_closure_falls_back_to_the_base_together(tmp_path):
    joints = JointLedger(tmp_path / "chains")
    first = _cycle(tmp_path, "first", make_rows(60, prefix="a"))
    second = _cycle(tmp_path, "second", make_rows(60, prefix="c"))
    one_first, _, _ = first.step()
    one_second, _, _ = second.step()
    joints.record("pair", {"first": one_first.id, "second": one_second.id})
    first.source.erase({"a-1"})
    Erasure({"first": first.store, "second": second.store}, chains=[Chain("pair", ("first", "second"))],
            joints=joints).erase({"a-1"})
    assert first.store.revisions.serving().id == first.store.revisions.base.id
    assert second.store.revisions.serving().id == second.store.revisions.base.id


def _pair(tmp_path, *, audience="owner"):
    joints = JointLedger(tmp_path / "chains")
    first = _cycle(tmp_path, "first", make_rows(60, prefix="a", audience=audience))
    second = _cycle(tmp_path, "second", make_rows(60, prefix="c", audience=audience))
    one = {"first": first.step()[0].id, "second": second.step()[0].id}
    joints.record("pair", one)
    first.source.add(make_rows(30, prefix="b", audience=audience))
    second.source.add(make_rows(30, prefix="d", audience=audience))
    two = {"first": first.step()[0].id, "second": second.step()[0].id}
    joints.record("pair", two)
    return joints, first, second, one, two


def _serving(*cycles):
    return tuple(cycle.store.revisions.serving().id for cycle in cycles)


def _pinned(joints, first, second, requester="owner"):
    pins = joints.pin(Chain("pair", ("first", "second")),
                      {"first": first.store.revisions, "second": second.store.revisions}, requester)
    return pins["first"].closure.id, pins["second"].closure.id


def test_a_chain_interrupted_between_its_members_serves_no_mix_and_a_retry_finishes_the_fallback(tmp_path,
                                                                                               monkeypatch):
    joints, first, second, one, two = _pair(tmp_path)
    bases = (first.store.revisions.base.id, second.store.revisions.base.id)
    first.source.erase({"b-4"})
    erasure = Erasure({"first": first.store, "second": second.store}, chains=[Chain("pair", ("first", "second"))],
                      joints=joints)

    def crash(*args, **kwargs):
        raise OSError("power lost")

    monkeypatch.setattr(second.store.revisions, "promote", crash)
    monkeypatch.setattr(second.store.revisions, "unserve", crash)
    with pytest.raises(OSError):
        erasure.erase({"b-4"})
    monkeypatch.undo()
    assert _serving(first, second) == (one["first"], two["second"])  # the pointers are a mix on disk ...
    assert _pinned(joints, first, second) == bases  # ... but a request of the chain reads its bases, never the mix

    result = erasure.erase({"b-4"})
    assert result.clean, result.left
    assert _serving(first, second) == (one["first"], one["second"])
    assert _pinned(joints, first, second) == (one["first"], one["second"])
    assert joints.pending("pair") is None
    assert [record["members"] for record in joints.records("pair")] == [one]


def test_an_erasure_of_no_refs_finishes_a_chain_move_a_crash_interrupted(tmp_path, monkeypatch):
    joints, first, second, one, two = _pair(tmp_path)
    first.source.erase({"b-4"})
    chains = [Chain("pair", ("first", "second"))]
    stores = {"first": first.store, "second": second.store}

    def crash(*args, **kwargs):
        raise OSError("power lost")

    monkeypatch.setattr(second.store.revisions, "promote", crash)
    monkeypatch.setattr(second.store.revisions, "unserve", crash)
    with pytest.raises(OSError):
        Erasure(stores, chains=chains, joints=joints).erase({"b-4"})
    monkeypatch.undo()
    Erasure(stores, chains=chains, joints=joints).erase(())
    # Never back to the joint record the interrupted move left as contaminated, although this erasure names no ref.
    assert _serving(first, second) == (one["first"], one["second"]) and joints.pending("pair") is None


def test_a_chain_is_read_as_one_recorded_joint_closure_or_as_its_bases(tmp_path):
    joints, first, second, one, two = _pair(tmp_path)
    bases = (first.store.revisions.base.id, second.store.revisions.base.id)
    assert _pinned(joints, first, second) == (two["first"], two["second"])
    first.store.revisions.promote(one["first"])  # a member moved alone: a tuple no joint record names
    assert _pinned(joints, first, second) == bases
    second.store.revisions.promote(one["second"])
    assert _pinned(joints, first, second) == (one["first"], one["second"])
    assert _pinned(joints, first, second, requester="guest") == bases  # an owner chain serves owner turns only
    first.store.revisions.unserve()
    second.store.revisions.unserve()
    assert _pinned(joints, first, second) == bases


def test_an_erasure_repairs_a_chain_left_as_a_mix_whatever_it_erases(tmp_path):
    joints, first, second, one, two = _pair(tmp_path)
    first.store.revisions.promote(one["first"])  # what a crash between two member swaps leaves
    Erasure({"first": first.store, "second": second.store}, chains=[Chain("pair", ("first", "second"))],
            joints=joints).erase({"never-seen"})
    assert _serving(first, second) == (two["first"], two["second"])


def test_no_file_containing_the_row_remains(tmp_path):
    cycle = _cycle(tmp_path, "probe", make_rows(60))
    cycle.step()
    cycle.source.add(make_rows(30, prefix="late"))
    cycle.step()
    erased = {"late-2", "case-9"}
    hashes = {state_hash("synthetic_yes_no late 2"), state_hash("synthetic_yes_no case 9")}
    assert _files_containing(tmp_path, erased | hashes)
    cycle.source.erase(erased)
    result = Erasure({"probe": cycle.store}).erase(erased, readmit={"probe": (cycle.admission, cycle.source)},
                                                   rebuild={"probe": cycle.rebuild})
    assert result.clean and result.left == ()
    assert _files_containing(tmp_path, erased | hashes) == []
    assert scan(tmp_path, erased | hashes) == ()
    for path in (tmp_path / "learners" / "probe" / "erasures").glob("*.json"):
        record = json.loads(path.read_text())
        assert record["refs"] == 2 and "late-2" not in path.read_text()


def test_a_state_hash_another_surviving_row_shares_is_not_a_contaminated_needle(tmp_path):
    rows = make_rows(20) + make_rows(3, refs="again", at=(EPOCH + timedelta(hours=1)).isoformat())
    cycle = _cycle(tmp_path, "probe", rows)
    cycle.step()
    cycle.source.erase({"case-1"})
    result = Erasure({"probe": cycle.store}).erase({"case-1"}, readmit={"probe": (cycle.admission, cycle.source)},
                                                   rebuild={"probe": cycle.rebuild})
    assert result.clean
    # again-1 has the same state as the erased row and still trains: its hash stays, the erased ref does not.
    assert any("again-1" in manifest.refs() for manifest in cycle.store.current_manifests())
    assert not any("case-1" in manifest.refs() for manifest in cycle.store.current_manifests())


def test_erasing_a_ref_nothing_learned_from_changes_nothing(tmp_path):
    cycle = _cycle(tmp_path, "probe", make_rows(40))
    first, _, _ = cycle.step()
    result = Erasure({"probe": cycle.store}).erase({"never-seen"})
    assert result.contaminated == {} and result.fallbacks == {} and result.clean
    assert cycle.store.revisions.serving().id == first.id


def test_a_manifest_an_earlier_erasure_rewrote_still_reaches_the_closures_built_on_it(tmp_path):
    # A text row under a day old is named by the period's manifest without training. Erasing it rewrites that
    # manifest under a new id while the serving closure still names the old one; a later erasure of a row the
    # closure did learn from must still reach it.
    young = make_rows(1, prefix="young", at=(NOW - timedelta(hours=1)).isoformat())
    cycle = _cycle(tmp_path, "probe", make_rows(60) + young)
    first, _, _ = cycle.step()
    [manifest] = cycle.store.current_manifests()
    assert {entry.ref: entry.verdict for entry in manifest.entries}["young-0"] == "quarantine"
    cycle.source.erase({"young-0"})
    result = Erasure({"probe": cycle.store}).erase({"young-0"})
    assert result.contaminated == {} and cycle.store.revisions.serving().id == first.id
    assert first.manifests != tuple(manifest.id for manifest in cycle.store.current_manifests())

    cycle.source.erase({"case-4"})
    result = Erasure({"probe": cycle.store}).erase({"case-4"})
    assert result.contaminated == {"probe": (first.id,)}
    assert cycle.store.revisions.serving().id == cycle.store.revisions.base.id


def test_a_stacked_learner_is_unserved_when_the_closure_that_first_learned_the_row_was_pruned(tmp_path):
    store = LearnerStore(tmp_path, "stacked", base="synthetic-base@1", mode="stack", declared=("threshold",),
                         keep=2, clock=lambda: NOW)
    cycle = Cycle(store, Controller(store, live=False, clock=lambda: NOW),
                  source=SyntheticSource(make_rows(40, prefix="n0")), clock=lambda: NOW)
    cycle.step()
    for night in range(1, 4):
        cycle.source.add(make_rows(40, prefix=f"n{night}"))
        cycle.step()
    serving = store.revisions.serving()
    assert len(store.revisions.retained()) == 2  # the closures of the first two nights were pruned
    cycle.source.erase({"n0-3"})
    result = Erasure({"stacked": store}).erase({"n0-3"})
    assert serving.id in result.contaminated["stacked"]
    assert store.revisions.serving().id == store.revisions.base.id
    assert [period.state for period in store.periods.periods()] == ["discarded"] * 4
    assert store.revisions.retained() == []


def test_a_downstream_closure_is_reached_when_its_upstream_closure_was_pruned(tmp_path):
    parent_store = LearnerStore(tmp_path, "parent", base="synthetic-base@1", declared=("threshold",), keep=2,
                                clock=lambda: NOW)
    parent = Cycle(parent_store, Controller(parent_store, live=False, clock=lambda: NOW),
                   source=SyntheticSource(make_rows(60, prefix="p")), clock=lambda: NOW)
    upstream, _, _ = parent.step()
    child = LearnerStore(tmp_path, "child", base="synthetic-base@1", declared=("threshold",), clock=lambda: NOW)
    state = {"w": np.ones(3)}
    closure = child.revisions.compose(state=builds.state_digest(state), objective="o", calibration=("c",),
                                      dependents={"threshold": "t"}, parent=child.revisions.base.id,
                                      upstream=(("parent", upstream.id),), audience="owner", mode="rebuild")
    child.revisions.write(closure, state)
    child.revisions.promote(closure.id)
    child.revisions.set_status(closure.id, "serving")
    for night in range(2):
        parent.source.add(make_rows(20, prefix=f"q{night}"))
        parent.step()
    assert upstream.id not in {found.id for found in parent_store.revisions.retained()}

    parent.source.erase({"p-5"})
    result = Erasure({"parent": parent_store, "child": child}).erase({"p-5"})
    assert closure.id in result.contaminated["child"]
    assert child.revisions.serving().id == child.revisions.base.id


def _downstream(root, name, upstream_learner, upstream):
    """A learner holding one serving closure learned from ``upstream``, a closure of ``upstream_learner``."""
    store = LearnerStore(root, name, base="synthetic-base@1", declared=("threshold",), clock=lambda: NOW)
    state = {"w": np.full(3, float(sum(map(ord, name))))}
    closure = store.revisions.compose(state=builds.state_digest(state), objective="o", calibration=("c",),
                                      dependents={"threshold": "t"}, parent=store.revisions.base.id,
                                      upstream=((upstream_learner, upstream.id),), audience="owner", mode="rebuild")
    store.revisions.write(closure, state)
    store.revisions.promote(closure.id)
    store.revisions.set_status(closure.id, "serving")
    return store, closure


@pytest.mark.parametrize("crash_after", [1, 2])
def test_a_crash_between_deletions_never_strands_a_descendant(tmp_path, monkeypatch, crash_after):
    # a_up learned the row; b_mid learned from a_up's closure, and c_low from b_mid's. The learners' names sort
    # upstream first, so a delete in name order removes the link to a descendant before the descendant itself.
    up = _cycle(tmp_path, "a_up", make_rows(60, prefix="u"))
    root, _, _ = up.step()
    mid, middle = _downstream(tmp_path, "b_mid", "a_up", root)
    low, lowest = _downstream(tmp_path, "c_low", "b_mid", middle)
    stores = {"a_up": up.store, "b_mid": mid, "c_low": low}
    up.source.erase({"u-3"})
    deleted = []
    delete = RevisionStore.delete

    def crash(self, closure_id):
        if len(deleted) == crash_after:
            raise OSError("power lost")
        delete(self, closure_id)
        deleted.append(closure_id)

    monkeypatch.setattr(RevisionStore, "delete", crash)
    with pytest.raises(OSError):
        Erasure(stores).erase({"u-3"})
    monkeypatch.undo()
    assert len(deleted) == crash_after

    result = Erasure(stores).erase({"u-3"})
    assert result.clean, result.left
    for store, closure in ((up.store, root), (mid, middle), (low, lowest)):
        assert store.revisions.find(closure.id) is None
        assert not store.revisions.state_path(closure.state).exists()
        assert store.revisions.serving().id == store.revisions.base.id


def test_a_retried_erasure_leaves_no_contaminated_state_after_a_crash(tmp_path, monkeypatch):
    cycle = _cycle(tmp_path, "probe", make_rows(60))
    first, _, _ = cycle.step()
    state_file = cycle.store.revisions.state_path(first.state)
    remove = _io.remove

    def crash(path):
        if Path(path) == state_file:
            raise OSError("power lost")
        return remove(path)

    cycle.source.erase({"case-7"})
    monkeypatch.setattr(_io, "remove", crash)
    with pytest.raises(OSError):
        Erasure({"probe": cycle.store}).erase({"case-7"})
    monkeypatch.undo()
    assert cycle.store.revisions.serving().id == cycle.store.revisions.base.id  # unserved before the crash
    assert state_file.exists() and not cycle.store.revisions.exists(first.id)

    result = Erasure({"probe": cycle.store}).erase({"case-7"})
    assert result.clean, result.left
    assert not state_file.exists()


def test_a_contaminated_state_left_on_disk_is_reported(tmp_path, monkeypatch):
    cycle = _cycle(tmp_path, "probe", make_rows(60))
    first, _, _ = cycle.step()
    state_file = cycle.store.revisions.state_path(first.state)
    remove = _io.remove
    monkeypatch.setattr(_io, "remove", lambda path: False if Path(path) == state_file else remove(path))
    cycle.source.erase({"case-7"})
    result = Erasure({"probe": cycle.store}).erase({"case-7"})
    assert not result.clean and str(state_file) in result.left
