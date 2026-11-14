"""Revisions are closures (docs/PLASTICITY.md, Revisions): one id over the base identity, the learned state's
digest, the calibration record ids and the declared dependents. Promotion swaps one pointer atomically, a request
reads one whole closure, and the newest 14 closures are kept."""

from __future__ import annotations

import json

import numpy as np
import pytest

from colonymind.plasticity import _io, builds
from colonymind.plasticity.admission import Admission
from colonymind.plasticity.erasure import Erasure
from colonymind.plasticity.revisions import KEEP, RevisionError, RevisionStore
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import EPOCH, OBJECTIVE, make_rows


def _store(root, **options):
    options.setdefault("declared", ("threshold",))
    return RevisionStore(root, "probe", base="base@1", **options)


def _state(value):
    return {"w": np.full(4, float(value))}


def _write(store, value, *, dependents=None, calibration=None, parent=None):
    state = _state(value)
    digest = builds.state_digest(state)
    parent = parent if parent is not None else store.serving().id
    closure = store.compose(state=digest, objective="objective@1", calibration=calibration or (f"cal-{value}",),
                            dependents={"threshold": f"t{value}"} if dependents is None else dependents,
                            parent=parent, audience="owner", mode="rebuild")
    return store.write(closure, state)


def test_the_base_serves_until_something_is_promoted(tmp_path):
    store = _store(tmp_path)
    assert store.serving().id == store.base.id and store.base.mode == "base"
    assert store.state(store.base) == {}


def test_a_withdrawn_identity_never_returns_after_pruning_or_rewriting(tmp_path):
    store = _store(tmp_path, keep=2)
    failed = _write(store, 1)
    store.set_status(failed.id, "withdrawn", reason="regression")
    with pytest.raises(RevisionError, match="withdrawn"):
        store.promote(failed.id)
    for value in range(2, 6):
        _write(store, value)
    assert not store.exists(failed.id)
    reopened = _store(tmp_path, keep=2)
    assert reopened.status(failed.id).state == "withdrawn"
    with pytest.raises(RevisionError, match="withdrawn"):
        reopened.write(failed, _state(1))
    with pytest.raises(RevisionError, match="withdrawn"):
        reopened.set_status(failed.id, "candidate")


def test_a_crash_after_withdrawal_never_serves_the_failed_pointer(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    second = _write(store, 2)
    store.promote(second.id, previous=first.id)
    store.set_status(second.id, "withdrawn", reason="regression")
    reopened = _store(tmp_path)
    assert reopened.pin("owner").closure.id == first.id


def test_a_small_store_keeps_its_live_comparator_and_new_candidate(tmp_path):
    store = _store(tmp_path, keep=2)
    first = _write(store, 1)
    store.promote(first.id)
    second = _write(store, 2)
    store.promote(second.id, previous=first.id)
    third = _write(store, 3)
    assert store.exists(first.id) and store.exists(third.id)
    store.promote(third.id, previous=second.id)
    assert store.exists(second.id) and not store.exists(first.id)


def test_promotion_is_one_atomic_pointer_swap(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    second = _write(store, 2)
    # A crash between the write and the swap: the process is gone before promote runs.
    reopened = _store(tmp_path)
    assert reopened.serving().id == first.id
    assert reopened.get(second.id) == second  # written whole, waiting


def test_a_crash_during_the_swap_leaves_the_old_revision_serving(tmp_path, monkeypatch):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    second = _write(store, 2)

    def crash(source, destination):
        raise OSError("power lost")

    monkeypatch.setattr(_io, "_replace", crash)
    with pytest.raises(OSError):
        store.promote(second.id)
    monkeypatch.undo()
    assert _store(tmp_path).serving().id == first.id
    assert not [path for path in tmp_path.rglob("*.tmp")]


def test_a_crash_while_writing_a_closure_leaves_no_closure_and_the_old_one_serving(tmp_path, monkeypatch):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    calls = []

    def crash(source, destination):
        calls.append(destination)
        raise OSError("power lost")

    monkeypatch.setattr(_io, "_replace", crash)
    with pytest.raises(OSError):
        _write(store, 2)
    monkeypatch.undo()
    reopened = _store(tmp_path)
    assert reopened.serving().id == first.id and len(reopened.retained()) == 1


def test_no_reader_sees_a_half_written_closure(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    pinned = store.pin("owner")
    second = _write(store, 2)
    path = store.path_of(second.id)
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
    with pytest.raises(RevisionError):
        store.get(second.id)
    with pytest.raises(RevisionError):
        store.promote(second.id)
    # A request holds one whole closure: a later promotion never changes what it reads.
    third = _write(store, 3)
    store.promote(third.id)
    assert pinned.closure.id == first.id and np.array_equal(pinned.state["w"], _state(1)["w"])
    assert store.pin("owner").closure.id == third.id


def test_a_closure_whose_content_does_not_match_its_id_is_refused(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    path = store.path_of(first.id)
    record = json.loads(path.read_text())
    record["closure"]["calibration"] = ["cal-forged"]
    path.write_text(json.dumps(record))
    with pytest.raises(RevisionError, match="id"):
        store.get(first.id)


def test_a_state_file_that_does_not_match_its_digest_is_refused(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.state_path(first.state).write_bytes(_io.save_arrays(_state(9)))
    with pytest.raises(RevisionError, match="digest"):
        store.state(first)


def test_a_revision_that_changes_a_key_it_did_not_declare_is_refused(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    with pytest.raises(RevisionError, match="cutoff"):
        _write(store, 2, dependents={"threshold": "t2", "cutoff": "0.7"})
    with pytest.raises(RevisionError, match="threshold"):
        _write(store, 3, dependents={})  # dropping a key changes it too
    _write(store, 4, dependents={"threshold": "t4"})


def test_the_closure_id_covers_base_state_calibration_and_dependents(tmp_path):
    store = _store(tmp_path)
    fields = dict(state="s1", objective="objective@1", calibration=("cal-1",), dependents={"threshold": "t1"},
                  parent=store.base.id, audience="owner", mode="rebuild")
    first = store.compose(**fields)
    assert store.compose(**fields).id == first.id
    for change in ({"state": "s2"}, {"calibration": ("cal-2",)}, {"dependents": {"threshold": "t2"}},
                   {"objective": "objective@2"}):
        assert store.compose(**{**fields, **change}).id != first.id, change
    assert RevisionStore(tmp_path / "other", "probe", base="base@2").compose(**fields).id != first.id


def test_the_newest_14_closures_are_kept_and_the_oldest_dropped(tmp_path):
    store = _store(tmp_path)
    assert KEEP == 14
    written = [_write(store, value, parent=store.base.id) for value in range(1, 16)]
    kept = store.retained()
    assert [closure.id for closure in kept] == [closure.id for closure in written[1:]]
    assert not store.path_of(written[0].id).exists()
    assert not store.state_path(written[0].state).exists()


def test_the_serving_closure_is_never_dropped_by_retention(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    for value in range(2, 18):
        _write(store, value, parent=store.base.id)
    assert store.serving().id == first.id and store.path_of(first.id).exists()
    assert len(store.retained()) == KEEP


def test_the_serving_closure_cannot_be_deleted_while_it_serves(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    with pytest.raises(RevisionError, match="serving"):
        store.delete(first.id)
    store.unserve()
    store.delete(first.id)
    assert store.serving().id == store.base.id


def test_a_new_base_serves_the_base_never_a_closure_learned_on_the_old_one(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    swapped = RevisionStore(tmp_path, "probe", base="base@2", declared=("threshold",))
    assert swapped.serving().id == swapped.base.id != store.base.id
    assert swapped.pin("owner").closure == swapped.base
    assert swapped.retained() == [] and swapped.find(first.id) is None
    # What was learned on the old base is retired: its state is an orphan the next sweep removes.
    assert swapped.orphans() == [swapped.state_path(first.state)]


def test_a_pointer_to_a_closure_that_is_not_whole_is_an_error_not_a_silent_base(tmp_path):
    store = _store(tmp_path)
    first = _write(store, 1)
    store.promote(first.id)
    path = store.path_of(first.id)
    path.write_bytes(path.read_bytes()[:10])
    with pytest.raises(RevisionError):
        store.serving()


def test_a_second_derivation_of_one_closure_keeps_the_first_whose_audience_and_rows_answer_for_what_serves(tmp_path):
    # The id covers everything that serves (base, state digest, objective, calibration, dependents), so two builds
    # of identical bytes are one closure, and the record kept is the first derivation. It answers for exactly the
    # bytes that serve: a guest reads what guest rows alone produced, and erasing a row only the second derivation
    # learned from leaves serving what a derivation without it produced.
    now = EPOCH.replace(day=11)
    store = LearnerStore(tmp_path, "probe", base="base@1", declared=("threshold",), clock=lambda: now)
    admission = Admission(OBJECTIVE, learner="probe", source_of=lambda row: None, clock=lambda: now)
    manifests, periods = [], []
    for rows in (make_rows(3, prefix="g", audience="guest"), make_rows(3, prefix="o", audience="owner")):
        period = store.periods.open()
        manifest = admission.admit(rows, period=period.id).manifest
        store.periods.add_manifest(period.id, store.manifests.put(manifest))
        store.periods.close(period.id)
        manifests.append(manifest.id)
        periods.append(period.id)
    state = _state(7)
    revisions = store.revisions

    def derivation(count, audience):
        return revisions.compose(state=builds.state_digest(state), objective="objective@1", calibration=("cal-7",),
                                 dependents={"threshold": "t7"}, parent=revisions.base.id,
                                 manifests=manifests[:count], periods=periods[:count], audience=audience)

    guest, owner = derivation(1, "guest"), derivation(2, "owner")
    assert guest.id == owner.id
    revisions.write(guest, state)
    revisions.write(owner, state)
    revisions.promote(owner.id)
    kept = revisions.get(owner.id)
    assert kept.audience == "guest" and kept.manifests == (manifests[0],)
    assert _io.state_digest(revisions.pin("guest").state) == guest.state  # the guest-only derivation's bytes

    result = Erasure({"probe": store}).erase({"o-1"})
    assert result.clean and result.contaminated == {}
    assert revisions.serving().id == guest.id and revisions.serving().state == guest.state

    result = Erasure({"probe": store}).erase({"g-1"})
    assert result.contaminated == {"probe": (guest.id,)} and revisions.serving().id == revisions.base.id
