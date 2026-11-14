"""Compaction retries retain deletion witnesses and protect every remaining manifest reference.

The tiny fixture states and retention count are test assumptions, not production settings.
"""
import numpy as np
import pytest

from colonymind.plasticity import _io
from colonymind.plasticity.compaction import compact, discarded_rows
from colonymind.plasticity.manifests import Entry, Manifest
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import EPOCH, OBJECTIVE, make_rows


def manifest(store, *, attached=True):
    period = store.periods.open()
    row = make_rows(1)[0]
    value = Manifest(learner=store.name, objective=OBJECTIVE.name, objective_identity=OBJECTIVE.identity,
                     period=period.id, admitted_at=EPOCH.isoformat(), entries=(Entry.of(row, "admitted"),))
    store.manifests.put(value)
    if attached:
        store.periods.add_manifest(period.id, value.id)
    store.periods.close(period.id)
    return period.id, value


def store_at(path, *, keep=14):
    return LearnerStore(path, "sample", base="frozen-base@1", keep=keep, clock=lambda: EPOCH)


def test_compaction_retry_sweeps_manifest_unlinked_before_interruption(tmp_path, monkeypatch):
    store = store_at(tmp_path)
    period, value = manifest(store)
    store.periods.discard(period, reason="failed gate")
    delete = store.manifests.delete

    def interrupted(ident):
        assert ident == value.id
        assert store.periods.get(period).manifests == ()
        assert _io.digest([value.entries[0].ref, value.entries[0].teacher]) in discarded_rows(store)
        raise OSError("interrupted after detaching manifest")

    monkeypatch.setattr(store.manifests, "delete", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        compact(store)
    monkeypatch.setattr(store.manifests, "delete", delete)
    assert store.manifests.ids() == [value.id]
    assert compact(store)["manifests_pruned"] == 1
    assert store.manifests.ids() == []
    assert _io.digest([value.entries[0].ref, value.entries[0].teacher]) in discarded_rows(store)


def test_orphan_is_not_deleted_before_its_exclusion_witness_is_durable(tmp_path, monkeypatch):
    store = store_at(tmp_path)
    _, value = manifest(store, attached=False)
    write = _io.write_json

    def interrupted(path, data):
        if path.name == "discarded-rows.json":
            raise OSError("witness could not be persisted")
        return write(path, data)

    monkeypatch.setattr(_io, "write_json", interrupted)
    with pytest.raises(OSError, match="witness"):
        compact(store)
    assert store.manifests.ids() == [value.id]
    monkeypatch.setattr(_io, "write_json", write)
    assert compact(store)["manifests_pruned"] == 1
    assert _io.digest([value.entries[0].ref, value.entries[0].teacher]) in discarded_rows(store)


@pytest.mark.parametrize("pruned", [False, True])
def test_compaction_keeps_manifest_named_by_any_known_closure(tmp_path, pruned):
    store = store_at(tmp_path, keep=1)
    period, value = manifest(store)
    state = {"w": np.array([1.0])}
    closure = store.revisions.compose(state=_io.state_digest(state), objective=OBJECTIVE.identity,
                                      manifests=(value.id,), periods=(period,), audience="owner")
    store.revisions.write(closure, state)
    if pruned:
        other_state = {"w": np.array([2.0])}
        other = store.revisions.compose(state=_io.state_digest(other_state), objective=OBJECTIVE.identity)
        store.revisions.write(other, other_state)
        assert not store.revisions.exists(closure.id)
    store.periods.replace_manifest(period, value.id, None)
    assert closure.id in {item.id for item in store.revisions.known()}
    assert compact(store)["manifests_pruned"] == 0
    assert store.manifests.ids() == [value.id]
    assert discarded_rows(store) == set()


def test_compaction_preserves_manifest_still_named_by_another_period(tmp_path):
    store = store_at(tmp_path)
    first, value = manifest(store)
    second = store.periods.open()
    store.periods.add_manifest(second.id, value.id)
    store.periods.close(second.id)
    store.periods.discard(first, reason="failed gate")
    assert compact(store)["manifests_pruned"] == 0
    assert store.manifests.get(value.id) == value
    assert store.periods.get(first).manifests == ()
    assert store.periods.get(second.id).manifests == (value.id,)
