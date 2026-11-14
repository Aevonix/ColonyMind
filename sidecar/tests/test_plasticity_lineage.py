"""Revision lineage (docs/PLASTICITY.md, Lineage): a child's closure names its parent, the chain is walkable, and
erasing a parent's row reaches the child. The carry hook is where the owner's per-point decision state follows a
learned child that passed its gates."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from colonymind.plasticity import builds
from colonymind.plasticity.erasure import Erasure
from colonymind.plasticity.lineage import Lineage
from colonymind.plasticity.states import Controller
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import EPOCH, Cycle, SyntheticSource, make_rows

NOW = EPOCH + timedelta(days=10)


def _cycle(root, name, rows, *, mode="rebuild", lineage=None):
    store = LearnerStore(root, name, base="synthetic-base@1", mode=mode, declared=("threshold",), clock=lambda: NOW)
    controller = Controller(store, live=False, lineage=lineage, clock=lambda: NOW)
    return Cycle(store, controller, source=SyntheticSource(list(rows)), clock=lambda: NOW)


def test_a_childs_closure_names_its_parent_and_the_chain_is_walkable(tmp_path):
    cycle = _cycle(tmp_path, "stacked", make_rows(40, prefix="one"), mode="stack")
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="two"))
    second, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="three"))
    third, _, _ = cycle.step()
    lineage = Lineage({"stacked": cycle.store.revisions})
    assert third.parent == second.id and second.parent == first.id
    assert first.parent == cycle.store.revisions.base.id
    assert [closure.id for closure in lineage.chain(third)] == [third.id, second.id, first.id,
                                                                cycle.store.revisions.base.id]
    assert lineage.parent(third) == second
    assert {closure.id for closure in lineage.children(first)} == {second.id}


def test_erasing_a_parents_row_reaches_the_child(tmp_path):
    parent = _cycle(tmp_path, "parent", make_rows(60, prefix="p"))
    upstream, _, _ = parent.step()
    child = LearnerStore(tmp_path, "child", base="synthetic-base@1", declared=("threshold",), clock=lambda: NOW)
    state = {"w": np.ones(3)}
    # The child learned from the parent's outputs, not from the row itself: its own manifests never name it.
    closure = child.revisions.compose(state=builds.state_digest(state), objective="o", calibration=("c",),
                                      dependents={"threshold": "t"}, parent=child.revisions.base.id,
                                      upstream=(("parent", upstream.id),), audience="owner", mode="rebuild")
    child.revisions.write(closure, state)
    child.revisions.promote(closure.id)
    child.revisions.set_status(closure.id, "serving")
    lineage = Lineage({"parent": parent.store.revisions, "child": child.revisions})
    assert {(name, found.id) for name, found in lineage.descendants({("parent", upstream.id)})} == {
        ("parent", upstream.id), ("child", closure.id)}

    parent.source.erase({"p-5"})
    rebuilt = []
    result = Erasure({"parent": parent.store, "child": child}).erase(
        {"p-5"}, rebuild={"child": lambda: rebuilt.append(child.revisions.serving().id)})
    assert result.contaminated == {"parent": (upstream.id,), "child": (closure.id,)}
    assert child.revisions.serving().id == child.revisions.base.id
    assert not child.revisions.path_of(closure.id).exists()
    assert rebuilt == [child.revisions.base.id]


def test_the_carry_hook_runs_when_a_child_that_passed_its_gates_is_promoted(tmp_path):
    calls = []
    cycle = _cycle(tmp_path, "probe", make_rows(60))
    lineage = Lineage({"probe": cycle.store.revisions})
    lineage.add_hook(lambda parent, child: calls.append((parent.id, child.id)))
    cycle.controller.lineage = lineage
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(20, prefix="more"))
    second, _, _ = cycle.step()
    cycle.source.add(make_rows(20, prefix="failed"))
    cycle.step(calibration=())  # fails its gates: nothing carries
    assert calls == [(cycle.store.revisions.base.id, first.id), (first.id, second.id)]


async def test_binding_instinct_carries_the_child_calibration_before_its_first_answer(tmp_path):
    from colonymind.instinct import state
    from test_instinct_lineage import Live

    live = await Live(tmp_path / "instinct").promoted()
    cycle = _cycle(tmp_path / "plasticity", "probe", make_rows(60))
    child = live.child("rev-2", "rev-1")
    live.frozen("rev-2")
    lineage = Lineage({"probe": cycle.store.revisions})
    lineage.bind_instinct(live.store, registry=live.registry, clock=lambda: NOW)
    cycle.controller.lineage = lineage
    cycle.refit = lambda digest: (child.id,)
    cycle.step()
    latest = live.store.transitions()[0]
    assert latest["source"].startswith(state.CARRIED)
    assert latest["evidence"]["fingerprint"] == child.fingerprint
    assert live.effective() == "live"


async def test_binding_instinct_demotes_when_the_child_cannot_carry(tmp_path):
    from test_instinct_lineage import Live

    live = await Live(tmp_path / "instinct").promoted()
    cycle = _cycle(tmp_path / "plasticity", "probe", make_rows(60))
    child = live.child("rev-2", "rev-1")
    lineage = Lineage({"probe": cycle.store.revisions})
    lineage.bind_instinct(live.store, registry=live.registry, clock=lambda: NOW)
    cycle.controller.lineage = lineage
    cycle.refit = lambda digest: (child.id,)
    cycle.step()
    assert live.effective() == "demoted"
    assert "no offline pass" in live.store.transitions()[0]["evidence"]["why"]
