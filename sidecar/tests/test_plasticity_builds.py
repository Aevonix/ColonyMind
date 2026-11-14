"""The two ways to build a revision (docs/PLASTICITY.md, Builds): a clean rebuild from the base on the current
manifest, and stacked periods. Each is checked against erasure the way the design's drill states it: mode 1's rebuilt
state equals a clean retrain within 1e-5 relative; mode 2 discards back to the last clean period."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from colonymind.plasticity import builds
from colonymind.plasticity.erasure import Erasure
from colonymind.plasticity.states import Controller
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import EPOCH, OBJECTIVE, Cycle, SyntheticLearner, SyntheticSource, make_rows

NOW = EPOCH + timedelta(days=10)
LATER = (EPOCH + timedelta(hours=1)).isoformat()


def _cycle(root, name, rows, *, mode="rebuild"):
    store = LearnerStore(root, name, base="synthetic-base@1", mode=mode, declared=("threshold",), clock=lambda: NOW)
    source = SyntheticSource(rows)
    return Cycle(store, Controller(store, live=False, clock=lambda: NOW), source=source, clock=lambda: NOW)


def _relative(first, second):
    return float(np.linalg.norm(first - second) / max(np.linalg.norm(second), 1e-12))


def test_mode_1_an_erased_row_is_absent_from_the_next_build_and_it_equals_a_clean_retrain(tmp_path):
    rows = make_rows(200) + make_rows(5, refs="again", at=LATER)  # case-0..4 each have a witness
    cycle = _cycle(tmp_path / "erased", "probe", rows)
    closure, report, outcome = cycle.step()
    assert outcome == "promoted" and report.passed
    erased = {"case-3", "case-120"}
    cycle.source.erase(erased)
    result = Erasure({"probe": cycle.store}).erase(erased, readmit={"probe": (cycle.admission, cycle.source)},
                                                   rebuild={"probe": cycle.rebuild})
    assert result.clean, result.left
    serving = cycle.store.revisions.serving()
    assert serving.id != closure.id
    rebuilt = cycle.store.revisions.state(serving)
    trained = {row.ref for row in builds.training_rows(OBJECTIVE, cycle.store.current_manifests(), cycle.source)}
    assert not erased & trained and "again-3" in trained  # the witness of an erased row becomes the row

    clean = _cycle(tmp_path / "clean", "probe", [row for row in rows if row["ref"] not in erased])
    clean.step()
    retrained = clean.store.revisions.state(clean.store.revisions.serving())
    assert _relative(rebuilt["w"], retrained["w"]) <= 1e-5


def test_a_row_whose_state_no_longer_re_renders_leaves_the_build(tmp_path):
    rows = make_rows(20)
    cycle = _cycle(tmp_path, "probe", rows)
    cycle.period()
    cycle.source.alter("case-4", state="the source changed since")
    trained = {row.ref for row in builds.training_rows(OBJECTIVE, cycle.store.current_manifests(), cycle.source)}
    assert "case-4" not in trained and len(trained) == 19


def test_a_row_whose_admission_metadata_changed_since_leaves_the_build_until_it_is_admitted_again(tmp_path):
    # Admitted as guest rows from the owner's own words; since then the source re-renders some of them otherwise.
    cycle = _cycle(tmp_path, "probe", make_rows(20, audience="guest"))
    cycle.period()
    cycle.source.alter("case-4", audience="owner")  # narrowed: training it as a guest row would widen it
    cycle.source.alter("case-5", source_kind="contact")  # now a stream admission refuses as a label
    cycle.source.alter("case-6", version="v2")  # another version of its point
    cycle.source.alter("case-7", at=(EPOCH + timedelta(minutes=5)).isoformat())
    cycle.source.alter("case-8", at=EPOCH.isoformat().replace("+00:00", "Z"))  # the same time, written otherwise
    rows = builds.training_rows(OBJECTIVE, cycle.store.current_manifests(), cycle.source)
    trained = {row.ref for row in rows}
    assert not trained & {"case-4", "case-5", "case-6", "case-7"}
    assert "case-8" in trained and len(trained) == 16
    assert {row.audience for row in rows} == {"guest"}
    for ref in ("case-9", "case-10"):
        cycle.source.alter(ref, audience="owner")
    build = cycle.build()
    assert build.audience == "guest" and not build.refs & {"case-4", "case-9", "case-10"}
    # Admitted again, the narrowed rows train, and the build is an owner build.
    readmitted = cycle.admission.readmit(cycle.store.current_manifests()[0], cycle.source.rows(OBJECTIVE))
    assert {entry.audience for entry in readmitted.entries if entry.ref in ("case-4", "case-9")} == {"owner"}
    rows = builds.training_rows(OBJECTIVE, [readmitted], cycle.source)
    assert {"case-4", "case-9", "case-10"} <= {row.ref for row in rows}
    assert builds.rebuild(cycle.learner, OBJECTIVE, [readmitted], cycle.source).audience == "owner"


def test_a_rebuild_is_deterministic_and_independent_of_the_order_rows_arrive(tmp_path):
    rows = make_rows(50)
    forward = _cycle(tmp_path / "forward", "probe", rows)
    backward = _cycle(tmp_path / "backward", "probe", list(reversed(rows)))
    forward.period()
    backward.period()
    learner = SyntheticLearner()
    one = builds.rebuild(learner, OBJECTIVE, forward.store.current_manifests(), forward.source)
    two = builds.rebuild(learner, OBJECTIVE, backward.store.current_manifests(), backward.source)
    assert one.digest == two.digest and np.array_equal(one.state["w"], two.state["w"])
    assert one.audience == "owner"


def test_mode_2_erasure_discards_back_to_the_last_clean_period(tmp_path):
    cycle = _cycle(tmp_path, "stacked", [], mode="stack")
    cycle.source.add(make_rows(60, prefix="one"))
    first, _, outcome = cycle.step()
    assert outcome == "promoted"
    after_first = cycle.store.revisions.state(first)
    cycle.source.add(make_rows(60, prefix="two"))
    second, _, _ = cycle.step()
    cycle.source.add(make_rows(60, prefix="three"))
    third, _, _ = cycle.step()
    assert cycle.store.revisions.serving().id == third.id
    assert not np.array_equal(cycle.store.revisions.state(third)["w"], after_first["w"])

    cycle.source.erase({"two-5"})
    result = Erasure({"stacked": cycle.store}).erase({"two-5"})
    assert result.clean, result.left
    assert cycle.store.revisions.serving().id == first.id
    assert np.array_equal(cycle.store.revisions.state(cycle.store.revisions.serving())["w"], after_first["w"])
    assert [period.state for period in cycle.store.periods.periods()] == ["closed", "discarded", "discarded"]
    assert result.discarded == {"stacked": ("p000002", "p000003")}
    assert set(result.contaminated["stacked"]) == {second.id, third.id}


def test_a_stacked_period_trains_only_on_its_own_rows_from_the_serving_state(tmp_path):
    cycle = _cycle(tmp_path, "stacked", make_rows(40, prefix="one"), mode="stack")
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="two"))
    cycle.period()
    build = cycle.build()
    assert build.mode == "stack" and build.refs == frozenset(f"two-{index}" for index in range(40))
    learner = SyntheticLearner()
    start = cycle.store.revisions.state(first)
    rows = builds.training_rows(OBJECTIVE, cycle.store.manifests_of(cycle.store.periods.retained()[-1:]),
                                cycle.source)
    assert np.array_equal(build.state["w"], learner.fit(start, rows, objective=OBJECTIVE, seed=0)["w"])


def test_the_learned_state_digest_covers_names_types_shapes_and_values():
    state = {"w": np.zeros(3), "b": np.ones(1)}
    assert builds.state_digest(state) == builds.state_digest({"b": np.ones(1), "w": np.zeros(3)})
    assert builds.state_digest(state) != builds.state_digest({"w": np.zeros(3, dtype=np.float32), "b": np.ones(1)})
    assert builds.state_digest(state) != builds.state_digest({"w": np.zeros((3, 1)), "b": np.ones(1)})
    assert builds.state_digest(state) != builds.state_digest({"w": np.zeros(3), "b": np.ones(1) * 2})
    with pytest.raises(ValueError):
        builds.check_state({"w": np.array([1.0, np.nan])})


def test_a_manifest_admitted_under_another_objective_never_trains(tmp_path):
    from dataclasses import replace as changed

    cycle = _cycle(tmp_path, "probe", make_rows(20))
    cycle.period()
    other = changed(OBJECTIVE, blend={**OBJECTIVE.blend, "owner_label": 0.5})
    with pytest.raises(ValueError, match="objective"):
        builds.training_rows(other, cycle.store.current_manifests(), cycle.source)


def test_build_reports_teacher_losses_as_the_declared_blend_and_zero_means_off():
    # Row counts and blend weights are test assumptions.
    from dataclasses import replace
    from colonymind.plasticity.admission import Admission
    from colonymind.plasticity.objectives import teacher_lines

    rows = make_rows(9, prefix="truth") + make_rows(3, prefix="deep", teacher="deliberation", source_kind="model")
    objective = replace(OBJECTIVE, losses={"owner_label": "cross_entropy", "deliberation": "kl"},
                        blend={"owner_label": 1.0, "deliberation": 0.25})
    source = SyntheticSource(rows)
    manifest = Admission(objective, learner="probe", source_of=source.source_of).admit(rows, period="p1", now=NOW).manifest
    learner = SyntheticLearner()
    build = builds.rebuild(learner, objective, [manifest], source)
    terms = {line["teacher"]: line for line in teacher_lines(build)}
    assert terms["owner_label"]["rows"] == 9 and terms["deliberation"]["rows"] == 3
    assert terms["owner_label"]["counts_as_proof"] and not terms["deliberation"]["counts_as_proof"]
    assert build.loss == pytest.approx(sum(line["loss"] for line in terms.values()))
    raw = learner.losses(build.state, builds.training_rows(objective, [manifest], source), objective=objective)
    assert terms["deliberation"]["loss"] == pytest.approx(0.25 * raw["deliberation"])
    off = replace(objective, blend={"owner_label": 1.0, "deliberation": 0.0})
    manifest = Admission(off, learner="probe", source_of=source.source_of).admit(rows, period="p1", now=NOW).manifest
    zero = builds.rebuild(learner, off, [manifest], source)
    assert teacher_lines(zero)[0:1]
    assert next(line for line in teacher_lines(zero) if line["teacher"] == "deliberation")["loss"] == 0.0
    training = builds.training_rows(off, [manifest], source)
    expected = learner.fit(learner.base(), [row for row in training if row.teacher == "owner_label"], objective=off, seed=0)
    assert np.array_equal(zero.state["w"], expected["w"])
    only_zero = replace(off, blend={"owner_label": 0.0, "deliberation": 0.0})
    assert np.array_equal(learner.fit(learner.base(), training, objective=only_zero, seed=0)["w"], learner.base()["w"])
