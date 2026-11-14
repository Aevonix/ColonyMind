"""One synthetic learner through the whole core (docs/PLASTICITY.md): observe, admit, build, gate, promote, erase
and rebuild, each step through the module that owns it."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from colonymind.plasticity.admission import Admission
from colonymind.plasticity.audience import AudiencePolicy
from colonymind.plasticity.budgets import Budget, BudgetCheck
from colonymind.plasticity.builds import rebuild, training_rows
from colonymind.plasticity.erasure import Erasure, scan
from colonymind.plasticity.gates import Evidence, SetLedger, evaluate, pooled_suite
from colonymind.plasticity.states import Controller
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import (BUDGET, EPOCH, OBJECTIVE, SyntheticLearner, SyntheticSource,
                                            SyntheticTemplate, canaries, correct, make_rows, shuffled, state_hash)

NOW = EPOCH + timedelta(days=10)


def test_observe_admit_build_gate_promote_erase_and_rebuild(tmp_path):
    clock = lambda: NOW  # noqa: E731
    learner = SyntheticLearner()
    store = LearnerStore(tmp_path, "learner", base=learner.identity, declared=("threshold",), clock=clock)
    store.budgets.declare(Budget("learner", learner.budget_type, BUDGET))
    sets = SetLedger(tmp_path / "sets")
    suite = pooled_suite([SyntheticTemplate()], seeds=range(100, 110))
    sets.register(suite.id, "gate_suite", suite.refs())
    controller = Controller(store, live=True, clock=clock)
    planted = canaries()

    # Observe: the owner's labels, a contact source's verified outcomes, and deep answers (their own teacher).
    source = SyntheticSource(make_rows(150, prefix="owner"))
    source.add(make_rows(40, prefix="outcome", teacher="verified_outcome", source_kind="resolver",
                         audience="contact"), source="contact-1")
    source.add(make_rows(30, prefix="deep", teacher="deliberation", source_kind="model"))
    admission = Admission(OBJECTIVE, learner="learner", source_of=source.source_of,
                          protected_refs=sets.protected_refs(), clock=clock)

    def admit():
        period = store.periods.open()
        admitted = admission.admit(source.rows(OBJECTIVE), period=period.id, already=store.decided())
        store.periods.add_manifest(period.id, store.manifests.put(admitted.manifest))
        store.periods.close(period.id)
        return period.id, admitted

    def build_and_gate(period=None):
        serving = store.revisions.serving()
        serving_state = store.revisions.state(serving)
        build = rebuild(learner, OBJECTIVE, store.current_manifests(), source)
        closure = store.revisions.compose(
            state=build.digest, objective=OBJECTIVE.identity, calibration=(f"calibration-{build.digest[:12]}",),
            dependents={"threshold": "0.5"}, parent=serving.id, manifests=build.manifests,
            periods=tuple(found.id for found in store.periods.retained()), audience=build.audience, mode="rebuild")
        controller.propose(closure, build.state)
        # The engine computes the budget; the core records it and refuses a revision over it.
        store.budgets.record(BudgetCheck("learner", build.digest, learner.budget_type, BUDGET,
                                         learner.budget_used(build.state)))
        rows = training_rows(OBJECTIVE, store.current_manifests(), source)
        control = learner.fit(learner.base(), shuffled(rows, seed=7), objective=OBJECTIVE, seed=0)
        states = [item.state for item in suite.items]
        evidence = Evidence(
            reference=correct(learner, serving_state, suite.items), base=correct(learner, learner.base(), suite.items),
            candidate=correct(learner, build.state, suite.items),
            control=correct(learner, control, suite.items), canaries_base=learner.exposure(learner.base(), planted),
            canaries_candidate=learner.exposure(build.state, planted),
            served=learner.predict(store.revisions.state(closure), states),
            offline=learner.predict(build.state, states),
            metrics={"accuracy": float(np.mean(correct(learner, build.state, suite.items)))},
            budget=store.budgets.result("learner", build.digest), calibration=closure.calibration,
            parent_calibration=serving.calibration, state=build.state)
        report = evaluate(closure.id, suite, evidence, sets=sets)
        return closure, build, report, controller.consider(closure.id, report, period=period)

    # Admit: the contact source is capped at 10% of the period; deep answers are rows under their own teacher.
    period, admitted = admit()
    by_teacher = {}
    for row in admitted.rows:
        by_teacher[row["teacher"]] = by_teacher.get(row["teacher"], 0) + 1
    assert by_teacher == {"owner_label": 150, "verified_outcome": 20, "deliberation": 30}

    # Build and gate; the first live revision serves as soon as its gates pass.
    closure, build, report, outcome = build_and_gate(period)
    assert report.passed, report.failed
    assert outcome == "promoted"
    assert store.revisions.serving().id == closure.id
    assert np.mean(correct(learner, build.state, suite.items)) > 0.85

    # Audience: the stack learned from owner rows, so it serves owner turns only.
    policy = AudiencePolicy()
    assert build.audience == "owner"
    serving, base = store.revisions.serving(), store.revisions.base
    assert policy.select("learner", "owner", serving=serving, base=base).id == closure.id
    assert policy.select("learner", "guest", serving=serving, base=base).id == base.id

    # Erase one of the owner's rows: unserve, delete, rebuild, and nothing left on disk.
    erased = {"owner-3"}
    source.erase(erased)
    result = Erasure({"learner": store}).erase(erased, readmit={"learner": (admission, source)},
                                               rebuild={"learner": lambda: build_and_gate()})
    assert result.clean
    assert result.fallbacks == {"learner": store.revisions.base.id}
    rebuilt = store.revisions.serving()
    assert rebuilt.id not in (closure.id, store.revisions.base.id)
    assert all("owner-3" not in manifest.refs() for manifest in store.current_manifests())
    assert scan(tmp_path, erased | {state_hash("synthetic_yes_no owner 3")}) == ()
    assert np.mean(correct(learner, store.revisions.state(rebuilt), suite.items)) > 0.85
