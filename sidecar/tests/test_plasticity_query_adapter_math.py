"""Bounded query transforms, revision interchange and teacher fences."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from colonymind.plasticity.builds import Learner, TrainingRow
from colonymind.plasticity.objectives import RECALL_USE
from colonymind.plasticity.query_adapter import (
    QueryAdapterLearner, load_revision, normalize, project_factors, save_revision, spectral, weights_digest,
)


def row(ref="first", *, teacher="owner_label", push=False, query=(1., 0., 0., 0.),
        preferred="good", adversarial=(), shown=None, passages=None):
    passages = passages or {"good": [0., 1., 0., 0.], "bad": [1., 0., 0., 0.],
                            "unseen": [0., 0., 1., 0.]}
    label = {"preferred": preferred, "disfavoured": "bad", "shown": shown or {"good": 1, "bad": 2},
             "passage_sources": {key: "web" if key in adversarial else "owner" for key in passages}}
    return TrainingRow(ref, "recall", json.dumps({"query": query, "passages": passages}), label, teacher,
                       push, RECALL_USE.losses[teacher], RECALL_USE.blend[teacher], "owner")


def fit(learner, rows):
    return learner.fit(learner.base(), rows, objective=RECALL_USE, seed=7)


def test_identity_is_literal_and_outputs_normalized_input():
    learner = QueryAdapterLearner(4)
    state = learner.base()
    assert isinstance(learner, Learner)
    assert state["U"].shape == (4, 32)
    assert weights_digest(state["U"], state["V"]) == "identity"
    query = [3., 4., 0., 0.]
    np.testing.assert_array_equal(learner.apply(state, query), normalize(query))
    empty = fit(learner, [])
    for key in state:
        np.testing.assert_array_equal(empty[key], state[key])


def test_literal_apply_and_exact_revision_format_round_trip(tmp_path):
    u = np.array([[.02, 0], [0, .01], [.01, 0], [0, 0]], dtype=np.float32)
    v = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32)
    identity = {"model": "fixture", "dims": 4}
    path = tmp_path / "revision"
    metadata = save_revision(path, u, v, bound=.05, document_identity=identity)
    assert metadata == {"schema": "query-adapter-1", "dims": 4, "rank": 2,
                        "operator_norm_bound": .05, "document_identity": identity,
                        "digest": hashlib.sha256(u.tobytes() + v.tobytes()).hexdigest(), "scope": "global"}
    assert {item.name for item in path.iterdir()} == {"revision.json", "U.npy", "V.npy"}
    loaded = load_revision(path, document_identity=identity)
    np.testing.assert_allclose(loaded.apply([.6, .8, 0, 0]), [.60377113, .79713574, .00591932, 0.], atol=1e-6)
    assert not loaded.u.flags.writeable and not loaded.v.flags.writeable
    with pytest.raises(FileExistsError):
        save_revision(path, u, v, bound=.05, document_identity=identity)
    with pytest.raises(ValueError, match="identity"):
        load_revision(path, document_identity={"model": "different"})
    altered = u.copy()
    altered[0, 0] += .001
    np.save(path / "U.npy", altered)
    with pytest.raises(ValueError, match="digest"):
        load_revision(path)


def test_qr_projection_and_large_steps_obey_norm_and_displacement_bound():
    random = np.random.default_rng(12)
    u, v = project_factors(random.normal(size=(17, 32)) * 100, random.normal(size=(17, 32)) * 100)
    assert spectral(u, v) <= .05
    assert spectral(u, v) == pytest.approx(np.linalg.norm(u.astype(float) @ v.astype(float).T, ord=2))
    learner = QueryAdapterLearner(4, learning_rate=100., steps=35)
    state = fit(learner, [row(query=(.7, .7, .1, 0.))])
    assert 0 < learner.budget_used(state) <= .05
    for key in ("", "shown_"):
        assert spectral(state[key + "U"], state[key + "V"]) <= .05
    queries = normalize(random.normal(size=(100, 4)))
    assert np.max(np.linalg.norm(learner.apply(state, queries) - queries, axis=1)) <= .1


def test_contrastive_uses_in_batch_negatives_and_squared_arm_trains():
    learner = QueryAdapterLearner(4, steps=25)
    first = row()
    extra = row("second", query=(0., 0., 1., 0.), preferred="other", shown={"other": 1, "bad": 2},
                passages={"other": [0., 0., 0., 1.], "bad": [1., 0., 0., 0.]})
    alone = learner.losses(learner.base(), [first], objective=RECALL_USE)["owner_label"]
    batched = learner.losses(learner.base(), [first, extra], objective=RECALL_USE)["owner_label"]
    extra_alone = learner.losses(learner.base(), [extra], objective=RECALL_USE)["owner_label"]
    assert batched > (alone + extra_alone) / 2
    for arm in ("contrastive", "normalised_squared_error"):
        candidate = QueryAdapterLearner(4, steps=25, arm=arm)
        state = fit(candidate, [first])
        assert candidate.apply(state, [1., 0., 0., 0.])[1] > 0
        assert candidate.losses(state, [first], objective=RECALL_USE)["owner_label"] < candidate.losses(
            candidate.base(), [first], objective=RECALL_USE)["owner_label"]


def test_push_only_never_uses_preferred_and_cannot_raise_disfavoured():
    learner = QueryAdapterLearner(4, learning_rate=2., steps=20)
    first = row(push=True, query=(.7, .7, 0., 0.))
    changed = replace(first, label={**first.label, "preferred": "unseen"})
    state, other = fit(learner, [first]), fit(learner, [changed])
    for key in state:
        np.testing.assert_array_equal(state[key], other[key])
    assert learner.apply(state, [.7, .7, 0., 0.])[0] < normalize([.7, .7, 0., 0.])[0]


@pytest.mark.parametrize("teacher", ["deliberation", "stronger_model", "behavioural_proxy"])
def test_model_teacher_cannot_change_global_or_unseen_scores_and_cannot_lift_adversarial(teacher):
    learner = QueryAdapterLearner(4, steps=30)
    training = row(teacher=teacher, query=(.8, .5, .2, 0.), push=teacher == "behavioural_proxy")
    state = fit(learner, [training])
    query = normalize([.8, .5, .2, 0.])
    np.testing.assert_array_equal(learner.apply(state, query), normalize(query))
    passages = json.loads(training.state)["passages"]
    baseline = learner.rank_scores(learner.base(), query, passages, shown=training.label["shown"])
    scores = learner.rank_scores(state, query, passages, shown=training.label["shown"], adversarial=("good",))
    assert scores["unseen"] == baseline["unseen"]
    assert scores["good"] <= baseline["good"]
    if teacher != "behavioural_proxy":
        plain = learner.rank_scores(state, query, passages, shown=training.label["shown"])
        assert plain["good"] > baseline["good"]


def test_model_unshown_or_adversarial_positive_yields_no_update():
    learner = QueryAdapterLearner(4)
    for training in (row(teacher="deliberation", preferred="unseen"),
                     row(teacher="stronger_model", adversarial=("good",))):
        state = fit(learner, [training])
        for key in state:
            np.testing.assert_array_equal(state[key], learner.base()[key])


def test_shown_revision_refuses_global_apply_and_preserves_unshown(tmp_path):
    learner = QueryAdapterLearner(4)
    training = row(teacher="deliberation")
    state = fit(learner, [training])
    save_revision(tmp_path / "shown", state["shown_U"], state["shown_V"], bound=.05,
                  document_identity={"model": "fixture"}, scope="shown")
    revision = load_revision(tmp_path / "shown")
    with pytest.raises(ValueError, match="shown"):
        revision.apply([1., 0., 0., 0.])
    scores = revision.rank_scores([1., 0., 0., 0.], {"good": [0., 1., 0., 0.], "unseen": [0., 1., 0., 0.]},
                                  shown={"good": 1})
    assert scores["good"] > 0 and scores["unseen"] == 0


def test_rebuild_is_canonical_and_teacher_losses_are_unblended():
    learner = QueryAdapterLearner(4)
    rows = [row(), row("second", teacher="deliberation", query=(.6, .5, .1, 0.))]
    first, second = fit(learner, rows), fit(learner, list(reversed(rows)))
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    losses = learner.losses(first, rows, objective=RECALL_USE)
    assert set(losses) == set(RECALL_USE.losses)
    assert all(np.isfinite(value) and value >= 0 for value in losses.values())
    assert losses["verified_outcome"] == 0
    only_owner = fit(learner, rows[:1])
    off = replace(RECALL_USE, blend={**RECALL_USE.blend, "deliberation": 0.})
    disabled = learner.fit(learner.base(), rows, objective=off, seed=7)
    for key in first:
        np.testing.assert_array_equal(only_owner[key], disabled[key])


@pytest.mark.parametrize("query", [[0., 0., 0., 0.], [float("nan"), 0., 0., 0.], [1., 0.]])
def test_invalid_query_is_rejected(query):
    learner = QueryAdapterLearner(4)
    with pytest.raises(ValueError):
        learner.apply(learner.base(), query)


def test_step_cap_also_holds_for_valid_unbalanced_start_factors():
    learner = QueryAdapterLearner(4, learning_rate=.1, steps=1)
    start = learner.base()
    start["V"][:] = 0
    start["U"][1, 0], start["V"][0, 0] = .005, 5.
    state = learner.fit(start, [row()], objective=RECALL_USE, seed=0)
    displacement = sum(float(np.sum((state[key].astype(float) - start[key]) ** 2)) for key in start) ** .5
    assert displacement <= learner.learning_rate * (1 + 1e-6)


def test_unknown_target_provenance_is_rejected_by_the_learner():
    learner = QueryAdapterLearner(4)
    training = row(teacher="deliberation")
    training = replace(training, label={**training.label, "passage_sources": {"good": "unrecognized", "bad": "owner"}})
    with pytest.raises(ValueError, match="provenance"):
        fit(learner, [training])


def test_one_passage_id_cannot_supply_conflicting_vectors_in_a_batch():
    learner = QueryAdapterLearner(4)
    first = row()
    payload = json.loads(first.state)
    payload["passages"]["good"] = [0., 0., 1., 0.]
    second = replace(first, ref="second", state=json.dumps(payload))
    with pytest.raises(ValueError, match="conflicting"):
        fit(learner, [first, second])


def test_untrusted_contrastive_training_cannot_lift_adversarial_negative():
    learner = QueryAdapterLearner(4)
    # Identical vectors force any reward for good to lift the adversarial negative.
    # Only a proof teacher may make that change; an untrusted update must stop.
    training = row(teacher="synthetic", adversarial=("unseen",),
                   passages={"good": [0., 1., 0., 0.], "bad": [1., 0., 0., 0.],
                             "unseen": [0., 1., 0., 0.]})
    state = fit(learner, [training])
    assert learner.apply(state, [1., 0., 0., 0.])[1] <= 0
    trusted = replace(training, teacher="owner_label", weight=1.)
    assert learner.apply(fit(learner, [trusted]), [1., 0., 0., 0.])[1] > 0


@pytest.mark.parametrize("arm,teacher,push", [("contrastive", "owner_label", False),
                                             ("normalised_squared_error", "owner_label", False),
                                             ("contrastive", "deliberation", False),
                                             ("contrastive", "owner_label", True)])
def test_factor_derivative_matches_finite_differences(arm, teacher, push):
    learner = QueryAdapterLearner(4, rank=2, arm=arm)
    training = row(teacher=teacher, push=push, query=(.6, .5, .2, .1))
    items = learner._rows([training], RECALL_USE)
    item, pool = items[0], learner._pool(items)
    random = np.random.default_rng(81)
    u, v = random.normal(0., .03, (2, 4, 2))
    _, derivative = learner._term(item, u, v, pool)
    analytical = (np.outer(derivative, v.T @ item.query), np.outer(item.query, u.T @ derivative))
    # Central differences at 1e-5 distinguish normalization and chain-rule errors.
    # The tolerance allows subtraction of nearly equal float64 losses.
    for factor_index, factor in enumerate((u, v)):
        numerical = np.zeros_like(factor)
        for index in np.ndindex(factor.shape):
            above, below = [u.copy(), v.copy()], [u.copy(), v.copy()]
            above[factor_index][index] += 1e-5
            below[factor_index][index] -= 1e-5
            numerical[index] = (learner._term(item, *above, pool)[0] - learner._term(item, *below, pool)[0]) / 2e-5
        np.testing.assert_allclose(analytical[factor_index], numerical, rtol=1e-4, atol=1e-7)
