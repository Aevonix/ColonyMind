"""A query revision gets a cutoff fit on development and checked once on held-out queries."""
from __future__ import annotations

import copy
import socket

import pytest

from colonymind import recall_cli
from colonymind.memory.recall import calibration_fingerprint


TRANSFORM = "query-adapter-1:" + "a" * 64
METADATA = {"provider": "fixture", "model": "fixture-reranker", "index_generation": "generation-1"}


def rows(prefix, *, negatives=150, positive_score=.8, negative_score=.2, transform=TRANSFORM):
    stamp = calibration_fingerprint({**METADATA, "query_transform": transform})
    examples = [
        {"query_id": prefix + "-answer", "scores": [positive_score, .1],
         "relevant": [True, False], "no_answer": False},
        *[{"query_id": f"{prefix}-negative-{index}", "scores": [negative_score],
           "relevant": [False], "no_answer": True} for index in range(negatives)],
    ]
    return [{**row, "calibration_fingerprint": stamp} for row in examples]


def refit(development=None, held_out=None, **kwargs):
    return recall_cli.refit_query_adapter_cutoff(
        METADATA, TRANSFORM, development if development is not None else rows("dev"),
        held_out if held_out is not None else rows("held"), **kwargs)


def test_refit_freezes_development_cutoff_and_binds_only_transform_metadata(monkeypatch):
    from colonymind.vector import embedder, reranker

    def forbidden(*args, **kwargs):
        pytest.fail("cutoff refitting cannot load models, call endpoints or re-embed")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(embedder.EmbeddingPipeline, "embed_query", forbidden)
    monkeypatch.setattr(embedder.EmbeddingPipeline, "embed_batch", forbidden)
    monkeypatch.setattr(embedder.CPUEmbeddingProvider, "_load_model", forbidden)
    monkeypatch.setattr(reranker.CPURerankerProvider, "_load_model", forbidden)
    before = copy.deepcopy(METADATA)
    result = refit(held_out=rows("held", positive_score=.9, negative_score=.6))
    assert result is not None
    assert result["cutoff"] == .8
    assert result["query_transform"] == TRANSFORM
    assert result["metadata"] == {**METADATA, "query_transform": TRANSFORM}
    assert result["fingerprint"] == calibration_fingerprint(result["metadata"])
    assert result["fingerprint"] != calibration_fingerprint(METADATA)
    assert result["held_out"]["no_answer_queries"] == 150
    assert result["held_out"]["false_injection_rate"] == 0.0
    assert result["held_out"]["answer_recall"] == 1.0
    assert result["held_out_checks"] == 1
    assert METADATA == before
    assert result["id"].startswith("recall-cutoff-1:")
    assert refit(held_out=rows("held", positive_score=.9, negative_score=.6)) == result


def test_held_out_failure_does_not_tune_the_threshold_to_holdout():
    # .9 would pass this holdout, but development selected .8 and must stay frozen.
    assert refit(held_out=rows("held", positive_score=.95, negative_score=.85)) is None
    assert refit(held_out=rows("held", positive_score=.7)) is None


def test_held_out_queries_are_evaluated_once(monkeypatch):
    evaluate = recall_cli._cutoff_metrics
    checked = []

    def capture(examples, cutoff):
        if examples[0]["query_id"].startswith("held"):
            checked.append(cutoff)
        return evaluate(examples, cutoff)

    monkeypatch.setattr(recall_cli, "_cutoff_metrics", capture)
    assert refit() is not None
    assert checked == [.8]


@pytest.mark.parametrize("development,held_out", [
    (rows("dev"), rows("held", negatives=149)),
    (rows("dev", negatives=0), rows("held")),
    (rows("dev")[1:], rows("held")),
    (rows("dev"), rows("held")[1:]),
])
def test_insufficient_query_controls_never_produce_a_record(development, held_out):
    assert refit(development, held_out) is None


def test_no_answer_controls_count_queries_not_candidates():
    held = rows("held", negatives=1)
    held[-1]["scores"] *= 150
    held[-1]["relevant"] *= 150
    assert refit(held_out=held) is None


def test_failed_development_fit_does_not_inspect_the_holdout():
    class Unopened:
        def __iter__(self):
            pytest.fail("a failed development fit must leave holdout unopened")

    assert refit(rows("dev", negative_score=.9), Unopened()) is None


def test_development_selects_maximum_recall_inside_declared_false_injection_limit():
    dev = rows("dev", negative_score=.3)
    dev.append({**dev[0], "query_id": "dev-second-answer", "scores": [.6], "relevant": [True]})
    held = rows("held", positive_score=.7)
    assert refit(dev, held)["cutoff"] == .6


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "0.8", True])
def test_bad_scores_are_rejected(bad):
    dev = rows("dev")
    dev[0]["scores"][0] = bad
    with pytest.raises(ValueError, match="score"):
        refit(dev)


def test_labels_and_splits_cannot_leak_or_count_a_query_twice():
    with pytest.raises(ValueError, match="disjoint"):
        refit(rows("same"), rows("same"))
    duplicate = rows("held")
    duplicate.append(copy.deepcopy(duplicate[-1]))
    with pytest.raises(ValueError, match="unique"):
        refit(held_out=duplicate)
    inconsistent = rows("dev")
    inconsistent[0]["no_answer"] = True
    with pytest.raises(ValueError, match="no.answer"):
        refit(inconsistent)
    inconsistent = rows("dev")
    inconsistent[0]["relevant"] = [True]
    with pytest.raises(ValueError, match="aligned"):
        refit(inconsistent)


def test_transform_and_quality_bars_are_validated():
    with pytest.raises(ValueError, match="query_transform"):
        recall_cli.refit_query_adapter_cutoff(METADATA, "unbound", rows("dev"), rows("held"))
    for kwargs in ({"max_false_injection_rate": -1}, {"max_false_injection_rate": float("nan")},
                   {"min_answer_recall": 0}, {"min_answer_recall": 2}):
        with pytest.raises(ValueError, match="rate|recall"):
            refit(**kwargs)


def test_new_revision_changes_record_and_fingerprint_without_an_index_change():
    first = refit()
    other = "query-adapter-1:" + "b" * 64
    second = recall_cli.refit_query_adapter_cutoff(METADATA, other,
                                                 rows("dev", transform=other), rows("held", transform=other))
    assert first["fingerprint"] != second["fingerprint"]
    assert first["id"] != second["id"]
    assert first["metadata"]["index_generation"] == second["metadata"]["index_generation"]


def test_captured_scores_must_have_the_full_transform_configuration_fingerprint():
    with pytest.raises(ValueError, match="fingerprint"):
        refit(rows("dev", transform="identity"))
    stale = rows("held")
    stale[-1]["calibration_fingerprint"] = calibration_fingerprint(
        {**METADATA, "index_generation": "old-generation", "query_transform": TRANSFORM})
    with pytest.raises(ValueError, match="fingerprint"):
        refit(held_out=stale)
    missing = rows("dev")
    missing[0].pop("calibration_fingerprint")
    with pytest.raises(ValueError, match="fingerprint"):
        refit(missing)


def test_relaxed_quality_bars_cannot_weaken_the_previous_closure():
    with pytest.raises(ValueError, match="reference"):
        refit(min_answer_recall=.5)
    reference = {"answer_recall": 1.0, "false_injection_rate": 0.0}
    assert refit(held_out=rows("held", positive_score=.7), min_answer_recall=.5,
                 reference_metrics=reference) is None
    assert refit(held_out=rows("held", negative_score=.85), max_false_injection_rate=.1,
                 reference_metrics=reference) is None
    result = refit(min_answer_recall=.5, max_false_injection_rate=.1, reference_metrics=reference)
    assert result["quality_bars"] == {"min_answer_recall": 1.0, "max_false_injection_rate": 0.0}
