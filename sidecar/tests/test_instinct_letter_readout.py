"""IN1: descriptions on the wire, labels only after reading the letters.

The measured readout uses one declared order, with no rotation override.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
import sys

import httpx
import pytest

from colonymind.instinct import REGISTRY, calibration
from colonymind.instinct.providers import local_decision as local
from colonymind.instinct.registry import Point, Tier
from colonymind.instinct.tiers import Instinct

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "decisions"))
import analysis  # noqa: E402
import questions as bench_questions  # noqa: E402

pytestmark = pytest.mark.asyncio


def response(probabilities, *, revision="fixture"):
    return httpx.Response(200, json={
        "protocol": local.PROTOCOL, "input_truncated": False,
        "backend": {"checkpoint": "fixture", "model_revision": revision},
        "answers": {"decision": {"type": "choice", "probabilities": probabilities}},
    })


def make_provider(handler, **kwargs):
    return local.LocalDecision("local", url="http://decide.test",
                               transport=httpx.MockTransport(handler), **kwargs)


def first_letter(request):
    letters = json.loads(request.content)["questions"]["decision"]["criteria"]
    return response({letter: float(index == 0) for index, letter in enumerate(letters)})


@pytest.mark.parametrize("point", list(REGISTRY.values()), ids=list(REGISTRY))
async def test_every_registered_point_sends_only_letter_keys_and_descriptions(point):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return first_letter(request)

    result = await make_provider(handler).answer("Fixture text.", {point.name: point.question()})
    assert local.ROTATIONS == 1
    assert len(bodies) == 1
    descriptions = list(point.labels.values())
    for body in bodies:
        expected = descriptions
        assert body == {"state": "Fixture text.", "questions": {"decision": {
            "type": "choice", "instructions": point.instructions,
            "criteria": dict(zip("abcdefghijklmnopqrstuvwxyz", expected)),
        }}}
    assert set(result.probabilities[point.name]) == set(point.labels)


@pytest.mark.parametrize("point", list(REGISTRY.values()), ids=list(REGISTRY))
async def test_renaming_labels_changes_neither_request_bytes_nor_mapped_answer(point, monkeypatch):
    # MockTransport completes each request; this checks bytes and mapping regardless
    # of scheduler latency. The hanging-question tests below exercise the deadline.
    deadline = asyncio.timeout
    monkeypatch.setattr(local.asyncio, "timeout", lambda _: deadline(None))
    bodies = []

    def handler(request):
        bodies.append(request.content)
        return first_letter(request)

    names = {label: f"private_label_{index}" for index, label in enumerate(point.labels)}
    renamed = replace(point, type="choice", labels={names[k]: v for k, v in point.labels.items()})
    provider = make_provider(handler)
    original = await provider.answer("Fixture text.", {point.name: point.question()})
    changed = await provider.answer("Fixture text.", {point.name: renamed.question()})
    halfway = len(bodies) // 2
    assert bodies[:halfway] == bodies[halfway:]
    assert original.probabilities[point.name] == {
        label: changed.probabilities[point.name][names[label]] for label in point.labels
    }


async def test_letters_map_back_in_declared_order_and_questions_run_concurrently():
    labels = {"alpha_name": "first meaning", "beta_name": "second meaning", "gamma_name": "third meaning"}
    expected = dict(zip(labels.values(), (0.6, 0.3, 0.1)))
    entered = 0
    ready = asyncio.Event()
    count = 3

    async def handler(request):
        nonlocal entered
        entered += 1
        if entered == count:
            ready.set()
        await ready.wait()
        criteria = json.loads(request.content)["questions"]["decision"]["criteria"]
        return response({letter: expected[description] for letter, description in criteria.items()})

    questions = {f"probe_{index}": {
        "type": "choice", "instructions": "Which meaning?", "labels": labels,
    } for index in range(count)}
    answer = await asyncio.wait_for(make_provider(handler).answer("text", questions), 1)
    for name in questions:
        assert answer.probabilities[name] == pytest.approx(dict(zip(labels, (0.6, 0.3, 0.1))))
    assert entered == count


async def test_yes_no_maps_the_single_declared_order():
    question = {"type": "yes_no", "instructions": "Settled?", "labels": {"yes": "settled", "no": "open"}}
    answer = await make_provider(first_letter).answer("text", {"probe": question})
    assert answer.probabilities["probe"] == {"yes": 1.0, "no": 0.0}


async def test_one_hanging_question_fails_the_whole_ask_and_cancels_pending_requests():
    finished = []

    async def handler(request):
        criteria = json.loads(request.content)["questions"]["decision"]["criteria"]
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(criteria["a"])

    question = {"type": "yes_no", "instructions": "Settled?", "labels": {"yes": "settled", "no": "open"}}
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(make_provider(handler, timeout_ms=20).answer("text", {"probe": question, "other": question}), 1)
    # Builder assumption: a half-second test allowance distinguishes the provider's deadline from the outer guard.
    assert asyncio.get_running_loop().time() - started < 0.5
    assert len(finished) == 2


async def test_named_records_do_not_apply_and_identity_records_do():
    from colonymind.instinct.providers import calibration_kind

    point = Point(name="probe", type="choice", effect="read", instructions="Which meaning?",
                  labels={"first_name": "first meaning", "second_name": "second meaning"}, template="{text}",
                  tiers=(Tier("fast", state="live", may_settle=("first_name", "second_name")),))
    provider = make_provider(first_letter)
    identity = calibration_kind(provider)
    assert identity == "local-decision.v1/letters-r1"
    named = calibration.Record(point=point.name, version=point.version, tier="fast", provider_kind=local.KIND,
                               revision="fixture@fixture", bars={label: 0.4 for label in point.labels})
    engine = Instinct(providers={"local": provider}, fast="local", registry={point.name: point}, records=[named])
    await engine.decide(point.name, {"text": "text"})
    assert engine.health()["points"][point.name]["counts"]["fast"]["no_record"] == 1
    assert engine.record_for(point, "fast", provider, named.revision) is None
    matched = replace(named, provider_kind=identity)
    engine = Instinct(providers={"local": provider}, fast="local", registry={point.name: point}, records=[matched])
    assert engine.record_for(point, "fast", provider, matched.revision) == matched
    assert (await engine.decide(point.name, {"text": "text"})).tier == "fast"


async def test_fake_answers_file_has_zero_renamed_flip_for_every_bench_point(tmp_path):
    provider = make_provider(first_letter)
    document = {"perturbations": {"base": {}, "renamed": {}}}
    for point in [bench_questions.spec(name) for name in bench_questions.points()]:
        for kind in document["perturbations"]:
            question, back = bench_questions.perturb(point, kind, key=point.name)
            answer = await provider.answer("Fixture text.", {point.name: question})
            document["perturbations"][kind][point.name] = {
                "raw": bench_questions.mapped_back(answer.probabilities[point.name], back),
            }
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(document))
    loaded = json.loads(path.read_text())["perturbations"]
    for point in [bench_questions.spec(name) for name in bench_questions.points()]:
        measured = analysis.flip_rate({point.name: loaded["base"][point.name]["raw"]},
                                      {point.name: loaded["renamed"][point.name]["raw"]})
        assert measured["rate"] == 0


async def test_wire_question_uses_the_same_letter_contract_and_maps_back():
    point = REGISTRY["owner_verdict"]
    spec = local.WireQuestion.of(point)
    assert spec.question() == {"type": "choice", "instructions": point.instructions,
                               "criteria": dict(zip("ab", point.labels.values()))}
    assert local.answer_probabilities(spec, response({"a": 0.8, "b": 0.2}).json()) == {
        label: probability for label, probability in zip(point.labels, (0.8, 0.2))
    }


async def test_shipped_records_match_only_the_selected_letter_identity():
    from colonymind.instinct import records

    provider = make_provider(first_letter)
    identity = "local-decision.v1/letters-r1"
    engine = Instinct(providers={"local": provider}, fast="local")
    shipped = records.load(strict=True)
    assert engine.shipped_records == shipped
    assert len(shipped) == 1
    assert {(record.point, record.tier) for record in shipped} == {("opt_out", "fast")}
    assert all(record.revision for record in shipped)
    for record in shipped:
        assert record.provider_kind == identity
        assert record.method == "isotonic" and record.n == 71
        assert record.maps and record.support == {"yes": (0.1601, 0.8439)}
        point = REGISTRY[record.point]
        assert record.version == point.version
        assert engine.record_for(point, "fast", provider, record.revision) == record
        assert calibration.find(engine.shipped_records, version=point.version, tier="fast",
                                provider_kind=local.KIND, model="", revision=record.revision) is None
    assert engine.record_for(REGISTRY["opt_out"], "fast", provider, "unmeasured") is None
    assert engine.record_for(REGISTRY["opt_out"], "fast", provider, "") is None


@pytest.mark.parametrize("point", [point for point in REGISTRY.values()
                                  if point.tier("fast") is not None and point.name != "opt_out"],
                         ids=lambda point: point.name)
@pytest.mark.parametrize("revision", ["measured-revision", ""])
async def test_a_point_without_a_shipped_letter_record_abstains_with_no_record(point, revision):
    def handler(request):
        reply = first_letter(request)
        body = reply.json()
        body["backend"] = {"checkpoint": revision}
        return httpx.Response(200, json=body)

    # Put the real question's fast tier live so an abstention cannot be attributed to shadow state.
    point = replace(point, tiers=(replace(point.tier("fast"), state="live"),), rules=None)
    engine = Instinct(providers={"local": make_provider(handler)}, fast="local", registry={point.name: point})
    decision = await engine.decide(point.name, {field: "Fixture text." for field in point.fields})
    assert decision.tier == "fallback"
    assert engine.health()["points"][point.name]["counts"]["fast"]["no_record"] == 1


@pytest.mark.parametrize("probabilities", [
    {"yes": 0.7, "no": 0.3}, {"a": 0.8, "c": 0.2}, {"a": 0.2, "b": 0.2},
    {"a": True, "b": 0}, {"a": "0.8", "b": 0.2}, {"a": -0.1, "b": 1.1},
])
async def test_a_letter_answer_is_validated_before_mapping(probabilities):
    provider = make_provider(lambda request: response(probabilities))
    with pytest.raises(ValueError):
        await provider.answer("text", {"probe": {
            "type": "yes_no", "instructions": "Settled?", "labels": {"yes": "settled", "no": "open"},
        }})


async def test_a_failed_question_cancels_its_hanging_sibling():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(request):
        criteria = json.loads(request.content)["questions"]["decision"]["criteria"]
        if criteria["a"] == "settled":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        await entered.wait()
        return httpx.Response(503)

    with pytest.raises(httpx.HTTPStatusError):
        await make_provider(handler).answer("text", {
            "probe": {"type": "yes_no", "instructions": "Settled?",
                      "labels": {"yes": "settled", "no": "open"}},
            "other": {"type": "choice", "instructions": "Open?",
                      "labels": {"open": "open", "settled": "settled"}},
        })
    assert cancelled.is_set()


async def test_a_timed_out_question_counts_toward_the_tiers_breaker():
    async def handler(request):
        await asyncio.Event().wait()

    point = Point(name="probe", type="yes_no", effect="read", instructions="Settled?",
                  labels={"yes": "settled", "no": "open"}, template="{text}",
                  tiers=(Tier("fast", state="live", may_settle=("yes", "no")),))
    provider = make_provider(handler, timeout_ms=20)
    engine = Instinct(providers={"local": provider}, fast="local", registry={point.name: point}, records=[])
    for _ in range(3):
        assert (await engine.decide(point.name, {"text": "text"})).tier == "fallback"
    assert engine.health()["points"][point.name]["counts"]["fast"]["failed"] == 3
    await engine.decide(point.name, {"text": "text"})
    assert engine.health()["points"][point.name]["counts"]["fast"]["provider_down"] == 1
