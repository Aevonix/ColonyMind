"""The fast tier over a ``local-decision.v1`` server (``colonymind.instinct``): a decision model's typed answer, or
nothing applied, so the caller keeps its existing path. The 1.11 decision layer's cases, on Instinct."""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest

from colonymind import instinct
from colonymind.instinct import FAST_POINTS, REGISTRY
from colonymind.instinct.calibration import Record
from colonymind.instinct.log import Store
from colonymind.instinct.providers import calibration_kind
from colonymind.instinct.providers.local_decision import DEFAULT_TIMEOUT_MS, MAX_STATE_CHARS, LocalDecision
from colonymind.instinct.tiers import Instinct

URL = "http://decide.test"
MEASURED = {"name": "fake", "checkpoint": "typed-decisions",
            "model_revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982"}


def choice_answer(probabilities, *, question="decision", backend=None):
    probabilities = {dict(zip(REGISTRY["outreach_reply"].labels, "abcde")).get(label, label): value
                     for label, value in probabilities.items()}
    top = max(probabilities, key=probabilities.get)
    return {"protocol": "local-decision.v1", "backend": backend or MEASURED, "elapsed_ms": 3.0,
            "usage": {"input_tokens": 40, "output_tokens": 0}, "input_truncated": False,
            "answers": {question: {"type": "choice", "choice": top, "probabilities": probabilities,
                                   "confidence": 0.5, "action": {"act_probability": 1.0}}}}


def yes_no_answer(p_yes, *, question="decision"):
    return choice_answer({"a": p_yes, "b": 1 - p_yes}, question=question)


class Server:
    """A fake decision endpoint: ``answer(request_json)`` is the JSON body, or an ``httpx.Response``."""

    def __init__(self, answer):
        self.answer, self.requests = answer, []

    def transport(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append((request.url.path, body))
            result = self.answer(body)
            return result if isinstance(result, httpx.Response) else httpx.Response(200, json=result)
        return httpx.MockTransport(handler)


def engine(server, *live, timeout_ms=250, store=None, transport=None, url=URL, **points):
    provider = LocalDecision("local", url=url, timeout_ms=timeout_ms,
                             transport=transport or (server.transport() if server else None))
    # Synthetic records exercise the wire without claiming measured shipped calibration.
    records = [Record(point=point.name, version=point.version, tier="fast",
                      provider_kind=calibration_kind(provider),
                      revision=f"{MEASURED['checkpoint']}@{MEASURED['model_revision']}",
                      bars={label: 0.85 for label in point.labels}, source="fixture")
               for point in REGISTRY.values() if point.name in FAST_POINTS]
    made = Instinct(providers={"local": provider}, fast="local", points_config=points, store=store, records=records)
    for name in live:
        made.set_state(name, "fast", "live", source="test")
    return made


def counts(made, point):
    return made.health()["points"][point]["counts"].get("fast", {})


REPLY = {"engaged": 0.02, "dig_deeper": 0.9, "not_interested": 0.03, "not_now": 0.03, "stop": 0.02}


async def test_a_live_point_asks_one_typed_question_and_applies_the_answer():
    server = Server(lambda body: choice_answer(REPLY))
    decision = await engine(server, "outreach_reply").decide("outreach_reply", {"text": "Who ran it?",
                                                                               "topic": "tidal energy"})
    assert decision.applied and decision.label == "dig_deeper" and decision.tier == "fast"
    [(path, body)] = server.requests
    assert path == "/v1/decide" and set(body) == {"state", "questions"}
    [(name, question)] = body["questions"].items()
    assert name == "decision" and question["type"] == "choice"
    assert question["criteria"] == dict(zip("abcde", REGISTRY["outreach_reply"].labels.values()))
    assert "tidal energy" in body["state"] and "Who ran it?" in body["state"]


async def test_a_shadow_point_is_asked_and_never_applied_and_no_endpoint_makes_no_call():
    server = Server(lambda body: choice_answer(REPLY))
    shadow = engine(server)
    decision = await shadow.decide("outreach_reply", {"text": "Who ran it?", "topic": "tidal energy"})
    await shadow.drain()
    assert decision.applied is False and decision.label is None and len(server.requests) == 1
    nowhere = Instinct.from_settings({"providers": {"local": {"kind": "local-decision.v1", "url": ""}},
                                      "fast": "local"})
    assert nowhere.provider_for("fast") is None
    assert (await nowhere.decide("outreach_reply", {"text": "x", "topic": "y"})).applied is False
    assert len(server.requests) == 1


async def test_an_answer_below_the_records_bars_is_no_answer():
    unsure = {"engaged": 0.3, "dig_deeper": 0.3, "not_interested": 0.2, "not_now": 0.1, "stop": 0.1}
    made = engine(Server(lambda body: choice_answer(unsure)), "outreach_reply")
    assert (await made.decide("outreach_reply", {"text": "Who ran it?", "topic": "tidal"})).applied is False
    assert counts(made, "outreach_reply")["below_bar"] == 1
    band = engine(Server(lambda body: yes_no_answer(0.6)), "opt_out")         # under the fixture bar of 0.85
    assert (await band.decide("opt_out", {"text": "whatever"})).applied is False
    sure = await engine(Server(lambda body: yes_no_answer(0.99)), "opt_out").decide("opt_out", {"text": "x"})
    assert sure.label == "yes" and sure.probability > 0.85
    no = await engine(Server(lambda body: yes_no_answer(0.01)), "opt_out").decide("opt_out", {"text": "x"})
    assert no.label == "no"


@pytest.mark.parametrize("response", [
    httpx.Response(429, json={"detail": "inference busy"}),
    httpx.Response(503, json={"detail": "inference failed"}),
    httpx.Response(422, json={"detail": {"code": "input_would_truncate"}}),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json={"protocol": "other", "answers": {}}),
    httpx.Response(200, json=choice_answer({"engaged": 0.9, "unknown_label": 0.1})),
    httpx.Response(200, json=choice_answer({**REPLY, "stop": 1.7})),
    httpx.Response(200, json=yes_no_answer(0.9)),          # two letters where five were asked
])
async def test_a_failed_or_malformed_answer_is_no_answer(response):
    made = engine(Server(lambda body: response), "outreach_reply")
    assert (await made.decide("outreach_reply", {"text": "Who ran it?", "topic": "tidal"})).applied is False
    assert counts(made, "outreach_reply")["failed"] == 1


_DROP = object()


@pytest.mark.parametrize("changes", [
    {"input_truncated": True},                  # the model read a cut input: its answer is about other words
    {"input_truncated": _DROP},                 # the contract says it was not cut; an answer that does not say so
    {"input_truncated": None},
    {"input_truncated": 0},
    {"input_truncated": "false"},
])
async def test_an_answer_that_does_not_say_its_input_was_read_whole_is_no_answer(changes):
    def respond(body):
        answer = {**yes_no_answer(0.01), **changes}
        return {key: value for key, value in answer.items() if value is not _DROP}
    made = engine(Server(respond))
    decision = await made.decide("owner_verdict", {"reply": "The order code is C4411.", "text": "That was wrong."})
    assert decision.applied is False and counts(made, "owner_verdict")["failed"] == 1


async def test_an_answer_to_other_questions_than_the_one_asked_is_no_answer():
    extra = yes_no_answer(0.01)
    extra["answers"]["another"] = dict(extra["answers"]["decision"])
    made = engine(Server(lambda body: httpx.Response(200, json=extra)))
    assert (await made.decide("owner_verdict", {"reply": "r", "text": "t"})).applied is False
    assert counts(made, "owner_verdict")["failed"] == 1


@pytest.mark.parametrize("probabilities", [
    {"engaged": 0.0, "dig_deeper": 0.0, "not_interested": 0.0, "not_now": 0.0, "stop": 0.05},   # renormalises to 1.0
    {"engaged": 0.0, "dig_deeper": 0.0, "not_interested": 0.0, "not_now": 0.0, "stop": 0.0},
    {"engaged": 0.4, "dig_deeper": 0.4, "not_interested": 0.4, "not_now": 0.4, "stop": 0.9},
    {"engaged": 0.1, "dig_deeper": 0.1, "not_interested": 0.1, "not_now": 0.1, "stop": 0.5},    # sums to 0.9
])
async def test_a_choice_that_is_not_a_distribution_is_no_answer(probabilities):
    """A choice's probabilities sum to one (within 0.01, the contract's rounding); anything else is malformed, and
    renormalising it would turn a faint label into a certain one."""
    made = engine(Server(lambda body: choice_answer(probabilities)), "outreach_reply")
    assert (await made.decide("outreach_reply", {"text": "Who ran it?", "topic": "tidal"})).applied is False
    assert counts(made, "outreach_reply")["failed"] == 1


async def test_a_choice_rounded_within_the_contracts_tolerance_is_read():
    rounded = {"engaged": 0.02, "dig_deeper": 0.9, "not_interested": 0.02, "not_now": 0.02, "stop": 0.035}
    made = engine(Server(lambda body: choice_answer(rounded)), "outreach_reply")
    assert (await made.decide("outreach_reply", {"text": "Who ran it?", "topic": "tidal"})).label == "dig_deeper"


class Hanging(httpx.AsyncBaseTransport):
    """An endpoint that accepts the request and never answers."""

    async def handle_async_request(self, request):
        await asyncio.Event().wait()


async def test_a_slow_endpoint_is_no_answer_within_the_timeout():
    made = engine(None, "outreach_reply", timeout_ms=50, transport=Hanging())
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert (await made.decide("outreach_reply", {"text": "Who ran it?", "topic": "tidal"})).applied is False
    assert loop.time() - started < 1.0


@pytest.mark.parametrize("raw", [math.inf, math.nan, -5, 0, 0.5, 5001, 60000, 1e308, "abc", "", None, True])
async def test_an_unusable_time_limit_keeps_the_default_one(raw):
    """A time limit outside 1 to 5000 ms (what colonymind.yaml accepts), or not a finite number, is ignored: the
    call still ends at the default limit, so a hanging endpoint never holds the caller."""
    provider = LocalDecision("local", url=URL, timeout_ms=raw, transport=Hanging())
    assert provider.timeout_ms == DEFAULT_TIMEOUT_MS
    made = Instinct(providers={"local": provider}, fast="local")
    made.set_state("opt_out", "fast", "live", source="test")
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert (await asyncio.wait_for(made.decide("opt_out", {"text": "leave it"}), 2.0)).applied is False
    assert loop.time() - started < DEFAULT_TIMEOUT_MS / 1000 + 0.5


def test_a_usable_time_limit_is_kept():
    assert LocalDecision("local", url=URL, timeout_ms=1).timeout_ms == 1
    assert LocalDecision("local", url=URL, timeout_ms=5000).timeout_ms == 5000
    assert LocalDecision("local", url=URL, timeout_ms=50).timeout_ms == 50


async def test_an_unreachable_endpoint_is_no_answer():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)
    made = engine(None, "opt_out", transport=httpx.MockTransport(refuse))
    assert (await made.decide("opt_out", {"text": "leave it"})).applied is False


async def test_an_input_too_long_for_the_model_is_never_sent(tmp_path):
    server = Server(lambda body: yes_no_answer(0.99))
    made = engine(server, "opt_out", store=Store(tmp_path / "instinct.db"))
    assert (await made.decide("opt_out", {"text": "word " * 400})).applied is False
    assert server.requests == [] and counts(made, "opt_out")["skipped"] == 1
    assert len(REGISTRY["opt_out"].render({"text": "word " * 400})) > MAX_STATE_CHARS


async def test_an_unknown_point_is_refused():
    with pytest.raises(KeyError):
        await engine(Server(lambda body: yes_no_answer(0.9))).decide("anything", {"text": "x"})


def test_every_fast_point_is_a_short_typed_question():
    for name, point in FAST_POINTS.items():
        assert point.kind in {"choice", "yes_no"}, name
        assert point.kind != "choice" or 2 <= len(point.labels) <= 8, name
        assert len(point.instructions) <= 200, name


def test_the_shared_engine_follows_the_environment(monkeypatch):
    monkeypatch.delenv("COLONYMIND_INSTINCT", raising=False)
    assert instinct.shared().provider_for("fast") is None
    monkeypatch.setenv("COLONYMIND_INSTINCT", json.dumps({"providers": {"local": {"kind": "local-decision.v1",
                                                                                "url": URL}}, "fast": "local"}))
    assert instinct.shared().provider_for("fast").url == URL and instinct.shared() is instinct.shared()


def test_a_malformed_points_section_from_the_environment_never_breaks_the_caller():
    raw = json.dumps({"providers": {"local": {"kind": "local-decision.v1", "url": URL}}, "fast": "local",
                      "points": {"opt_out": {"bars": {"yes": "hot"}}, "no_reminders": 3}})
    made = Instinct.from_environment({"COLONYMIND_INSTINCT": raw})
    assert made.provider_for("fast").url == URL and made.points_config == {}


async def test_an_unusable_url_is_no_answer():
    made = Instinct(providers={"local": LocalDecision("local", url="not a url at all")}, fast="local")
    made.set_state("opt_out", "fast", "live", source="test")
    assert (await made.decide("opt_out", {"text": "leave it"})).applied is False
    assert counts(made, "opt_out")["failed"] == 1
