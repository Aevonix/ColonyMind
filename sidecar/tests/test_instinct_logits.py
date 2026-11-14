"""The reader provider (``colonymind.instinct.providers.logits``): an LLM already served, read as option
probabilities.

Every test talks to a fake OpenAI-compatible server (``httpx.MockTransport``): no model endpoint is
contacted."""

from __future__ import annotations

import json
import math
import re

import httpx
import pytest

from colonymind.instinct.calibration import Record
from colonymind.instinct.providers import Answer
from colonymind.instinct.providers import logits
from colonymind.instinct.providers.logits import Logits, NoAnswer, Unavailable
from colonymind.instinct.registry import Point, Tier, register
from colonymind.instinct.tiers import Instinct

YES_NO = {"type": "yes_no", "instructions": "Does the sender ask to receive no more messages?",
          "labels": {"yes": "they want no more messages at all", "no": "anything else"}}
CHOICE = {"type": "choice", "instructions": "How does the owner answer?",
          "labels": {"engaged": "welcomes it", "dig_deeper": "asks for more", "not_now": "busy right now",
                     "stop": "wants no more messages"}}
_OPTION = re.compile(r"^([A-H])\. (.+)$", re.MULTILINE)
_MEANINGS = {description: label for question in (YES_NO, CHOICE)
             for label, description in question["labels"].items()}
_MEANINGS["does not want the topic"] = "not_interested"


def options_of(prompt):
    """Read each fixture's meaning from its description; no label name is on the wire."""
    return {letter: _MEANINGS[description] for letter, description in _OPTION.findall(prompt)}


class Server:
    """A fake completions server. ``content(labels) -> {label: p}`` is what the model means; ``position`` mixes in
    a bias toward a letter whatever it stands for; ``mass`` is the share of the next-token mass on the letters."""

    def __init__(self, content, *, position=None, bias=0.0, mass=0.98, logprobs=True, fingerprint="fp-1",
                 variants=False):
        self.content, self.position, self.bias, self.mass = content, position or {}, bias, mass
        self.logprobs, self.fingerprint, self.variants = logprobs, fingerprint, variants
        self.requests = []

    def letters(self, prompt):
        shown = options_of(prompt)
        meaning = self.content(list(shown.values()))
        out = {}
        for letter, label in shown.items():
            out[letter] = (1 - self.bias) * meaning.get(label, 0.0) + self.bias * self.position.get(letter, 0.0)
        return out

    def handler(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        prompt = body["prompt"]
        choice = {"index": 0, "text": " A", "finish_reason": "length"}
        if self.logprobs:
            top = {"The": math.log(max(1e-9, 1 - self.mass))}
            for letter, p in self.letters(prompt).items():
                if p <= 0:
                    continue
                if self.variants:                   # the same letter under two tokens: " A" and "A"
                    top[f" {letter}"] = math.log(self.mass * p * 0.75)
                    top[letter] = math.log(self.mass * p * 0.25)
                else:
                    top[f" {letter}"] = math.log(self.mass * p)
            choice["logprobs"] = {"tokens": [" A"], "token_logprobs": [top.get(" A", -9.0)],
                                  "top_logprobs": [top], "text_offset": [len(prompt)]}
        else:
            choice["logprobs"] = None
        return httpx.Response(200, json={"id": "c-1", "object": "text_completion", "model": body["model"],
                                         "system_fingerprint": self.fingerprint, "choices": [choice]})

    def transport(self):
        return httpx.MockTransport(self.handler)


def reader(server, **kwargs):
    return Logits("reader", url="http://reader.test/v1", model="served-model", transport=server.transport(),
                  **kwargs)


def by_meaning(table):
    return lambda labels: {label: table.get(label, 0.0) for label in labels}


async def test_letter_log_probabilities_become_a_distribution_over_the_labels():
    server = Server(by_meaning({"engaged": 0.1, "dig_deeper": 0.7, "not_now": 0.15, "stop": 0.05}))
    answer = await reader(server).answer("The owner said: \"tell me more\"", {"outreach_reply": CHOICE})
    assert isinstance(answer, Answer) and answer.revision == "fp-1"
    got = answer.probabilities["outreach_reply"]
    assert set(got) == set(CHOICE["labels"]) and math.isclose(sum(got.values()), 1.0, abs_tol=1e-9)
    assert got["dig_deeper"] == pytest.approx(0.7) and got["stop"] == pytest.approx(0.05)
    # Only the completions route, one token, greedy, with next-token log-probabilities: never a chat call.
    assert {path for path, _ in server.requests} == {"/v1/completions"}
    for _, body in server.requests:
        assert body["max_tokens"] == 1 and body["temperature"] == 0 and body["logprobs"] >= len(CHOICE["labels"])
        assert body["model"] == "served-model" and "messages" not in body


async def test_a_letter_under_two_tokens_counts_once_with_both_shares():
    server = Server(by_meaning({"yes": 0.8, "no": 0.2}), variants=True)
    got = (await reader(server).answer("Message: \"stop texting me\"", {"opt_out": YES_NO})).probabilities["opt_out"]
    assert got["yes"] == pytest.approx(0.8)


async def test_rotations_average_the_order_back_to_the_right_labels():
    """A model that half follows the meaning and half favours letter A: every order is asked (at most four), each
    letter is mapped back to the label it stood for, and the average cancels the letter's pull."""
    server = Server(by_meaning({"yes": 0.9, "no": 0.1}), position={"A": 0.8, "B": 0.2}, bias=0.5)
    got = (await reader(server).answer("Message: \"no more please\"", {"opt_out": YES_NO})).probabilities["opt_out"]
    orders = [tuple(options_of(body["prompt"]).values()) for _, body in server.requests]
    assert sorted(orders) == [("no", "yes"), ("yes", "no")]
    # order yes,no: yes 0.5*0.9 + 0.5*0.8 = 0.85; order no,yes: yes 0.5*0.9 + 0.5*0.2 = 0.55; the mean is 0.7
    assert got["yes"] == pytest.approx(0.7) and got["no"] == pytest.approx(0.3)


async def test_at_most_four_rotations_and_each_is_a_rotation_of_the_labels():
    server = Server(by_meaning({"engaged": 0.25, "dig_deeper": 0.25, "not_now": 0.25, "stop": 0.25}))
    five = {**CHOICE, "labels": {**CHOICE["labels"], "not_interested": "does not want the topic"}}
    await reader(server).answer("x", {"outreach_reply": five})
    orders = [tuple(options_of(body["prompt"]).values()) for _, body in server.requests]
    base = list(five["labels"])
    assert len(orders) == logits.MAX_ROTATIONS == 4
    assert sorted(orders) == sorted(tuple(base[r:] + base[:r]) for r in range(4))
    single = Server(by_meaning({"yes": 1.0}))
    await reader(single, rotations=1).answer("x", {"opt_out": YES_NO})
    assert len(single.requests) == 1


async def test_letter_mass_under_the_floor_is_no_answer():
    server = Server(by_meaning({"yes": 0.9, "no": 0.1}), mass=0.4)
    with pytest.raises(NoAnswer):
        await reader(server).answer("Message: \"hm\"", {"opt_out": YES_NO})
    assert logits.LETTER_FLOOR == 0.5
    # Just above the floor it answers.
    await reader(Server(by_meaning({"yes": 0.9, "no": 0.1}), mass=0.55)).answer("x", {"opt_out": YES_NO})


async def test_a_server_without_log_probabilities_is_unavailable_never_a_guess():
    server = Server(by_meaning({"yes": 1.0}), logprobs=False)
    with pytest.raises(Unavailable):
        await reader(server).answer("Message: \"stop\"", {"opt_out": YES_NO})


def top_server(top_of):
    """A completions server whose next-token alternatives are ``top_of({letter: label})``: malformed ones too."""
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        payload = {"system_fingerprint": "fp-1", "choices": [
            {"index": 0, "logprobs": {"top_logprobs": [top_of(options_of(body["prompt"]))]}}]}
        # Written as some servers write them: NaN and infinities as bare literals, which Python's reader accepts.
        return httpx.Response(200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})

    return requests, httpx.MockTransport(handler)


def letter_of(letters, label):
    return next(letter for letter, shown in letters.items() if shown == label)


@pytest.mark.parametrize("bad", ["invalid", None, float("nan"), float("inf"), True, 2.0, 0.5])
async def test_a_malformed_log_probability_is_refused_never_normalised_into_an_answer(bad):
    """The yes letter at log(0.6) beside a no letter whose value is not a log-probability (not a number, not
    finite, or positive) is not an answer of yes at 1.0: the server's answer is malformed."""
    _, transport = top_server(lambda letters: {f" {letter_of(letters, 'yes')}": math.log(0.6),
                                               f" {letter_of(letters, 'no')}": bad})
    provider = Logits("reader", url="http://reader.test/v1", model="m", transport=transport)
    with pytest.raises(logits.Malformed):
        await provider.answer("Message: \"stop\"", {"opt_out": YES_NO})


async def test_next_token_mass_above_one_is_refused_and_minus_infinity_is_zero():
    _, over = top_server(lambda letters: {f" {letter_of(letters, 'yes')}": math.log(0.8),
                                          f" {letter_of(letters, 'no')}": math.log(0.7)})
    with pytest.raises(logits.Malformed):
        await Logits("reader", url="http://reader.test/v1", model="m", transport=over).answer("x", {"opt_out": YES_NO})
    # A token the server rules out (-inf) holds no mass; the rest is still an answer.
    _, ruled_out = top_server(lambda letters: {f" {letter_of(letters, 'yes')}": math.log(0.9),
                                               f" {letter_of(letters, 'no')}": float("-inf"), "The": math.log(0.05)})
    got = await Logits("reader", url="http://reader.test/v1", model="m", transport=ruled_out).answer(
        "x", {"opt_out": YES_NO})
    assert got.probabilities["opt_out"]["yes"] == pytest.approx(1.0)
    # Rounding slack in the server's numbers is not an error.
    _, rounded = top_server(lambda letters: {f" {letter_of(letters, 'yes')}": 0.00001,
                                             f" {letter_of(letters, 'no')}": -12.0})
    await Logits("reader", url="http://reader.test/v1", model="m", transport=rounded).answer("x", {"opt_out": YES_NO})


async def test_a_malformed_answer_is_a_provider_failure_to_the_engine_never_a_decision():
    point = Point(name="probe_yes_no", type="yes_no", instructions=YES_NO["instructions"],
                  labels=dict(YES_NO["labels"]), template='Message: "{text}"', effect="read",
                  tiers=(Tier("reader", may_settle=("yes", "no"), state="live"),))
    table = {}
    register(point, registry=table)
    _, transport = top_server(lambda letters: {f" {letter_of(letters, 'yes')}": math.log(0.6),
                                               f" {letter_of(letters, 'no')}": "invalid"})
    provider = Logits("reader", url="http://reader.test/v1", model="served-model", transport=transport)
    record = Record(point=point.name, version=point.version, tier="reader",
                    provider_kind=f"logits/{provider.readout}",
                    model="served-model", revision="fp-1", bars={"yes": 0.8, "no": 0.8}, source="test")
    made = Instinct(providers={"reader": provider}, reader="reader", records=(record,), registry=table)
    decision = await made.decide(point.name, {"text": "stop"})
    assert decision.applied is False and decision.tier == "fallback" and decision.reason == "fallback_failed"
    assert made._breaker(provider).consecutive == 1


async def test_echo_mode_refuses_a_positive_log_probability_and_mass_above_one():
    def server(value_of):
        def handler(request):
            body = json.loads(request.content)
            choices = []
            for index, prompt in enumerate(body["prompt"]):
                base, letter = prompt[:-2], prompt[-1]
                choices.append({"index": index, "logprobs": {
                    "tokens": [base, f" {letter}"], "token_logprobs": [None, value_of(options_of(base)[letter])],
                    "text_offset": [0, len(base)]}})
            return httpx.Response(200, json={"choices": choices})
        return httpx.MockTransport(handler)

    for value_of in (lambda label: 0.7 if label == "yes" else -3.0,                  # a positive log-probability
                     lambda label: math.log(0.8) if label == "yes" else math.log(0.6)):  # mass above one
        provider = Logits("reader", url="http://reader.test/v1", model="m", transport=server(value_of), mode="echo")
        with pytest.raises(logits.Malformed):
            await provider.answer("x", {"opt_out": YES_NO})


async def test_a_state_longer_than_the_reader_reads_is_refused_never_cut():
    server = Server(by_meaning({"yes": 1.0}))
    provider = reader(server)
    with pytest.raises(ValueError):
        await provider.answer("x" * (provider.max_state_chars + 1), {"opt_out": YES_NO})
    assert server.requests == []
    await provider.answer("y" * provider.max_state_chars, {"opt_out": YES_NO})
    assert all(("y" * provider.max_state_chars) in body["prompt"] for _, body in server.requests)


async def test_the_prompt_carries_the_state_question_and_every_option_description():
    server = Server(by_meaning({"yes": 1.0}))
    await reader(server, rotations=1).answer("Message: \"please stop\"", {"opt_out": YES_NO})
    prompt = server.requests[0][1]["prompt"]
    assert 'Message: "please stop"' in prompt and YES_NO["instructions"] in prompt
    assert "A. they want no more messages at all" in prompt and "B. anything else" in prompt
    assert prompt.rstrip().endswith("Answer:")


def echo_server(meaning, *, mass=0.95):
    """A server that offers only echoed prompt log-probabilities: one prompt per option letter, the letter's own
    log-probability read at its offset."""
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        choices = []
        for index, prompt in enumerate(body["prompt"]):
            base, letter = prompt[:-2], prompt[-1]
            shown = options_of(base)
            p = meaning.get(shown[letter], 0.0) * mass
            tokens = [base[:10], base[10:], f" {letter}", "."]
            offsets = [0, 10, len(base), len(base) + 2]
            choices.append({"index": index, "text": prompt + ".", "logprobs": {
                "tokens": tokens, "token_logprobs": [None, -1.0, math.log(max(p, 1e-12)), -0.1],
                "top_logprobs": None, "text_offset": offsets}})
        return httpx.Response(200, json={"model": body["model"], "choices": choices})

    return requests, httpx.MockTransport(handler)


async def test_echo_mode_reads_each_letter_from_the_prompt_log_probabilities():
    requests, transport = echo_server({"yes": 0.75, "no": 0.25})
    provider = Logits("reader", url="http://reader.test/v1", model="m", transport=transport, mode="echo")
    got = (await provider.answer("Message: \"stop\"", {"opt_out": YES_NO})).probabilities["opt_out"]
    assert got["yes"] == pytest.approx(0.75)
    assert all(body["echo"] is True and body["max_tokens"] == 1 and len(body["prompt"]) == 2 for body in requests)
    low_requests, low = echo_server({"yes": 0.75, "no": 0.25}, mass=0.3)
    with pytest.raises(NoAnswer):
        await Logits("reader", url="http://reader.test/v1", model="m", transport=low, mode="echo").answer(
            "x", {"opt_out": YES_NO})


def test_the_provider_has_the_protocol_the_engine_reads():
    provider = Logits("reader", url="http://reader.test/v1/", model="m")
    assert provider.kind == logits.KIND == "logits" and provider.url == "http://reader.test/v1"
    assert provider.types == frozenset({"yes_no", "choice"}) and provider.max_state_chars == logits.MAX_STATE_CHARS
    assert provider.timeout_ms == logits.DEFAULT_TIMEOUT_MS == 800
    assert Logits("r", url="http://x/v1", model="m", timeout_ms=10 ** 9).timeout_ms == logits.DEFAULT_TIMEOUT_MS
    assert Logits("r", url="http://x/v1", model="m", timeout_ms=float("nan")).timeout_ms == logits.DEFAULT_TIMEOUT_MS
    with pytest.raises(ValueError):
        Logits("r", url="http://x/v1", model="m", mode="chat")
    with pytest.raises(ValueError):
        Logits("r", url="http://x/v1", model="m", rotations=0)
    # A setting entry (``instinct.providers.<name>``) builds the same provider.
    built = logits.from_entry("reader", {"kind": "logits", "url": "http://reader.test/v1", "model": "m",
                                         "timeout_ms": 900})
    assert built.kind == "logits" and built.model == "m" and built.timeout_ms == 900
    assert logits.from_entry("reader", {"kind": "logits", "url": "", "model": "m"}) is None
    assert logits.from_entry("reader", {"kind": "local-decision.v1", "url": "http://x/v1"}) is None


async def test_the_engine_settles_a_live_reader_tier_on_a_record_for_it():
    point = Point(name="probe_choice", type="choice", instructions="How does the owner answer?",
                  labels=dict(CHOICE["labels"]), template='The owner said: "{text}"', effect="read",
                  tiers=(Tier("reader", may_settle=tuple(CHOICE["labels"]), state="live"),))
    table = {}
    register(point, registry=table)
    server = Server(by_meaning({"engaged": 0.02, "dig_deeper": 0.95, "not_now": 0.02, "stop": 0.01}))
    provider = reader(server)
    record = Record(point=point.name, version=point.version, tier="reader",
                    provider_kind=f"logits/{provider.readout}",
                    model="served-model", revision="fp-1", bars={label: 0.8 for label in point.labels},
                    source="test")
    made = Instinct(providers={"reader": provider}, reader="reader", records=(record,), registry=table)
    decision = await made.decide("probe_choice", {"text": "tell me more"})
    assert decision.applied and decision.tier == "reader" and decision.label == "dig_deeper"
    # Another served revision matches no record: the tier abstains.
    other = Server(by_meaning({"dig_deeper": 1.0}), fingerprint="fp-2")
    again = Instinct(providers={"reader": reader(other)}, reader="reader", records=(record,), registry=table)
    assert (await again.decide("probe_choice", {"text": "tell me more"})).applied is False
    # Below the floor the reader gives no answer, and the point keeps its fallback.
    low = Instinct(providers={"reader": reader(Server(by_meaning({"dig_deeper": 1.0}), mass=0.2))},
                   reader="reader", records=(record,), registry=table)
    assert (await low.decide("probe_choice", {"text": "tell me more"})).applied is False


READER_SECTION = {"providers": {"local": {"kind": "local-decision.v1", "url": "", "timeout_ms": 250},
                                "reader": {"kind": "logits", "url": "http://reader.test/v1", "model": "served-model",
                                           "timeout_ms": 900}},
                  "fast": "local", "reader": "reader"}


def test_a_configured_reader_is_built_by_the_engine_the_sidecar_constructs():
    """A valid ``kind: logits`` entry named as the reader tier is the reader the running sidecar asks: through
    ``Instinct.from_settings`` and ``from_environment`` (what ``colonymind.yaml`` exports), not only ``from_entry``."""
    from colonymind.instinct import settings
    assert settings.validate(READER_SECTION)["reader"] == "reader"
    for made in (Instinct.from_settings(READER_SECTION),
                 Instinct.from_environment({"COLONYMIND_INSTINCT": json.dumps(READER_SECTION)})):
        provider = made.provider_for("reader")
        assert isinstance(provider, Logits) and provider.model == "served-model"
        assert provider.url == "http://reader.test/v1" and provider.timeout_ms == 900
    # Without a model the entry is incomplete: no reader, and every point keeps its path.
    bare = {**READER_SECTION, "providers": {**READER_SECTION["providers"],
                                            "reader": {"kind": "logits", "url": "http://reader.test/v1"}}}
    assert Instinct.from_settings(bare).provider_for("reader") is None


def test_registered_points_show_only_descriptions_and_renaming_changes_no_prompt():
    from colonymind.instinct.registry import REGISTRY

    assert REGISTRY
    for point in REGISTRY.values():
        question = {"instructions": point.instructions, "labels": dict(point.labels)}
        renamed = {**question, "labels": {f"option_name_{index}": description
                                         for index, description in enumerate(point.labels.values())}}
        original_orders = logits.rotations(list(question["labels"]), logits.MAX_ROTATIONS)
        renamed_orders = logits.rotations(list(renamed["labels"]), logits.MAX_ROTATIONS)
        for order, renamed_order in zip(original_orders, renamed_orders):
            shown = logits.prompt("Text to classify.", question, order)
            assert shown == logits.prompt("Text to classify.", renamed, renamed_order), point.name
            assert _OPTION.findall(shown) == [
                (logits.LETTERS[index], " ".join(point.labels[label].split()))
                for index, label in enumerate(order)
            ], point.name
            assert "option_name_" not in shown


def chat_choice(top=None, *, token=" A", message=None):
    if top is None:
        top = [{"token": " A", "logprob": math.log(0.6)},
               {"token": "A.", "logprob": math.log(0.2)},
               {"token": " B", "logprob": math.log(0.15)},
               {"token": "The", "logprob": math.log(0.05)}]
    return {"index": 0, "message": {"role": "assistant", "content": token, **(message or {})},
            "logprobs": {"content": [{"token": token, "top_logprobs": top}]}}


def chat_reader(choice=None, **kwargs):
    requests = []

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        payload = {"system_fingerprint": "fp-chat", "choices": [choice if choice is not None else chat_choice()]}
        return httpx.Response(200, content=json.dumps(payload).encode(),
                              headers={"content-type": "application/json"})

    provider = Logits("reader", url="http://reader.test/v1", model="served-model", route="chat", thinking="off",
                      transport=httpx.MockTransport(handler), **kwargs)
    return provider, requests


async def test_chat_continues_the_prefilled_answer_with_exactly_one_token_and_thinking_off():
    provider, requests = chat_reader(rotations=1)
    answer = await provider.answer("Text to classify.", {"opt_out": YES_NO})
    base = logits.prompt("Text to classify.", YES_NO, list(YES_NO["labels"]))
    assert requests == [("/v1/chat/completions", {
        "model": "served-model", "messages": [{"role": "user", "content": base.rsplit("\n", 1)[0]},
                                                {"role": "assistant", "content": "Answer:"}],
        "continue_final_message": True, "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": logits.TOP_LOGPROBS,
    })]
    assert answer.revision == "fp-chat"
    assert answer.probabilities["opt_out"] == pytest.approx({"yes": 0.8 / 0.95, "no": 0.15 / 0.95})


async def test_chat_rotations_map_letters_to_their_descriptions_before_averaging():
    provider, requests = chat_reader()
    answer = await provider.answer("Text to classify.", {"outreach_reply": CHOICE})
    assert len(requests) == logits.MAX_ROTATIONS
    assert answer.probabilities["outreach_reply"] == pytest.approx(dict.fromkeys(CHOICE["labels"], 0.25))


@pytest.mark.parametrize("reasoning", ["reasoning", "reasoning_content"])
async def test_chat_with_reasoning_is_no_answer_even_when_the_letters_have_mass(reasoning):
    provider, _ = chat_reader(chat_choice(message={reasoning: "Reasoning was generated."}))
    with pytest.raises(NoAnswer):
        await provider.answer("Text to classify.", {"opt_out": YES_NO})


@pytest.mark.parametrize("token", ["<think>", "\n\n", "The", "I", "", None])
async def test_chat_next_token_outside_the_options_is_no_answer(token):
    provider, _ = chat_reader(chat_choice(token=token))
    with pytest.raises(NoAnswer):
        await provider.answer("Text to classify.", {"opt_out": YES_NO})


async def test_chat_thinking_distribution_and_low_letter_mass_are_no_answer():
    for token, mass in (("<think>", 0.0002), (" A", 0.49)):
        provider, _ = chat_reader(chat_choice([
            {"token": " A", "logprob": math.log(mass)},
            {"token": "<think>", "logprob": math.log(1 - mass)},
        ], token=token))
        with pytest.raises(NoAnswer):
            await provider.answer("Text to classify.", {"opt_out": YES_NO})


@pytest.mark.parametrize("logprobs", [None, {}, {"content": None}, {"content": []},
                                      {"content": [{"token": " A", "top_logprobs": []}]}])
async def test_chat_missing_log_probabilities_are_unavailable(logprobs):
    provider, _ = chat_reader({**chat_choice(), "logprobs": logprobs})
    with pytest.raises(Unavailable):
        await provider.answer("Text to classify.", {"opt_out": YES_NO})


@pytest.mark.parametrize("bad", ["invalid", None, float("nan"), float("inf"), True, 2.0])
async def test_chat_impossible_log_probability_is_malformed(bad):
    provider, _ = chat_reader(chat_choice([{"token": " A", "logprob": math.log(0.6)},
                                          {"token": " B", "logprob": bad}]))
    with pytest.raises(logits.Malformed):
        await provider.answer("Text to classify.", {"opt_out": YES_NO})


@pytest.mark.parametrize("bad", [None, "invalid", float("nan"), float("inf"), True, 2.0])
async def test_chat_impossible_selected_token_probability_is_malformed(bad):
    choice = chat_choice()
    choice["logprobs"]["content"][0]["logprob"] = bad
    provider, _ = chat_reader(choice)
    with pytest.raises(logits.Malformed):
        await provider.answer("Text to classify.", {"opt_out": YES_NO})


async def test_chat_more_than_one_returned_token_is_no_answer():
    choice = chat_choice()
    choice["logprobs"]["content"].append({"token": "<think>", "logprob": 0.0, "top_logprobs": []})
    provider, _ = chat_reader(choice)
    with pytest.raises(NoAnswer):
        await provider.answer("Text to classify.", {"opt_out": YES_NO})


@pytest.mark.parametrize("top", [
    [{"token": " A", "logprob": math.log(0.8)}, {"token": " B", "logprob": math.log(0.7)}],
    [{"token": " A", "logprob": math.log(0.6)}, {"token": "The", "logprob": 2.0}],
    [{"token": " A", "logprob": math.log(0.6)}, {"token": " A", "logprob": math.log(0.6)}],
    [{"token": " A"}], [{"logprob": -1.0}], [None],
])
async def test_chat_malformed_alternatives_are_never_silently_dropped(top):
    provider, _ = chat_reader(chat_choice(top))
    with pytest.raises(logits.Malformed):
        await provider.answer("Text to classify.", {"opt_out": YES_NO})


async def test_chat_empty_reasoning_fields_and_zero_probability_are_supported():
    provider, _ = chat_reader(chat_choice([
        {"token": " A", "logprob": 0.0}, {"token": " B", "logprob": float("-inf")},
    ], message={"reasoning": None, "reasoning_content": ""}), rotations=1)
    answer = await provider.answer("Text to classify.", {"opt_out": YES_NO})
    assert answer.probabilities["opt_out"] == {"yes": 1.0, "no": 0.0}


@pytest.mark.parametrize("options", [
    {"route": "invalid"}, {"route": "chat"}, {"route": "chat", "thinking": "on"},
    {"route": "chat", "thinking": False}, {"route": "chat", "thinking": "off", "mode": "echo"},
])
def test_invalid_chat_configuration_is_refused_before_any_request(options):
    with pytest.raises(ValueError):
        Logits("reader", url="http://reader.test/v1", model="served-model", **options)


def reader_section(**options):
    return {"providers": {"reader": {"kind": "logits", "url": "http://reader.test/v1", "model": "served-model",
                                      **options}}, "fast": "", "reader": "reader"}


@pytest.mark.parametrize("options", [
    {"route": "chat"}, {"route": "chat", "thinking": "on"}, {"route": "chat", "thinking": False},
    {"route": "chat", "thinking": None}, {"route": "chat", "thinking": "off", "mode": "echo"},
    {"route": "wrong"}, {"rotations": 0}, {"rotations": 5}, {"rotations": True}, {"rotations": 1.5},
])
def test_invalid_reader_entry_is_rejected_and_lenient_parse_leaves_it_off(options):
    from colonymind.instinct import settings

    section = reader_section(**options)
    with pytest.raises(settings.SettingsError):
        settings.validate(section)
    assert settings.parse(section)["providers"] == {}
    assert settings.parse(section)["reader"] == ""
    assert Instinct.from_settings(section).provider_for("reader") is None


@pytest.mark.parametrize("route,thinking", [("completions", None), ("chat", "off")])
@pytest.mark.parametrize("count", [1, logits.MAX_ROTATIONS])
def test_reader_settings_keep_route_thinking_and_rotations(route, thinking, count):
    from colonymind.instinct import settings

    options = {"route": route, "rotations": count}
    if thinking is not None:
        options["thinking"] = thinking
    section = reader_section(**options)
    checked = settings.validate(section)
    for key, value in options.items():
        assert checked["providers"]["reader"][key] == value
    for made in (Instinct.from_settings(checked),
                 Instinct.from_environment({"COLONYMIND_INSTINCT": json.dumps(section)})):
        provider = made.provider_for("reader")
        assert provider.route == route and provider.thinking == thinking and provider.rotations == count
        assert provider.kind == "logits" and provider.readout == f"names-free/{route}/r{count}"


@pytest.mark.parametrize("option", ["route", "thinking", "rotations"])
def test_reader_only_settings_are_not_accepted_by_other_provider_kinds(option):
    from colonymind.instinct import settings

    entry = {"kind": "local-decision.v1", "url": "", option: {"route": "chat", "thinking": "off",
                                                            "rotations": 1}[option]}
    with pytest.raises(settings.SettingsError):
        settings.validate({"providers": {"local": entry}})


@pytest.mark.parametrize("record_kind,route,count,matches", [
    ("logits/names-free/completions/r4", "completions", 4, True),
    ("logits/names-free/completions/r4", "chat", 4, False),
    ("logits/names-free/completions/r4", "completions", 1, False),
    ("logits/names-free/chat/r4", "chat", 4, True),
    ("logits/names-free/chat/r4", "completions", 4, False),
    ("logits/names-free/chat/r4", "chat", 1, False),
    ("logits", "completions", 4, False),
    ("logits", "chat", 4, False),
])
def test_reader_calibration_cannot_cross_route_rotation_count_or_legacy_readout(record_kind, route, count, matches):
    point = Point(name="probe_yes_no", type="yes_no", instructions=YES_NO["instructions"],
                  labels=dict(YES_NO["labels"]), template="Text: {text}", effect="read",
                  tiers=(Tier("reader", may_settle=("yes", "no"), state="live"),))
    record = Record(point=point.name, version=point.version, tier="reader", provider_kind=record_kind,
                    model="served-model", revision="fp-1", bars={"yes": 0.8, "no": 0.8}, source="test")
    made = Instinct(records=(record,), registry={point.name: point})
    provider = Logits("reader", url="http://reader.test/v1", model="served-model", route=route,
                      thinking="off" if route == "chat" else None, rotations=count)
    assert provider.kind == "logits" and provider.readout == f"names-free/{route}/r{count}"
    assert made.record_for(point, "reader", provider, "fp-1") == (record if matches else None)
