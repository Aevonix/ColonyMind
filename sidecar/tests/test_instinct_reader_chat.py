"""The reader benchmark selects its route without probing inference endpoints."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from colonymind.instinct.providers import logits


def _measure():
    directory = Path(__file__).resolve().parents[2] / "benchmarks" / "decisions"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("reader_chat_measure", directory / "measure.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))


measure = _measure()


def _args(**values):
    return SimpleNamespace(**{"provider": "logits", "url": "http://reader.test/v1", "model": "served-model",
                              "timeout_ms": 0, "reader_mode": "top", "rotations": 4, **values})


@pytest.mark.parametrize("options, expected", [({}, ("completions", None)),
                                              ({"reader_route": "chat", "reader_thinking": "off"},
                                               ("chat", "off"))])
def test_build_provider_forwards_the_reader_route_and_thinking(monkeypatch, options, expected):
    received = {}
    sentinel = object()

    def constructor(id, **kwargs):
        received.update(id=id, **kwargs)
        return sentinel

    monkeypatch.setattr(logits, "Logits", constructor)
    transport = httpx.MockTransport(lambda request: pytest.fail("building a reader must not contact its server"))
    assert measure.build_provider(_args(rotations=2, **options), [], None, transport=transport) is sentinel
    assert (received["route"], received["thinking"]) == expected
    assert received["rotations"] == 2 and received["mode"] == "top"
    assert received["transport"] is transport


@pytest.mark.parametrize("step", ["decide", "check-reader"])
@pytest.mark.parametrize("options, expected", [([], ("completions", None)),
                                              (["--reader-route", "chat", "--reader-thinking", "off"],
                                               ("chat", "off"))])
def test_reader_cli_accepts_the_explicit_chat_route_and_keeps_the_completions_default(
        monkeypatch, step, options, expected, tmp_path):
    received = []
    monkeypatch.setattr(measure, "decide" if step == "decide" else "check_reader", received.append)
    argv = [step, "--url", "http://reader.test/v1", "--model", "served-model", *options]
    if step == "decide":
        argv += ["--provider", "logits", "--out", str(tmp_path / "answers.json")]
    measure.main(argv)
    assert len(received) == 1
    assert (received[0].reader_route, received[0].reader_thinking) == expected


@pytest.mark.parametrize("route, thinking", [("completions", None), ("chat", "off")])
@pytest.mark.parametrize("model, config_text, support, outcome", [
    ("served-model", "max_logprobs: 20\n", "yes", "runs"),
    ("served-model", "max_logprobs: 0\n", "no", "dropped and recorded"),
    ("served-model", None, "unknown", "dropped and recorded"),
    ("missing-model", "max_logprobs: 20\n", "yes", "dropped and recorded"),
])
def test_reader_check_uses_only_the_model_list_and_configuration_for_either_route(
        tmp_path, route, thinking, model, config_text, support, outcome):
    seen = []

    def handler(request):
        assert request.method == "GET", "reader checks must never post a chat or completion probe"
        assert request.url.path == "/v1/models"
        seen.append(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "served-model"}]})

    config = None
    if config_text is not None:
        config = tmp_path / "serving.yaml"
        config.write_text(config_text)
    args = _args(model=model, serving_config=config, reader_route=route, reader_thinking=thinking)
    verdict = measure.check_reader(args, transport=httpx.MockTransport(handler))
    assert verdict == {"model_served": model == "served-model", "logprobs": support, "arm_e": outcome}
    assert seen == ["/v1/models"]


@pytest.mark.parametrize("options", [{"reader_route": "chat"},
                                    {"reader_route": "chat", "reader_thinking": "on"},
                                    {"reader_route": "chat", "reader_thinking": "off", "reader_mode": "echo"}])
def test_reader_check_refuses_invalid_chat_settings_before_contacting_the_server(options):
    def handler(request):
        pytest.fail("invalid reader settings must fail before any model-list request")

    with pytest.raises(ValueError):
        measure.check_reader(_args(**options), transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("data", [1, True, "served-model", {"id": "served-model"}])
def test_reader_check_malformed_model_list_drops_the_arm(tmp_path, data):
    config = tmp_path / "serving.yaml"
    config.write_text("max_logprobs: 20\n")

    def handler(request):
        assert request.method == "GET" and request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": data})

    verdict = measure.check_reader(_args(serving_config=config), transport=httpx.MockTransport(handler))
    assert verdict == {"model_served": False, "logprobs": "yes", "arm_e": "dropped and recorded"}


@pytest.mark.parametrize("route", ["completions", "chat"])
async def test_benchmark_answers_use_the_route_identity_and_renaming_keeps_the_wire(tmp_path, monkeypatch, route):
    arms, catalog, questions, report = measure._bakeoff_modules()
    built = {"opt_out": catalog.frozen(names=["opt_out"])["opt_out"][:1]}
    monkeypatch.setattr(measure, "_frozen", lambda *args: built)
    seen = []

    def handler(request):
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"data": [{"id": "served-model"}]})
        body = json.loads(request.content)
        seen.append(body)
        top = {" A": math.log(0.8), " B": math.log(0.2)}
        if route == "chat":
            assert request.url.path == "/v1/chat/completions"
            choice = {"message": {"content": " A"}, "logprobs": {"content": [
                {"token": " A", "logprob": math.log(0.8),
                 "top_logprobs": [{"token": token, "logprob": value} for token, value in top.items()]}]}}
        else:
            assert request.url.path == "/v1/completions"
            choice = {"logprobs": {"top_logprobs": [top]}}
        return httpx.Response(200, json={"choices": [choice], "system_fingerprint": "fixture-revision"})

    config = tmp_path / "serving.yaml"
    config.write_text("max_logprobs: 20\n")
    out = tmp_path / "answers.json"
    args = _args(reader_route=route, reader_thinking="off" if route == "chat" else None,
                 sets=[], serving_config=config, perturb=["base", "renamed"], concurrency=1, out=out)
    await measure.decide_bakeoff(args, transport=httpx.MockTransport(handler))
    document = json.loads(out.read_text())
    assert document["provider"]["kind"] == f"logits/names-free/{route}/r4"
    assert document["perturbations"]["base"].keys() == document["perturbations"]["renamed"].keys()
    for key, base in document["perturbations"]["base"].items():
        assert base["raw"] == document["perturbations"]["renamed"][key]["raw"] == {"yes": 0.5, "no": 0.5}
    assert seen[:2] == seen[2:]
