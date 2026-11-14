"""A client for a hosted typed-decision service's wire format: the bake-off's optional yardstick arm.

**Off by default, synthetic only, never part of the sidecar.** It lives here, outside the ``colonymind`` package: no
sidecar module imports it, the wheel does not carry it, and ``colonymind.yaml`` refuses its kind, so a running
sidecar can never construct it and nothing an agent reads or hears can reach it. The bake-off asks it only when an
operator names the endpoint, the model and the environment variable holding the key on the command line
(``measure.py decide --provider hosted``); it refuses any state outside the synthetic allowlist the runner builds
from the frozen sets, so only the repository's synthetic items can leave the machine.

Its answers are a yardstick: they are never labels, never fit another provider's calibration (each arm is fitted
on its own answers against the sets' constructed labels) and every number from it is written only to the private
output (``measure.py bakeoff --private-out``), never to the committed results.

The wire: ``POST <url>`` with a Bearer key and ``{"state", "model", "questions": {name: {"type": "choice"|"noul",
"instructions", "criteria"}}}``; the answer carries ``answers[name]`` with ``probabilities`` (a choice) or ``noul``
(the probability of yes), and ``model`` (the revision served). Many questions go in one request.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, Iterable, Mapping, Optional

import httpx

from colonymind.instinct.providers import Answer

KIND = "hosted-typed.v1"
DEFAULT_TIMEOUT_MS = 1000
#: The longest state sent (my choice): what is not needed is not sent.
MAX_STATE_CHARS = 4000
SUM_TOLERANCE = 0.01


class NotSynthetic(ValueError):
    """A state that is not one of the frozen synthetic items: never sent."""


def state_hash(state: str) -> str:
    return hashlib.sha256(str(state).encode("utf-8")).hexdigest()


def _wire(question: Mapping[str, Any]) -> Dict[str, Any]:
    labels = dict(question["labels"])
    if question["type"] == "yes_no":
        return {"type": "noul", "instructions": question["instructions"],
                "criteria": {"false": labels["no"], "true": labels["yes"]}}
    return {"type": "choice", "instructions": question["instructions"], "criteria": labels}


def _probability(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) if 0.0 <= value <= 1.0 else None


def parse(question: Mapping[str, Any], answer: Any) -> Dict[str, float]:
    """One answer as label probabilities; ValueError for anything but the typed answer asked for."""
    if not isinstance(answer, dict):
        raise ValueError("no answer to the question")
    if question["type"] == "yes_no":
        p_yes = _probability(answer.get("noul"))
        if p_yes is None:
            raise ValueError("not a yes/no answer")
        return {"yes": p_yes, "no": 1.0 - p_yes}
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or set(raw) != set(question["labels"]):
        raise ValueError("not a choice over the question's labels")
    values = {label: _probability(value) for label, value in raw.items()}
    if any(value is None for value in values.values()) or not math.isclose(sum(values.values()), 1.0,
                                                                           abs_tol=SUM_TOLERANCE):
        raise ValueError("not a distribution over the labels")
    return values


class HostedTyped:
    """The yardstick's client; built only with every setting given and a non-empty synthetic allowlist."""

    kind = KIND
    private = False
    types = frozenset({"yes_no", "choice"})
    batch = True
    max_state_chars = MAX_STATE_CHARS

    def __init__(self, id: str, *, url: str, model: str, key_env: str, synthetic: Iterable[str],
                 timeout_ms: int = DEFAULT_TIMEOUT_MS, environ: Optional[Mapping[str, str]] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.id, self.url, self.model = str(id), str(url or "").strip(), str(model or "").strip()
        if not self.url.startswith("https://") or not self.model:
            raise ValueError("the hosted yardstick needs an https url and a pinned model")
        environ = os.environ if environ is None else environ
        self._key = str(environ.get(str(key_env or "")) or "")
        if not self._key:
            raise ValueError(f"the key's environment variable {key_env or '(none)'} is not set")
        self.synthetic = frozenset(synthetic)
        if not self.synthetic:
            raise ValueError("the hosted yardstick needs the synthetic allowlist of the frozen sets")
        self.timeout_ms = int(timeout_ms)
        self.transport = transport

    async def answer(self, state: str, questions: Mapping[str, Mapping[str, Any]]) -> Answer:
        if state_hash(state) not in self.synthetic:
            raise NotSynthetic("the state is not one of the frozen synthetic items; nothing was sent")
        if len(str(state)) > self.max_state_chars:
            raise ValueError("the state is longer than the yardstick is sent; it is never cut")
        body = {"state": str(state), "model": self.model,
                "questions": {name: _wire(question) for name, question in questions.items()}}
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout_ms / 1000.0) as client:
            response = await client.post(self.url, json=body, headers={"Authorization": f"Bearer {self._key}"})
            response.raise_for_status()
            payload = response.json()
        answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(answers, dict):
            raise ValueError("not a typed answer")
        revision = payload.get("model") if isinstance(payload.get("model"), str) else ""
        return Answer({name: parse(question, answers.get(name)) for name, question in questions.items()},
                      revision=str(revision)[:200])


__all__ = ["DEFAULT_TIMEOUT_MS", "KIND", "MAX_STATE_CHARS", "HostedTyped", "NotSynthetic", "parse", "state_hash"]
