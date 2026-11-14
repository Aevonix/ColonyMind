"""The ``local-decision.v1`` provider: a non-generative decision model on the operator's own network.

A decision model reads a short text and answers one choice question about it in a single forward pass. Options
are lettered in declared order, including yes/no points. Only descriptions and the unchanged instructions reach
the model; label names stay in the sidecar. It generates no text, only probabilities over the letters.

The HTTP contract: ``POST <url>/v1/decide`` with ``{"state", "questions": {"decision": {"type": "choice",
"instructions", "criteria": {letter: description}}}}``; the answer carries ``answers["decision"]`` with
``probabilities`` over exactly those letters, and ``"input_truncated": false``. One question per request: the server answers the one
question it is asked. Anything the contract does not promise is an error (``answer_probabilities``), and so is a
timeout or a failing endpoint: the breaker counts it and the point keeps its existing path.

The served revision is what the answer's ``backend`` reports (``checkpoint`` and ``model_revision``), "" when it
reports neither: a calibration record applies only to the revision it was measured on.
The readout is also part of the calibration identity: records measured with label names cannot apply to letters.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import httpx

from colonymind import usage

from . import Answer

KIND = "local-decision.v1"
PROTOCOL = KIND
DEFAULT_TIMEOUT_MS = 250
#: The time limits one answer may be given, in milliseconds (``colonymind.yaml`` accepts the same). Anything else,
#: a non-finite number included, is refused for the default: no setting can leave a call without a deadline.
TIMEOUT_MS_BOUNDS = (1, 5000)
#: The longest state sent: about 300 tokens of English, inside the 512-token context with the question and its
#: options. A longer state is never cut to fit; the tier is skipped.
MAX_STATE_CHARS = 1200
QUESTION = "decision"
#: How far a choice's probabilities may sum from one (the contract's rounding) and still be a distribution.
SUM_TOLERANCE = 0.01
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# The measured readout uses one declared order. There is no rotation setting.
ROTATIONS = 1


@dataclass(frozen=True)
class WireQuestion:
    """A point's declared labels and type kept locally; ``question`` sends only letters and descriptions."""

    name: str
    kind: str
    instructions: str
    labels: Mapping[str, str]
    state: str

    @classmethod
    def of(cls, point: Any) -> "WireQuestion":
        return cls(point.name, point.type, point.instructions, dict(point.labels), point.template)

    def question(self) -> Dict[str, Any]:
        return _question(self.instructions, self.labels)


def _question(instructions: str, labels: Mapping[str, str]) -> Dict[str, Any]:
    if not 2 <= len(labels) <= len(LETTERS):
        raise ValueError("a letter question needs two or more options within the alphabet")
    return {"type": "choice", "instructions": instructions,
            "criteria": dict(zip(LETTERS, labels.values()))}


def _probability(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def answer_probabilities(spec: Any, body: Any) -> Dict[str, float]:
    """Read the letters in a ``local-decision.v1`` answer back to ``spec.labels`` in their posed order, including
    a yes/no point's declared order. ValueError for anything the contract does not promise:
    another protocol, an input the model did not read whole (``input_truncated`` must be ``false``), answers to
    other questions than the one asked, or an answer that is not the typed answer asked for (a choice's
    probabilities must sum to one within ``SUM_TOLERANCE``). The sidecar and the measurement read answers only
    through this, so a measured answer is one the sidecar would have acted on."""
    if not isinstance(body, dict) or body.get("protocol") != PROTOCOL:
        raise ValueError("not a decision answer")
    if body.get("input_truncated") is not False:
        raise ValueError("the input was not read whole")
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != {QUESTION} or not isinstance(answers[QUESTION], dict):
        raise ValueError("not an answer to the question asked")
    answer = answers[QUESTION]
    raw = answer.get("probabilities")
    letters = LETTERS[:len(spec.labels)]
    if answer.get("type") != "choice" or not isinstance(raw, dict) or set(raw) != set(letters):
        raise ValueError("not a choice over the posed letters")
    values = {label: _probability(raw[letter]) for letter, label in zip(letters, spec.labels)}
    if any(value is None for value in values.values()):
        raise ValueError("a probability is out of range")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=SUM_TOLERANCE):
        raise ValueError("the probabilities are not a distribution")
    return values


def served_revision(body: Any) -> str:
    """The revision an answer's ``backend`` reports: ``<checkpoint>@<model_revision>``, either alone, or ""."""
    backend = body.get("backend") if isinstance(body, dict) else None
    if not isinstance(backend, dict):
        return ""
    parts = [str(backend[key]).strip() for key in ("checkpoint", "model_revision")
             if isinstance(backend.get(key), str) and str(backend[key]).strip()]
    return "@".join(parts)[:200]


def usable_timeout_ms(value: Any) -> int:
    """A usable time limit in milliseconds (``TIMEOUT_MS_BOUNDS``), or the default one for anything else."""
    low, high = TIMEOUT_MS_BOUNDS
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not low <= value <= high:
        return DEFAULT_TIMEOUT_MS
    return int(value) if float(value).is_integer() else int(math.ceil(value))


class _Spec:
    def __init__(self, question: Mapping[str, Any]) -> None:
        self.kind = question["type"]
        self.labels = dict(question["labels"])


class LocalDecision:
    """A ``local-decision.v1`` server: one order per question, concurrent questions within one deadline."""

    kind = KIND
    readout = "letters-r1"
    private = True
    max_state_chars = MAX_STATE_CHARS
    types = frozenset({"yes_no", "choice"})
    batch = False

    def __init__(self, id: str, *, url: str, timeout_ms: Any = DEFAULT_TIMEOUT_MS, model: str = "",
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.id = str(id)
        self.url = str(url or "").strip().rstrip("/")
        self.timeout_ms = usable_timeout_ms(timeout_ms)
        self.model = str(model or "")
        self.transport = transport

    async def answer(self, state: str, questions: Mapping[str, Mapping[str, Any]]) -> Answer:
        if len(state) > self.max_state_chars:
            raise ValueError("the state exceeds the decision model's limit")
        names = list(questions)
        async with asyncio.timeout(self.timeout_ms / 1000.0):
            tasks = [asyncio.create_task(self._one(name, state, questions[name])) for name in names]
            try:
                results = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        revisions = {revision for _, revision in results}
        return Answer({name: probabilities for name, (probabilities, _) in zip(names, results)},
                      revision=revisions.pop() if len(revisions) == 1 else "")

    async def _one(self, name: str, state: str, question: Mapping[str, Any]) -> tuple:
        seconds = self.timeout_ms / 1000.0
        async with httpx.AsyncClient(transport=self.transport, timeout=seconds) as client:
            # Each request records usage under the local point name; no point or label name enters the body.
            with usage.call("instinct", task=name, role=self.id, endpoint=self.id):
                wire = _question(question["instructions"], question["labels"])
                response = await client.post(f"{self.url}/v1/decide", json={"state": state, "questions": {QUESTION: wire}})
                response.raise_for_status()
                body = response.json()
                return answer_probabilities(_Spec(question), body), served_revision(body)


__all__ = ["DEFAULT_TIMEOUT_MS", "KIND", "LETTERS", "MAX_STATE_CHARS", "PROTOCOL", "ROTATIONS",
           "SUM_TOLERANCE", "TIMEOUT_MS_BOUNDS", "LocalDecision", "WireQuestion", "answer_probabilities",
           "served_revision", "usable_timeout_ms"]
