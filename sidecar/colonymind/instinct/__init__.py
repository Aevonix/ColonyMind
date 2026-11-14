"""Instinct: one mechanism for the quick, typed decisions of the mind (docs/DECISIONS.md).

A decision point is a short typed question with a closed set of labels and one effect from a closed enum; its
tiers answer in order (rules, then a fast decision model, a reader over an LLM already served, the mind's own
typed call, and the point's conservative fallback), and the first that settles decides. A model tier ships in
shadow (asked and logged, never applied) and acts only once promoted on evidence; a down or unsure provider leaves
the point on its existing path. Every decision is logged by reference, never with the words it read.

``await instinct.shared().decide(point, fields, refs, deliberate=...)`` is the call; ``shared()`` is the process's
engine, from ``COLONYMIND_INSTINCT`` (the ``instinct`` section of ``colonymind.yaml``) and the instance's state
directory. ``maintenance`` is the sidecar's daily task (the log's retention, and the model usage ledger's).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from colonymind import usage

from . import points
from .calibration import calibrate, calibrate_yes
from .log import DB_FILE, Hashed, Store
from .providers.local_decision import MAX_STATE_CHARS, WireQuestion, answer_probabilities
from .registry import EFFECTS, REGISTRY, Point, RegistrationError, RulesAnswer, Tier, register
from .tiers import Decision, Instinct

logger = logging.getLogger(__name__)

points.register_all()
# A point's usage rows carry its own name (``colonymind.usage``): the mind's points are registered names.
usage.register_tasks(REGISTRY)

#: The fast points as a ``local-decision.v1`` server is asked about them (the measurement's view).
FAST_POINTS: Dict[str, WireQuestion] = {name: WireQuestion.of(REGISTRY[name]) for name in points.FAST}

DAY_S = 24 * 3600.0

_SHARED: Dict[Tuple[str, str], Instinct] = {}


def _state_dir() -> Path:
    from colonymind import get_state_dir
    return get_state_dir()


def shared() -> Instinct:
    """The process's engine, rebuilt when its environment or state directory changes."""
    try:
        directory: Optional[Path] = _state_dir()
    except OSError as error:
        logger.warning("the state directory is unavailable (%s); decisions are not logged", type(error).__name__)
        directory = None
    key = (os.environ.get("COLONYMIND_INSTINCT", ""), str(directory or ""))
    if key not in _SHARED:
        for old in _SHARED.values():
            _close_later(old)
        _SHARED.clear()
        _SHARED[key] = Instinct.from_environment(state_dir=directory)
    return _SHARED[key]


def _close_later(engine: Instinct) -> None:
    try:
        asyncio.get_running_loop().create_task(engine.aclose())
    except RuntimeError:
        pass


async def aclose() -> None:
    """Cancel the shadow asks of the process's engine (the sidecar's shutdown)."""
    for engine in list(_SHARED.values()):
        await engine.aclose()


async def maintenance(state_dir: Optional[Path] = None, *,
                      sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep, first_delay: float = 60.0) -> None:
    """The sidecar's daily task: drop model usage rows past their retention (``colonymind.usage``), and decision
    rows past the log's retention (their counts are kept), then the nightly pass (``fit.nightly``: outcomes, the
    refit, demotions). Runs until cancelled; a failed pass is logged and the next one runs a day later."""
    await sleep(first_delay)
    while True:
        try:
            directory = Path(state_dir) if state_dir is not None else _state_dir()
            await _prune_usage(directory)
            path = directory / DB_FILE
            if path.exists():
                dropped = Store(path).prune()
                if dropped:
                    logger.info("instinct: %d decision rows past retention dropped", dropped)
                from colonymind.util.temporal import now_utc

                from .fit import nightly
                summary = await asyncio.to_thread(nightly, directory, now=now_utc())
                logger.info("instinct nightly pass: %d refits, %d demotions", len(summary.get("refits") or []),
                            len(summary.get("demoted") or []))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("instinct maintenance failed (%s)", type(error).__name__)
        await sleep(DAY_S)


async def _prune_usage(directory: Path) -> None:
    """The model usage ledger's retention, apart from the decision log's: one failing never skips the other."""
    try:
        dropped = await asyncio.to_thread(usage.prune, directory)
        if dropped:
            logger.info("model usage: %d rows past retention dropped", dropped)
    except Exception as error:
        logger.warning("model usage retention failed (%s)", type(error).__name__)


__all__ = ["EFFECTS", "FAST_POINTS", "MAX_STATE_CHARS", "REGISTRY", "Decision", "Hashed", "Instinct", "Point",
           "RegistrationError", "RulesAnswer", "Tier", "aclose", "answer_probabilities", "calibrate",
           "calibrate_yes", "maintenance", "register", "shared"]
