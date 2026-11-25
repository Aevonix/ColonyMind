"""ColonyMind's Hermes gates: optional plugins that ask Instinct about three Hermes seams (docs/HERMES-GATES.md).

- the cron wake gate (``cron_gate.py``): a job's pre-run script that skips a fire with nothing new for the job;
- inbound triage (``triage.py``, ``pre_gateway_dispatch``): whether a non-owner's message needs an answer, asked
  in shadow only in this release;
- the tool-result screen (``screen.py``, ``transform_tool_result``): instructions aimed at the assistant in an
  untrusted tool's result.

Registration registers a hook only for a gate its configuration (``plugins.colonymind-gates``) enables, adds no tool,
command or prompt text, and does no network I/O. The gates ask the sidecar's ``/v1/instinct/decide`` under the
adapter's own settings and credential, act only on a decision it says to apply (a live gate), and fail open.
"""

from __future__ import annotations

import logging
from typing import Any

__version__ = "1.1.0"

from . import config
from .config import GateConfigError

logger = logging.getLogger(__name__)

_TRIAGE: Any = None


def register(ctx: Any) -> None:
    global _TRIAGE
    section = config.section()
    enabled = {}
    for name, read in (("triage", config.triage), ("screen", config.screen)):
        try:
            enabled[name] = read(section)
        except GateConfigError as error:
            logger.warning("ColonyMind gates: %s; the %s gate stays off", error, name)
    wanted = [name for name, value in enabled.items() if value.enabled]
    if not wanted:
        return
    from colonymind_hermes.client import load_settings

    from .client import GateClient
    settings = load_settings()
    client = GateClient(settings)
    if "triage" in wanted:
        from colonymind_hermes.capture import SessionMap

        from .triage import Triage
        _TRIAGE = Triage(settings, client, SessionMap(settings, client.client), enabled["triage"])
        ctx.register_hook("pre_gateway_dispatch", _TRIAGE.pre_gateway_dispatch)
    if "screen" in wanted:
        from .screen import Screen
        ctx.register_hook("transform_tool_result", Screen(client).transform_tool_result)
    logger.info("ColonyMind gates %s registered: %s", __version__, ", ".join(wanted))


def flush(timeout: float = 5.0) -> bool:
    """Wait for the triage questions queued so far (tests, a clean stop); True when none is left."""
    return _TRIAGE.flush(timeout) if _TRIAGE is not None else True


__all__ = ["__version__", "flush", "register"]
