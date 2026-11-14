"""Resident blocks through stock Hermes (Compass 5.6 and 15.9): a memory context that outgrows the host's
inline cap never makes every block vanish.

Stock Hermes replaces a memory prefetch longer than ``hooks.output_spill.max_chars`` (10,000 by default)
with a 500-character head, a 500-character tail and a file path. The provider asks the sidecar for a
context within that cap (``inline_chars``, less what it adds itself) and tells it which sections it will
send as one line (``shown``); against a sidecar that names no budget of its own (1.10) it holds whole
sections back itself, fresh state last, and names every one it held back.
"""

import hashlib
import re

import pytest

from conftest import CANARY, probe
from test_prefetch import PROVIDER_PRELUDE

STATE = '{"facts": ["the nightly backup ran at 02:00", "the build is green"], "stale": []}'
TITLES = {"temporal-context": "Current Time", "colonymind-memory": "Relevant Memories",
          "colonymind-owner-preferences": "How they want me to communicate", "colonymind-stances": "Your recorded views",
          "colonymind-lessons": "What you learned", "colonymind-current-state": "Current observed state and stale observations",
          "colonymind-goals": "Active Goals"}
PRIORITY = {"temporal-context": 100, "colonymind-memory": 90, "colonymind-owner-preferences": 88, "colonymind-stances": 87,
            "colonymind-lessons": 86, "colonymind-current-state": 76, "colonymind-goals": 80}


def growing(body):
    """A 1.10 sidecar's context on night N: every persisted block has gained a line each night (the Compass
    persist hook), the current state is as small as ever."""
    night = int(re.search(r"night (\d+)", body["incoming_message"]["content"]).group(1))

    def grown(prefix):
        return "\n".join(f"{prefix} {index:03d} " + "y" * 90 for index in range(night + 1))
    bodies = {"temporal-context": "Sunday, 10:00 AM UTC", "colonymind-memory": "\n".join(["memory " + "m" * 93] * 30),
              "colonymind-current-state": STATE}
    return {"sections": [{"id": key, "title": title, "priority": PRIORITY[key],
                          "body": bodies.get(key) or grown(key.rsplit("-", 1)[-1])} for key, title in TITLES.items()]}


def test_blocks_growing_past_the_hosts_cap_never_make_every_block_vanish(home, sidecar):
    sidecar.assemble = growing
    result = probe('''
from agent.memory_manager import MemoryManager
manager = MemoryManager()
manager.add_provider(provider)
emit(wakes=[manager.prefetch_all(f"night {night}: what is on today?", session_id=f"night-{night}") for night in range(30)])
''', home, prelude=PROVIDER_PRELUDE)
    wakes = result["wakes"]
    assert len(wakes) == 30
    for night, wake in enumerate(wakes):
        assert "output truncated" not in wake and len(wake) <= 10_000, (night, len(wake))
        assert STATE in wake and "## Current Time" in wake, night                     # fresh state, whole
        for title in TITLES.values():                                                   # every block shown or named
            assert title in wake, (night, title)
    assert "## Held back this turn" in wakes[-1] and "## Held back this turn" not in wakes[0]
    calls = sidecar.calls("/v1/host/context/assemble", "POST")
    assert all(0 < call["json"]["inline_chars"] < 10_000 and "budget_chars" not in call["json"] for call in calls)


def test_holding_back_still_ends_by_the_hosts_wait(home, sidecar):
    """Holding sections back refreshes the contact's clock for every smaller context it tries, and each refresh
    ends by the prefetch's one deadline like the first: a sidecar that spends every millisecond it is offered, a
    context over the cap and a clock refresh that hangs still reach Hermes in time, held back and with the
    provider's own clock. Driven through Hermes' own memory manager, with a shorter wait so the test stays quick."""
    sidecar.assemble = growing
    sidecar.delays["/v1/host/context/assemble"] = lambda request: request["json"].get("wait_ms", 10_000) / 1000 - 0.05
    sidecar.delays["/v1/host/context/temporal"] = 6.0
    result = probe('''
import time
import agent.memory_manager as memory_manager
memory_manager._EXTERNAL_PREFETCH_TIMEOUT_S = 3.0
manager = memory_manager.MemoryManager(external_prefetch_timeout=3.0)
manager.add_provider(provider)
set_session_vars(platform="telegram", user_id="1001", chat_id="1001", session_id="session-1")
started = time.monotonic()
context = manager.prefetch_all("night 29: what is on today?", session_id="session-1")
emit(context=context, elapsed=time.monotonic() - started)
''', home, prelude=PROVIDER_PRELUDE)
    assert "## Held back this turn" in result["context"] and STATE in result["context"], result["elapsed"]
    assert "## Current Time" in result["context"] and "Sunday, 10:00 AM UTC" not in result["context"]
    assert result["elapsed"] < 3.0


def test_the_sections_the_last_turn_carried_are_named_to_the_sidecar(home, sidecar):
    result = probe('''
from agent.memory_manager import MemoryManager
manager = MemoryManager()
manager.add_provider(provider)
manager.prefetch_all("what are my plans", session_id="session-1")
manager.sync_all("what are my plans", "Here they are.", session_id="session-1")
assert manager.flush_pending(timeout=10)
second = manager.prefetch_all("and after that?", session_id="session-1")
emit(second=second)
''', home, prelude=PROVIDER_PRELUDE)
    assert "unchanged since your last turn: Owner notes" in result["second"]
    first, second = sidecar.calls("/v1/host/context/assemble", "POST")
    assert "shown" not in first["json"]
    assert hashlib.sha256(f"Owner notes\n{CANARY}".encode()).hexdigest() in second["json"]["shown"]


@pytest.mark.requires_sidecar
def test_the_providers_fresh_state_is_the_sidecars_rank_prefix():
    """The provider cannot import the sidecar; the fresh-state ids it holds back last must be the same ones
    the sidecar admits first."""
    from colonymind.memory import blocks
    from colonymind_memory.provider import _FRESH_FIRST
    assert blocks.RANK[:len(_FRESH_FIRST)] == _FRESH_FIRST
    from colonymind_memory.provider import UNCHANGED_LINE
    assert blocks.UNCHANGED_LINE == UNCHANGED_LINE


def test_a_retired_fitting_report_cannot_disable_the_providers_spill_guard(home, sidecar):
    """Even a response declaring blocks disabled must fit the host's cap and keep fresh state."""
    recall = "\n".join(f"memory {index:03d} " + "m" * 90 for index in range(120))

    def unfitted(body):
        return {"budget": {"blocks": False}, "sections": [
            {"id": "temporal-context", "title": "Current Time", "priority": 100, "body": "Sunday, 10:00 AM UTC"},
            {"id": "colonymind-memory", "title": "Relevant Memories", "priority": 90, "body": recall},
            {"id": "colonymind-current-state", "title": TITLES["colonymind-current-state"], "priority": 76, "body": STATE}]}
    sidecar.assemble = unfitted
    result = probe('''
sections = provider._prefetch_sections("status?", session_id="session-1", contact_id=provider._contact_id)
context = provider.prefetch("status?", session_id="session-1")
emit(context=context, base=provider._with_fresh_temporal_sync(provider._format_sections(sections), contact_id=provider._contact_id))
''', home, prelude=PROVIDER_PRELUDE)
    assert len(result["base"]) > 10_000
    assert len(result["context"]) <= 10_000 and "output truncated" not in result["context"]
    assert recall not in result["context"] and "## Held back this turn" in result["context"]
    assert "Relevant Memories" in result["context"] and STATE in result["context"]
    assert "## Current Time" in result["context"]
