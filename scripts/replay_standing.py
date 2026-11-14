"""Replay anonymized live standing through the adapter's actual guard entry point.

This is an authority regression check. Tool arguments are deliberately absent;
it does not claim to re-evaluate the separate effect, recipient or budget rules.
Unresolved sessions are never counted as legitimate. Every residual refusal is
recorded separately by synthetic ordinal, class, tool name and evidence reason.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from colonymind_hermes.capture import SessionMap
from colonymind_hermes.guard import Guard


def replay(fixture: dict, scratch: Path) -> dict:
    settings = SimpleNamespace(outbox_path=scratch / "outbox.sqlite3", key_file=scratch / "api.key",
                               worker_profile="worker", mind=lambda: {}, owner_handles=lambda: {},
                               owner_contact_id=lambda: "", guests=lambda: {})
    # No model, tool, sidecar or live store is called by the replay.
    class NoNetwork:
        def __getattr__(self, name):
            raise AssertionError("replay attempted external access")
    sessions = SessionMap(settings, NoNetwork())
    for sid, evidence in fixture["sessions"].items():
        sessions.inherit(sid, "", standing=evidence["standing"], platform=evidence["class"])
    gate = Guard(NoNetwork(), sessions, settings)
    gate.sessions.parent = lambda sid: ""  # all ancestry was resolved on the live host
    # Explicit owner/system snapshots use the guard unchanged. Mind snapshots
    # are reported separately: without arguments this replay cannot determine
    # their existing floor/workspace verdict and must not invent one.
    totals, denied, own, own_denied = Counter(), Counter(), Counter(), Counter()
    residuals, mind = [], Counter()
    logging.getLogger("colonymind_hermes.guard").addHandler(logging.NullHandler())
    logging.getLogger("colonymind_hermes.guard").propagate = False
    for ordinal, call in enumerate(fixture["calls"], 1):
        evidence = fixture["sessions"][call["session"]]
        cls, standing = evidence["class"], evidence["standing"]
        totals[cls] += 1
        if standing == "mind":
            mind[cls] += 1
            continue
        result = gate.pre_tool_call(call["tool"], {}, call["session"])
        refused = bool(result and result.get("action") == "block")
        if standing in {"owner", "system"}:
            own[cls] += 1
            if refused:
                own_denied[cls] += 1
        if refused:
            denied[cls] += 1
            residuals.append({"ordinal": ordinal, "class": cls, "tool": call["tool"],
                              "reason": evidence["reason"], "standing": standing})
    return {"calls": dict(totals), "established_own": dict(own), "own_would_deny": dict(own_denied),
            "denied": dict(denied), "mind_requires_effect_arguments": dict(mind), "residuals": residuals}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="standing-replay-", dir=os.environ["TMPDIR"]) as directory:
        report = replay(json.loads(args.fixture.read_text()), Path(directory))
    print(json.dumps(report))
