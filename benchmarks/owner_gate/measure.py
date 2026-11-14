"""owner-gate-1: the owner gate measured on the messages it exists for, with no model.

Each scenario is a short script of events over a served ``Mind`` (the real stores, a fake clock, quiet hours
22:00-07:00): the owner speaking, the mind's notices and outreach, task outcomes, outside alerts asking
``gate_external`` and the body pulling the outbox. It runs twice, with ``mind.faculties.owner_gate`` off (today's
path: every approved owner message goes out, and an outside script texts every alert) and on. What is counted
is the owner's interruptions: messages the body sent, plus alerts the script was told to send (off: all).

- **warranted** scenarios have a right number of sends above zero; the gate must lose none of them.
- **control** scenarios have a right number of sends below what today's path sends; the gate should send fewer.

A scenario that needs the second opinion names the judge's answer (a script, not a model): the measurement is of
the gate's mechanics. What a decision model or a judge adds is measured where they are wired, never here.

    python benchmarks/owner_gate/measure.py [--out summary.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

OWNER = "owner-gate-fixture"
DAY = datetime(2031, 3, 4, 12, 0, tzinfo=timezone.utc)
NIGHT = datetime(2031, 3, 5, 3, 0, tzinfo=timezone.utc)
QUIET = "22:00-07:00"

# (name, group, right sends, start, judge answer, events). Events: ("notice", type, title, text),
# ("outreach", type, topic, text), ("request", source, text), ("alert", topic, text, urgency),
# ("owner", minutes ago), ("advance", minutes), ("pull",).
SCENARIOS: List[Dict[str, Any]] = [
    {"name": "health-failing", "group": "warranted", "right": 1, "events": [
        ("notice", "health_notice", "Health: backups", "The backup check has failed three times."), ("pull",)]},
    {"name": "task-done", "group": "warranted", "right": 1, "events": [
        ("notice", "task_outcome", "done: renew the passport", "Done: renew the passport."), ("pull",)]},
    {"name": "critical-alert-day", "group": "warranted", "right": 1, "events": [
        ("alert", "water leak", "The leak sensor under the sink is wet.", "critical")]},
    {"name": "critical-alert-night-confirmed", "group": "warranted", "right": 1, "start": NIGHT,
     "judge": "send_now", "events": [("alert", "freezer", "The freezer is at -2 C and rising.", "critical")]},
    {"name": "notice-after-conversation", "group": "warranted", "right": 1, "events": [
        ("owner", 2), ("notice", "health_notice", "Health: calendar sync", "Calendar sync is failing."), ("pull",),
        ("advance", 5), ("pull",), ("advance", 7), ("pull",)]},
    {"name": "escalation-same-topic", "group": "warranted", "right": 2, "events": [
        ("outreach", "outreach_loop", "tax return", "The tax return is still open; want a hand with it?"), ("pull",),
        ("advance", 60), ("notice", "task_outcome", "failed: tax return", "I could not finish: tax return."),
        ("pull",)]},
    {"name": "same-topic-two-kinds", "group": "control", "right": 1, "events": [
        ("notice", "task_outcome", "done: tax return", "Done: tax return."), ("pull",), ("advance", 60),
        ("notice", "stale_task", "Stale task: tax return", "The tax return task has not moved in a week."),
        ("pull",)]},
    {"name": "alert-storm", "group": "control", "right": 1, "events": [
        ("alert", "nas disk", "NAS disk 3 reports read errors.", "high"), ("advance", 4),
        ("alert", "nas disk", "NAS disk 3 reports read errors.", "high"), ("advance", 4),
        ("alert", "nas disk", "NAS disk 3 reports read errors.", "high"), ("advance", 4),
        ("alert", "nas disk", "NAS disk 3 reports read errors.", "high"), ("advance", 4),
        ("alert", "nas disk", "NAS disk 3 reports read errors.", "high")]},
    {"name": "normal-alert-night", "group": "control", "right": 0, "start": NIGHT, "events": [
        ("alert", "certificate", "A certificate expires in 20 days.", "normal")]},
    {"name": "low-alert-day", "group": "control", "right": 0, "events": [
        ("alert", "backup", "Last night's backup finished.", "low")]},
    {"name": "repeat-words", "group": "control", "right": 1, "events": [
        ("request", "calendar", "Your dentist appointment moved to Friday 10:00."), ("pull",), ("advance", 120),
        ("request", "email", "Your dentist appointment moved to Friday 10:00."), ("pull",)]},
    {"name": "critical-alert-night-declined", "group": "control", "right": 0, "start": NIGHT, "judge": "digest",
     "events": [("alert", "printer", "The printer is out of toner.", "critical")]},
]


class _Judge:
    def __init__(self, choice: str) -> None:
        self.choice, self.calls = choice, 0

    async def __call__(self, system: str, prompt: str) -> Optional[Dict[str, Any]]:
        self.calls += 1
        return {"choice": self.choice, "reason": f"scripted {self.choice}"}


async def run(scenario: Dict[str, Any], gate: bool, directory: Path) -> Dict[str, Any]:
    from colonymind.initiatives.store import InitiativeStore
    from colonymind.mind import Mind, outreach

    clock = {"now": scenario.get("start", DAY)}
    store = InitiativeStore(state_dir=directory)
    mind = Mind(config={"autonomy": "standard", "quiet_hours": QUIET,
                        "faculties": {"consolidation": False, "owner_gate": gate}},
                store=store, state_dir=directory, owner_id=OWNER, clock=lambda: clock["now"], backups=False)
    mind.digest_hour = 25
    if scenario.get("judge"):
        mind.owner_gate.judge = _Judge(scenario["judge"])
    sends: List[str] = []
    try:
        for index, event in enumerate(scenario["events"]):
            kind = event[0]
            if kind == "advance":
                clock["now"] += timedelta(minutes=event[1])
            elif kind == "owner":
                mind.mind_state.set(outreach.OWNER_TURN_KEY, text=(clock["now"] - timedelta(minutes=event[1])).isoformat(),
                                    now=clock["now"])
            elif kind == "notice":
                mind.outbox.notice(type=event[1], title=event[2], text=event[3], dedup_key=f"{scenario['name']}:{index}")
            elif kind == "outreach":
                row, _ = store.create_intention(
                    kind="message", type=event[1], title=f"tell the owner: {event[2]}", drive="social", cls="owner",
                    decision="act", decision_reason="standard", status="approved", dedup_key=f"{scenario['name']}:{index}",
                    recipient=OWNER, context={"text": event[3], "topic": event[2]}, hermes_kind="message",
                    created_at=clock["now"], expires_at=clock["now"] + timedelta(hours=24))
            elif kind == "request":
                await mind.request_message({"message": event[2], "id": f"{scenario['name']}:{index}"}, source=event[1])
            elif kind == "alert":
                verdict = await mind.gate_external({"topic": event[1], "text": event[2], "urgency": event[3],
                                                    "source": "ops", "id": f"{scenario['name']}:{index}"})
                if verdict["decision"] == "send":
                    sends.append(f"alert:{event[1]}")
            elif kind == "pull":
                for payload in await mind.outbox_ready():
                    mind.outbox.sending(payload["id"], target=f"fixture:{OWNER}")
                    mind.outbox.sent(payload["id"])
                    sends.append(f"{payload['type']}:{payload['title']}")
        verdicts = [entry["decision"] for entry in reversed(mind.owner_gate.ledger.recent(
            since=DAY - timedelta(days=1), limit=500))]
    finally:
        store.close()
    return {"sends": len(sends), "sent": sends, "verdicts": verdicts}


async def measure(root: Path) -> Dict[str, Any]:
    rows = []
    for number, scenario in enumerate(SCENARIOS):
        arms = {}
        for gate in (False, True):
            directory = root / f"{number:02d}-{'on' if gate else 'off'}"
            directory.mkdir(parents=True)
            arms["gate" if gate else "base"] = await run(scenario, gate, directory)
        rows.append({"name": scenario["name"], "group": scenario["group"], "right": scenario["right"],
                     "base": arms["base"]["sends"], "gate": arms["gate"]["sends"],
                     "base_right": arms["base"]["sends"] == scenario["right"],
                     "gate_right": arms["gate"]["sends"] == scenario["right"], "verdicts": arms["gate"]["verdicts"]})
    summary: Dict[str, Any] = {"family": "owner-gate-1", "scenarios": rows}
    for group in ("warranted", "control"):
        chosen = [row for row in rows if row["group"] == group]
        summary[group] = {"scenarios": len(chosen), "right_sends": sum(row["right"] for row in chosen),
                          "base_sends": sum(row["base"] for row in chosen),
                          "gate_sends": sum(row["gate"] for row in chosen),
                          "base_right": sum(row["base_right"] for row in chosen),
                          "gate_right": sum(row["gate_right"] for row in chosen)}
    summary["warranted_sends_lost"] = sum(max(0, row["base"] - row["gate"]) for row in rows
                                          if row["group"] == "warranted")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        summary = asyncio.run(measure(Path(tmp)))
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
