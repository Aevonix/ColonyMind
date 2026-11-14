"""Synthetic inputs for the reports: paired benchmark report files, and a live instance written through the real
stores. Every name, number and text here is made up; nothing describes a real deployment."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

from colonymind.qualification import paired
from colonymind.qualification.paired_statistics import contrast

RULE = dict(paired.RULE)


def scenarios(passes: str):
    """``'1101'`` as pass values, one per scenario."""
    return [character == "1" for character in passes]


def bench_report(label, dataset, arms, reference, outcomes, *, split="dev", unavailable=(), shipped_defaults=None):
    """One ``paired_report`` as ``colonymind models paired report --json`` writes it, with its statistics
    computed by the harness's own ``contrast``. ``outcomes``: {arm: '1011...'} per scenario, the same length
    for every arm; ``unavailable``: scenario indexes with no attributable result in the first non-reference arm;
    ``shipped_defaults``: what the report records its image shipped (``reports.arms.recorded_off``), if anything."""
    count = len(next(iter(outcomes.values())))
    manifest = hashlib.sha256(f"{label}|{dataset}|{','.join(arms)}".encode()).hexdigest()
    seed = int(manifest[:16], 16)
    profiles = {arm: {"name": arm, **paired.PROFILES[arm]} for arm in arms}
    contrasts = []
    for arm in arms:
        if arm == reference:
            continue
        entry = {"treatment": arm, "comparator": reference, "same_profile": False, "unit": "scenario"}
        missing = set(unavailable) if arm == next(a for a in arms if a != reference) else set()
        units = {f"s{index:02d}": (float(scenarios(outcomes[arm])[index]), float(scenarios(outcomes[reference])[index]))
                 for index in range(count) if index not in missing}
        entry.update(declared_units=count, unavailable_units=len(missing))
        if missing:
            entry["verdict"] = "unavailable"
        else:
            entry.update(contrast(units, seed=seed, alpha=RULE["alpha"], min_wins=RULE["min_wins"],
                                  non_inferior_pp=RULE["non_inferior_pp"]))
        contrasts.append(entry)
    arms_section = {arm: {"declared_episodes": count, "attributed_completed": sum(scenarios(outcomes[arm])),
                          "outcomes": {"pass": sum(scenarios(outcomes[arm])),
                                       "fail": count - sum(scenarios(outcomes[arm]))},
                          "accounting": {"total_model_calls": {"observed_total": 40 * count},
                                         "input_tokens": {"observed_total": 90000 * count},
                                         "output_tokens": {"observed_total": 7000 * count}}}
                    for arm in arms}
    recorded = {} if shipped_defaults is None else {"shipped_defaults": dict(shipped_defaults)}
    return {**recorded, "schema": 1, "kind": "paired_report", "report_protocol": "paired-attribution-3",
            "manifest_sha256": manifest, "label": label,
            "dataset": {"version": dataset, "split": split, "episode_ids": [f"e{index}" for index in range(count)]},
            "declared_episodes": count, "arm_order": list(arms), "reference_arm": reference, "profiles": profiles,
            "arms": arms_section, "statistics": {"protocol": "paired-statistics-1", "rule": RULE,
                                                  "bootstrap_seed": seed, "contrasts": contrasts}}


# 48 scenarios each. Skills: 12 wins, 2 losses (a demonstrated gain). The status block: 5 wins, 5 losses. The
# self-narrative: 3 wins, 9 losses (worse by more than the 10-point margin, not demonstrated).
BASE_48 = "110101101011010110101101011010110101101011010110"


def _flip(passes: str, wins: int, losses: int) -> str:
    """``passes`` with ``wins`` failures turned into passes and ``losses`` passes turned into failures."""
    values = list(passes)
    for index, value in enumerate(values):
        if wins and value == "0":
            values[index], wins = "1", wins - 1
    for index in range(len(values) - 1, -1, -1):
        if losses and passes[index] == "1" and values[index] == "1":
            values[index], losses = "0", losses - 1
    return "".join(values)


def write_bench_dir(root: Path) -> Path:
    """A benchmark results directory the way the harness leaves it: one directory per plan with its numbered
    reports, and the operator's ``<label>.report.json`` copies beside them."""
    root.mkdir(parents=True, exist_ok=True)
    all_on = BASE_48
    suite = bench_report(
        "suite-memory-1", "mind-memory-1",
        ["all-on", "all-on-skills", "all-on-status", "all-on-self_narrative", "all-on-wishes"], "all-on",
        {"all-on": all_on,
         "all-on-skills": _flip(all_on, 2, 12),        # without skills: 12 scenarios lost, 2 won
         "all-on-status": _flip(all_on, 5, 5),
         "all-on-self_narrative": _flip(all_on, 9, 3),
         "all-on-wishes": all_on},
        unavailable=())
    # all-on-wishes is the last arm: mark three of its scenarios unattributed by rebuilding its entry.
    suite["statistics"]["contrasts"][-1] = {"treatment": "all-on-wishes", "comparator": "all-on",
                                            "same_profile": False, "unit": "scenario", "declared_units": 48,
                                            "unavailable_units": 3, "verdict": "unavailable"}
    # The owner gate's pilot, full against full-plus-owner_gate, ran while the gate (and four other switches)
    # shipped off, and its report records that.
    outreach = bench_report(
        "gate-outreach-1", "mind-outreach-1", ["full", "full-plus-owner_gate"], "full",
        {"full": "1" * 30 + "0" * 18, "full-plus-owner_gate": "1" * 29 + "0" * 19}, split="heldout",
        shipped_defaults={"owner_gate": False, "learning": False, "wishes": False, "drift_check": False,
                          "status": False})
    initiative = bench_report(
        "initiative-1", "mind-initiative-1", ["base_hermes", "full"], "base_hermes",
        {"base_hermes": "1" * 20 + "0" * 36, "full": "1" * 20 + "1" * 18 + "0" * 18})
    affect = bench_report(
        "affect-1", "mind-affect-1", ["full-affect", "full", "full-affect-plus-rules"], "full-affect",
        {"full-affect": "1" * 27 + "0" * 20, "full": "1" * 27 + "0" * 20,
         "full-affect-plus-rules": "1" * 27 + "0" * 20})
    plans = [("suite-memory-1", suite), ("gate-outreach-1", outreach), ("initiative-1", initiative),
             ("affect-1", affect)]
    for index, (label, report) in enumerate(plans):
        directory = root / label
        directory.mkdir(exist_ok=True)
        path = directory / "report-001.json"
        path.write_text(json.dumps(report, indent=1))
        os.utime(path, (1_790_000_000 + index * 86400, 1_790_000_000 + index * 86400))
    # The operator's copy of the suite's report beside its directory: the same plan, read once.
    copy = root / "suite-memory-1.report.json"
    copy.write_text(json.dumps(suite))
    os.utime(copy, (1_789_000_000, 1_789_000_000))
    (root / "notes.json").write_text(json.dumps({"kind": "something_else"}))
    return root


# -- a live instance ----------------------------------------------------------------------------------------------

OWNER = "owner-1"
CONFIG = """owner:
  contact_id: owner-1
mind:
  enabled: true
  autonomy: standard
  faculties:
    self_narrative: false
    wishes_named: true
    affect_rules: false
    skills: true
    owner_gate: true
    learning: true
    wishes: true
    drift_check: true
context:
  status: true
"""


def write_instance(home: Path, now) -> Path:
    """An instance directory written through the real stores, as a week of a running mind leaves it."""
    from datetime import timedelta

    from colonymind.initiatives.store import InitiativeStore
    from colonymind.memory import learning
    from colonymind.mind import gate
    from colonymind.mind.concerns import MindState, open_mind_db
    from colonymind.mind.outcomes import Autobiography
    from colonymind.self_model import judgments
    from colonymind.turns import get_turn_idempotency_ledger

    home.mkdir(parents=True, exist_ok=True)
    (home / "colonymind.yaml").write_text(CONFIG)
    store = InitiativeStore(state_dir=home)
    ago = lambda **delta: now - timedelta(**delta)   # noqa: E731

    def row(kind, type, drive, decision, status, *, at, recipient=None, cost=0, context=None, ask_code=None,
            dedup_key=None, **updates):
        created, _ = store.create_intention(
            kind=kind, type=type, title=f"{type} work", drive=drive, cls="internal", decision=decision,
            decision_reason="fixture", status="proposed" if status != "done" else "done", dedup_key=dedup_key,
            recipient=recipient, context=context, ask_code=ask_code, created_at=at)
        if cost:
            store.update(created.id, cost_tokens=cost)
        if status != "proposed" or updates:
            store.transition(created.id, status, action="fixture", at=at, **updates)
        return created.id

    def rated(ident, verdict, by, at):
        store.transition(ident, store.get(ident).status, action="rated", details={"verdict": verdict, "by": by},
                         at=at, verdict=verdict)

    # Tasks: one done, one research the deliberation formed, one failed, one handed to the body days ago.
    row("task", "commitment_overdue", "duty", "act", "done", at=ago(days=2), outcome="done", verified="check",
        completed_at=ago(days=2) + timedelta(hours=2))
    row("task", "research", "curiosity", "act", "done", at=ago(days=3), cost=1200, outcome="done",
        verified="none", completed_at=ago(days=3) + timedelta(hours=1))
    row("task", "mastery_investigation", "mastery", "act", "failed", at=ago(days=1), cost=900, outcome="failed",
        verified="hermes_failure", failed_at=ago(days=1) + timedelta(hours=3), failed_reason="timed out")
    row("task", "upkeep_task", "upkeep", "act", "dispatched", at=ago(days=3), hermes_kind="task")
    # Asks: granted, declined, expired, waiting; and a name-only link question.
    granted = row("task", "research", "curiosity", "ask", "asked", at=ago(days=4), ask_code="K7F")
    store.transition(granted, "approved", action="queued", at=ago(days=4) + timedelta(hours=5), verdict="actioned",
                     details={"by": "owner", "code": "K7F"})
    store.transition(granted, "done", action="fixture", at=ago(days=3), outcome="done", completed_at=ago(days=3))
    row("task", "research", "curiosity", "ask", "cancelled", at=ago(days=5), ask_code="P2Q", outcome="denied",
        cancelled_at=ago(days=5) + timedelta(hours=1), cancelled_reason="denied")
    row("message", "check_in", "social", "ask", "expired", at=ago(days=6), recipient="contact-2", ask_code="Z9Z",
        outcome="expired", cancelled_at=ago(days=3), cancelled_reason="expired")
    row("task", "research", "curiosity", "ask", "asked", at=ago(hours=5), ask_code="W4W")
    row("note", "link_proposal", "upkeep", "ask", "asked", at=ago(days=1), ask_code="L1N")
    # Messages to the owner: a reminder, outreach (welcome, not wanted, unanswered, an answer), a failed send.
    row("message", "commitment_reminder", "duty", "act", "sent", at=ago(days=2), recipient=OWNER, outcome="done",
        completed_at=ago(days=2) + timedelta(minutes=1))
    welcome = row("message", "outreach_finding", "social", "act", "sent", at=ago(days=2), recipient=OWNER,
                  outcome="done", completed_at=ago(days=2) + timedelta(minutes=2))
    rated(welcome, "useful", "owner", ago(days=2) + timedelta(hours=1))
    unwanted = row("message", "outreach_care", "social", "act", "sent", at=ago(days=1), recipient=OWNER,
                   outcome="done", completed_at=ago(days=1) + timedelta(minutes=2))
    rated(unwanted, "not_useful", "owner", ago(days=1) + timedelta(hours=2))
    silent = row("message", "outreach_loop", "social", "act", "sent", at=ago(days=4), recipient=OWNER,
                 outcome="done", completed_at=ago(days=4) + timedelta(minutes=4))
    rated(silent, "ignored", "mind", ago(days=3))
    row("message", "outreach_answer", "social", "act", "sent", at=ago(hours=20), recipient=OWNER, outcome="done",
        completed_at=ago(hours=20) + timedelta(minutes=1))
    row("message", "commitment_due_soon", "duty", "act", "failed", at=ago(days=1), recipient=OWNER,
        outcome="failed", failed_at=ago(days=1), failed_reason="send failed")
    row("message", "breaker_notice", "upkeep", "act", "sent", at=ago(days=1), recipient=OWNER, outcome="done",
        completed_at=ago(days=1))
    # Messages to contacts: two check-ins (one answered), the owner's own notice to a contact.
    replied = row("message", "check_in", "social", "act", "sent", at=ago(days=3), recipient="contact-1", cost=300,
                  outcome="done", completed_at=ago(days=3) + timedelta(minutes=3))
    rated(replied, "actioned", "mind", ago(days=2))
    unanswered = row("message", "check_in", "social", "act", "sent", at=ago(days=5), recipient="contact-2",
                     cost=280, outcome="done", completed_at=ago(days=5) + timedelta(minutes=3))
    rated(unanswered, "ignored", "mind", ago(days=4))
    row("message", "commitment_notice", "duty", "act", "sent", at=ago(days=2), recipient="contact-1", outcome="done",
        completed_at=ago(days=2) + timedelta(minutes=1))
    # Goals: one satisfied through two steps, one that ran out of time.
    goal = row("goal", "goal", "curiosity", "act", "done", at=ago(days=6), cost=700, outcome="done",
               verified="check", completed_at=ago(days=2))
    for day in (5, 4):
        step = row("task", "goal_step", "curiosity", "act", "done", at=ago(days=day), outcome="done",
                   completed_at=ago(days=day) + timedelta(hours=1))
        store.update(step, parent_goal_id=goal)
    row("goal", "goal", "mastery", "act", "expired", at=ago(days=6), outcome="expired", cancelled_at=ago(days=1),
        cancelled_reason="expired")
    # Deliberation that formed nothing; a typed call answering a question.
    for day in (1, 2):
        row("note", "deliberation", "curiosity", "act", "done", at=ago(days=day), cost=800, outcome="done")
    row("note", "typed_call", "upkeep", "act", "done", at=ago(days=1), cost=150, outcome="done",
        context={"task": "mind_question_answer", "source": "question"})
    # Nights: two finished, one that did not finish.
    for day, errors in ((1, []), (2, ["episodes: TimeoutError"])):
        night = {"local_date": ago(days=day).date().isoformat(), "started_at": ago(days=day).isoformat(),
                 "calls": 4, "tokens": 5000, "done": ["lessons", "wishes", "contradictions", "digests", "episodes"],
                 "counts": {"contradictions": 1, "digests": 2, "episodes": 3, "claims_settled": 4},
                 "errors": errors}
        row("note", "consolidation", "upkeep", "act", "done", at=ago(days=day), cost=5000, outcome="done",
            completed_at=ago(days=day) + timedelta(minutes=6), context=night,
            result="4 call(s), 5000 tokens")
    row("note", "consolidation", "upkeep", "act", "done", at=ago(days=3), outcome="done",
        result="started; did not finish (it runs again at the next due tick)")
    # Questions from contradictions: one answered, one that expired unanswered.
    answered = row("message", "contradiction", "upkeep", "act", "sent", at=ago(days=2), recipient=OWNER,
                   dedup_key="reach_out:contradiction:a", outcome="done", completed_at=ago(days=2))
    store.update(answered, result_metadata={"question": {"state": "resolved", "answer": "first"}})
    lapsed = row("message", "contradiction", "upkeep", "act", "sent", at=ago(days=5), recipient=OWNER,
                 dedup_key="reach_out:contradiction:b", outcome="done", completed_at=ago(days=5))
    store.update(lapsed, result_metadata={"question": {"state": "expired"}})
    # Identity checks: one steady, one that found a drift.
    row("note", "drift_check", "upkeep", "act", "done", at=ago(days=3), cost=700, outcome="done",
        context={"verdict": "stable", "score": 0.9, "moved": []})
    row("note", "drift_check", "upkeep", "act", "done", at=ago(days=1), cost=700, outcome="done",
        context={"verdict": "drifted", "score": 0.6, "moved": ["battery 0.60 (usual 0.90)"]})
    # Wishes: one asked and granted, one that came with the model's own words, shown beside its typed capability.
    wish = row("note", "wish", "mastery", "ask", "asked", at=ago(days=4), recipient=OWNER, ask_code="WSH",
               context={"wish": {"kind": "data_source", "signature": "research:invoices", "topic": "invoices",
                                 "capability": "a data source for 'invoices'", "words": None, "words_check": None,
                                 "steps": ["research"]}})
    store.transition(wish, "done", action="fixture", at=ago(days=3), outcome="done", verified="owner",
                     result_metadata={"wish": {"state": "pursuing"}})
    row("note", "wish", "mastery", "ask", "asked", at=ago(days=1), recipient=OWNER, ask_code="WS2",
        context={"wish": {"kind": "tool", "signature": "research:calendar", "topic": "calendar",
                          "capability": "a tool for 'calendar'", "steps": ["research"], "words_check": "within",
                          "words": {"name": "a calendar", "why": "the model's own reason"}}})
    # Lessons served in the owner's turns, and a task that carried one.
    for hours in (5, 30, 60):
        row("note", "lesson_use", "mastery", "act", "done", at=ago(hours=hours), outcome="done")
    # Something formed before the window: never counted.
    row("task", "research", "curiosity", "act", "done", at=ago(days=20), outcome="done", completed_at=ago(days=19))
    store._db.close()

    # The workspace: concerns, the mind's feelings, its narrative and skills, the owner gate's record.
    conn = open_mind_db(home / "mind.db")
    state = MindState(conn, clock=lambda: now)
    for index, salience in enumerate((0.9, 0.7, 0.5, 0.3, 0.2)):
        conn.execute("INSERT INTO concerns (id, drive, kind, summary, dedup_key, salience, sources_json, "
                     "thoughts_spent, max_thoughts, status, last_touched, created_at, intention_id, detail_json) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (f"c{index}", "curiosity", "interest", f"concern {index}", f"k{index}", salience, "[]", 0, 3,
                      "open", ago(hours=index).timestamp(), ago(days=index + 1).timestamp(), None, "{}"))
    conn.execute("INSERT INTO concerns (id, drive, kind, summary, dedup_key, salience, sources_json, thoughts_spent, "
                 "max_thoughts, status, last_touched, created_at, intention_id, detail_json) "
                 "VALUES ('c9','duty','commitment','done','k9',0.1,'[]',1,3,'resolved',?,?,NULL,'{}')",
                 (ago(days=1).timestamp(), ago(days=2).timestamp()))
    conn.commit()
    state.set("affect.worry", level=0.3, now=ago(hours=2))
    state.set("affect.curiosity", level=0.5, now=ago(hours=2))
    state.set("affect.frustration:invoices", level=0.4, now=ago(hours=3))
    state.set("self.recent", text="Line one.\nLine two.", now=ago(days=1))
    state.set("skills.generation", level=2.0, now=ago(days=9))
    state.set("skills.loaded:colonymind-check-the-export", level=3.0, now=ago(days=2))
    ledger = gate.Ledger(conn)
    for index, (decision, stage, minutes, digested) in enumerate((
            ("send", "prompted", None, None), ("send", "decision", None, None), ("send", "decision", None, None),
            ("wait", "reachability", 10, None), ("digest", "dedup", None, ago(hours=10).isoformat()),
            ("drop", "dedup", None, None))):
        at = ago(days=index + 1)
        scores = {"second_opinion": {"choice": "send_now"}} if index == 2 else {}
        ledger.conn.execute(
            "INSERT INTO owner_gate (id, at, ident, ref, source, kind, type, topic, slug, fingerprint, urgency, "
            "decision, stage, reason, until, scores_json, inputs_json, digested_at, digest_ref) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"g{index}", at.isoformat(), f"i{index}", None, "mind", "notice", "outreach_finding", "topic", "s",
             "f", "normal", decision, stage, "fixture", (at + timedelta(minutes=minutes)).isoformat()
             if minutes else None, json.dumps(scores), "{}", digested, None))
    ledger.conn.commit()
    conn.close()
    skills = home / "skills" / "colonymind-check-the-export"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# A skill\n")
    (home / "skills" / ".colonymind-skills.json").write_text(
        json.dumps({"skills": {"colonymind-check-the-export": "L1"}}))     # the shape the mind writes

    # The ledger: lessons, opinions and the learning log.
    source_ledger = get_turn_idempotency_ledger(home)
    autobiography = Autobiography(source_ledger, owner_id=OWNER, clock=lambda: now)
    for ident, event, day, extra in (("L1", "admitted", 30, {"lesson": {"id": "L1", "status": "active"}}),
                                     ("L2", "admitted", 3, {"lesson": {"id": "L2", "status": "candidate"}}),
                                     ("L3", "admitted", 2, {"lesson": {"id": "L3", "status": "candidate"}}),
                                     ("L3", "activated", 1, {"lesson_id": "L3"}),
                                     ("L1", "retired", 1, {"lesson_id": "L1"})):
        autobiography.clock = lambda day=day: now - timedelta(days=day)
        autobiography.record(f"lesson:{ident}", event, f"A lesson {event}.",
                             lineage=[f"mind:lesson:{ident}:admitted"] if event != "admitted" else [],
                             scope="session", **extra)
    db = source_ledger._connect()
    try:
        judgments.initialize(db)
        learning.initialize(db)
        for index, (status, supersedes, correction, day) in enumerate((
                ("current", None, None, 2), ("current", None, None, 3), ("current", 1, None, 1),
                ("withdrawn", 2, "owner-said-so", 1))):
            db.execute("INSERT INTO self_judgment_revisions (owner_id, topic, payload_json, dependency_json, "
                       "supersedes, processor_json, created_at, status, source_turn_id, version, correction_id) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (OWNER, f"topic {index}", "{}", "[]", supersedes, "{}", (now - timedelta(days=day)).timestamp(),
                        status, f"t{index}", "1", correction))
        for store_name, action, count in (("claim", "rejected", 3), ("lesson", "witnessed", 2), ("claim", "pending", 1),
                                          ("claim", "ratified", 1)):
            for _ in range(count):
                db.execute("INSERT INTO learning_log (at, store, action, stage, reason) VALUES (?,?,?,?,?)",
                           ((now - timedelta(days=2)).timestamp(), store_name, action, "fixture", "fixture"))
        db.commit()
    finally:
        db.close()
    gc.collect()   # every connection the writers opened is closed: the files are as a stopped sidecar leaves them
    return home
