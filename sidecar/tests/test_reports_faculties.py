"""``colonymind report faculties``: what each faculty did over a window, read from a live instance, read-only."""
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from colonymind.reports import bench, faculties
from colonymind.reports.registry import FACULTIES
from report_fixtures import write_bench_dir, write_instance

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
# What a reader must never see outside the one sentence that names a faculty's setting: ids, codes, store field
# names, intention types, arm names and em dashes.
JARGON = re.compile(r"owner-1|contact-\d|K7F|W4W|WSH|[0-9a-f]{8}-[0-9a-f]{4}|cost_tokens|outreach_|commitment_|"
                    r"check_in|mind:|lesson_use|drift_check|wishes_named|typed_call|all-on|_pp|\u2014|\bNone\b")


@pytest.fixture
def instance(tmp_path):
    return write_instance(tmp_path / "instance", NOW)


def report(home, *, since="7d", bench_dir=None):
    return faculties.document(faculties.collect(home, since=since, now=NOW,
                                                bench=bench.load(bench_dir) if bench_dir else None))


def section(document, title):
    """The text of one faculty's section: its heading to the next heading."""
    lines, inside = [], False
    for block in document.blocks:
        if block.kind == "heading" and block.level == 2:
            inside = block.text.split(":")[0] == title
            continue
        if inside:
            lines.extend(block.items if block.kind == "list" else [block.text])
    return "\n".join(lines)


def readable(document):
    """The report's text without the sentences that name a setting (those carry the setting's name)."""
    return "\n".join(line for line in document.text().splitlines() if not line.startswith("To switch it "))


def test_every_faculty_has_a_section_that_says_whether_it_is_on_and_names_its_setting(instance):
    document = report(instance)
    headings = [block.text for block in document.blocks if block.kind == "heading" and block.level == 2]
    assert headings == [f"{faculty.title}: {'off' if faculty.key in {'self_narrative', 'affect_rules'} else 'on'}"
                        for faculty in FACULTIES]
    assert "To switch it off, set mind.faculties.skills to false in colonymind.yaml." in section(document, "Skills")
    assert "To switch it off, set context.status to false in colonymind.yaml." in section(document, "Status block")
    assert "Context fitting and its caps are mandatory." in section(document, "Memory blocks")
    assert "context.blocks" not in section(document, "Memory blocks")
    assert "To switch it on, set mind.faculties.self_narrative to true in colonymind.yaml." in section(
        document, "Self-narrative")
    assert "turn the mind off" in section(document, "Initiative")
    assert not JARGON.search(readable(document)), JARGON.search(readable(document))


def test_the_report_opens_with_three_lines_and_the_window(instance, tmp_path):
    document = report(instance, bench_dir=write_bench_dir(tmp_path / "results"))
    summary = document.blocks[0].items
    assert [line.split(":")[0] for line in summary] == ["What helped", "Unproven", "What looked harmful"]
    assert summary[0].startswith("What helped: skills")
    assert summary[2].startswith("What looked harmful: the self-narrative")
    assert "The mind is on at the standard level." in document.text()
    assert "This covers the 7 days to 2026-09-20 12:00 UTC." in document.text()
    assert "27 faculties are on and 2 are off: the self-narrative and the feeling rules." in document.text()


def test_initiative_says_what_the_mind_did_and_whom_it_reached(instance):
    text = section(report(instance), "Initiative")
    assert "It formed 23 intentions of its own: 9 tasks, 12 messages and 2 goals." in text
    assert "It asked the owner's approval 7 times: 2 granted, 1 declined, 1 expired unanswered and 3 still waiting." \
        in text
    assert "Of its 9 tasks, 5 finished, 1 failed, 1 was stopped before it ran, 1 is with the body with no outcome " \
           "yet and 1 waits for the owner." in text
    assert "It sent 10 messages: 7 to the owner and 3 to 2 contacts." in text
    assert "The owner rated 2 of its messages: 1 welcome and 1 not wanted." in text
    assert "Cost: 18 model calls and 16,530 tokens of the mind's own thinking" in text
    assert "Tasks took a median of 2 hours from forming to an outcome." in text
    assert "The owner answered an ask after a median of 3 hours." in text
    assert "1 task was handed to the body 3 days ago and has no outcome yet." in text
    assert "1 message could not be sent." in text
    assert "The breaker tripped once" in text


def _intention(home, kind, type, decision, at, *transitions, **updates):
    """One more intention written through the real store: ``transitions`` are ``(status, action, at)``."""
    from colonymind.initiatives.store import InitiativeStore
    store = InitiativeStore(state_dir=home)
    try:
        row, _ = store.create_intention(kind=kind, type=type, title=f"{type} work", drive="duty", cls="internal",
                                        decision=decision, decision_reason="fixture", status="proposed",
                                        dedup_key=None, created_at=at)
        for status, action, when in transitions:
            store.transition(row.id, status, action=action, at=when)
        if updates:
            store.update(row.id, **updates)
        return row.id
    finally:
        store._db.close()


def test_a_task_is_counted_once_in_its_outcomes(instance):
    """A blocked task keeps its open status (the body may still settle it): it is blocked, not also with the
    body with no outcome yet, and the parts add up to the tasks."""
    _intention(instance, "task", "upkeep_task", "act", NOW - timedelta(hours=3),
               ("dispatched", "bound", NOW - timedelta(hours=3)), outcome="blocked", result="held for the owner")
    text = section(report(instance), "Initiative")
    line = next(line for line in text.splitlines() if line.startswith("Of its "))
    total = int(re.match(r"Of its (\d+) tasks", line).group(1))
    parts = [int(value) for value in re.findall(r"(?:^|, | and )(\d+) ", line.split(", ", 1)[1])]
    assert sum(parts) == total, line
    assert "1 blocked" in line and "1 is with the body with no outcome yet" in line


def test_a_stall_is_a_task_with_the_body_for_over_a_day_since_it_was_handed_over(instance):
    """A goal is not handed to the body, a task approved a minute ago after a two-day ask has been with the body
    for a minute, and a blocked task has an outcome (it waits for the owner): none of them is a stall. The
    fixture's one task dispatched three days ago still is."""
    _intention(instance, "goal", "goal", "act", NOW - timedelta(days=3), ("approved", "queued", NOW - timedelta(days=3)))
    _intention(instance, "task", "research", "ask", NOW - timedelta(days=2),
               ("asked", "asked", NOW - timedelta(days=2)), ("approved", "queued", NOW - timedelta(minutes=1)))
    _intention(instance, "task", "upkeep_task", "act", NOW - timedelta(days=4),
               ("dispatched", "bound", NOW - timedelta(days=4)), outcome="blocked", result="held for the owner")
    text = section(report(instance), "Initiative")
    assert "1 task was handed to the body 3 days ago and has no outcome yet." in text
    assert "tasks were handed to the body" not in text


def test_the_gate_counts_messages_by_their_last_verdict_and_never_calls_one_sent(instance):
    """The gate lets a message go or holds it; sending is the sending path's record. A message it made to wait
    and later let through is one message with two reviews."""
    conn = sqlite3.connect(instance / "mind.db")
    for ident, decision, minutes in (("r1", "wait", 0), ("r1", "send", 20)):
        at = NOW - timedelta(days=1) + timedelta(minutes=minutes)
        conn.execute("INSERT INTO owner_gate (id, at, ident, decision, stage, until) VALUES (?,?,?,?,?,?)",
                     (f"{ident}-{minutes}", at.isoformat(), ident, decision, "reachability" if minutes == 0
                      else "decision", (at + timedelta(minutes=20)).isoformat() if decision == "wait" else None))
    conn.commit()
    conn.close()
    gate = section(report(instance), "Owner gate")
    assert ("It reviewed 7 messages to the owner in 8 reviews: 4 let through, 1 made to wait, 1 held for the "
            "digest and 1 dropped; 1 of them waited first." in gate), gate
    assert " sent" not in gate.split("To switch")[0].replace("unprompted", "")


@pytest.mark.parametrize("text, level, said", [
    ((NOW - timedelta(days=2)).isoformat(), 1.0, None),
    ("indefinite", 1.0, "The owner has paused it; it stays quiet until turned on again."),
    ((NOW + timedelta(hours=5)).isoformat(), 1.0, "The owner has paused it until 2026-09-20 17:00 UTC."),
    ("indefinite", 0.0, None),
])
def test_a_pause_is_reported_only_while_it_holds(instance, text, level, said):
    from colonymind.mind.concerns import MindState, open_mind_db
    conn = open_mind_db(instance / "mind.db")
    MindState(conn, clock=lambda: NOW).set("outreach.pause", level=level, text=text, now=NOW - timedelta(days=3))
    conn.commit()
    conn.close()
    outreach = section(report(instance), "Owner outreach")
    if said is None:
        assert "paused" not in outreach
    else:
        assert said in outreach


def test_the_broadcast_says_what_reaches_the_owners_turns_as_the_mind_picks_it(instance):
    """The broadcast carries the open concerns with thought left (an exhausted one is not), and none with the
    faculty off; concerns the drives raise are not the broadcast's own activity, so they never make an off
    broadcast look active."""
    config = (instance / "colonymind.yaml").read_text()
    (instance / "colonymind.yaml").write_text(config.replace("  faculties:\n", "  faculties:\n    broadcast: false\n"))
    broadcast = section(report(instance), "Broadcast")
    assert "With the broadcast off, none is carried into the owner's turns." in broadcast
    assert "It is off, yet" not in broadcast and "top 3" not in broadcast
    (instance / "colonymind.yaml").write_text(config)
    conn = sqlite3.connect(instance / "mind.db")
    conn.execute("UPDATE concerns SET thoughts_spent = max_thoughts WHERE id IN ('c0', 'c1')")
    conn.execute("UPDATE concerns SET drive = 'duty', kind = 'commitment' WHERE id = 'c3'")
    conn.commit()
    conn.close()
    broadcast = section(report(instance), "Broadcast")
    assert ("The top 3 with thought left widen recall on the owner's turns; 1 of them is shown in the Mind section "
            "(idle interests stay out of it).") in broadcast
    conn = sqlite3.connect(instance / "mind.db")
    conn.execute("UPDATE concerns SET thoughts_spent = max_thoughts")
    conn.commit()
    conn.close()
    assert "None has thought left, so none is carried into the owner's turns." in section(report(instance),
                                                                                          "Broadcast")


def test_the_drives_report_the_weights_the_mind_uses(instance):
    """With the drives faculty off every drive weighs the same whatever the file sets, so a zero weight there
    switches nothing off; intentions form either way, so they never make an off faculty look active."""
    config = (instance / "colonymind.yaml").read_text()
    (instance / "colonymind.yaml").write_text(config.replace(
        "  faculties:\n", "  drives:\n    social: 0\n  faculties:\n    drives: false\n"))
    drives = section(report(instance), "Drives")
    assert "With the drives off every drive weighs the same; the weights set in colonymind.yaml are not used." \
        in drives
    assert "set to zero" not in drives and "It is off, yet" not in drives
    (instance / "colonymind.yaml").write_text(config.replace("  faculties:\n", "  drives:\n    social: 0\n  faculties:\n"))
    drives = section(report(instance), "Drives")
    assert "The drive weights are duty 1, social 0, curiosity 0.5, mastery 1 and upkeep 1." in drives
    assert "The social drive is set to zero, so its work is off." in drives


def test_each_faculty_reads_its_own_record(instance):
    document = report(instance)
    assert "It thought about open-ended concerns with a model call 5 times: 3 became work and 2 formed nothing." \
        in section(document, "Deliberation")
    assert "It adopted 2 goals: 1 was achieved and 1 ran out of time." in section(document, "Goals")
    assert ("5 concerns are open now. The top 3 with thought left widen recall on the owner's turns; none of them "
            "is shown in the Mind section (idle interests stay out of it).") in section(document, "Broadcast")
    people = section(document, "People")
    assert "It sent 3 messages to 2 contacts: 2 check-ins and 1 message the owner asked it to pass on." in people
    assert "1 contact replied to a check-in and 1 did not." in people
    outreach = section(document, "Owner outreach")
    assert ("It sent the owner 4 messages of its own: 1 finding, 1 open loop, 1 message of care and 1 answer the "
            "owner asked for.") in outreach
    assert "The owner found 1 welcome and 1 not wanted; 1 went unanswered." in outreach
    gate = section(document, "Owner gate")
    assert ("It reviewed 6 messages to the owner: 3 let through, 1 made to wait, 1 held for the digest and 1 "
            "dropped." in gate)
    assert "It asked for a second opinion once." in gate
    assert "no embedding endpoint is set" in section(document, "Semantic recall")
    consolidation = section(document, "Nightly consolidation")
    assert "It ran 3 nights: 2 finished and 1 did not finish." in consolidation
    assert "It asked the owner 2 questions about statements that disagree: 1 answered and 1 expired." in consolidation
    assert "1 night hit an error." in consolidation
    assert "It formed 2 opinions, revised 1 and withdrew 1 (the owner asked for 1 of those)." in section(
        document, "Opinions")
    lessons = section(document, "Lessons")
    assert "It learned 2 new lessons, made 1 active and retired 1." in lessons
    assert "It keeps 1 active lesson and 1 candidate." in lessons
    assert "Lessons were served in 3 of the owner's turns." in lessons
    assert "It turned away 3 restated statements as nothing new" in section(document, "Learning rules")
    assert "It keeps 1 skill; skills were loaded 3 times in all." in section(document, "Skills")
    assert "It made 2 wishes: 1 granted and 1 waiting for the owner." in section(document, "Wishes")
    assert ("1 wish came with the model's own words for the capability: 1 shown to the owner beside the typed "
            "capability." in section(document, "Wishes in its own words"))
    assert "Right now it is quite curious and somewhat uneasy, and frustrated about 1 topic." in section(
        document, "Feelings")
    identity = section(document, "Identity check")
    assert "It ran 2 identity checks: 1 steady and 1 that found something had moved." in identity
    assert "The latest check found something had moved." in identity


def _heading(document, title):
    return next(block.text for block in document.blocks if block.kind == "heading" and block.text.startswith(title))


# The local embedder as init records it (router.embed_provider: local): no endpoint, the pinned model's commit and width.
LOCAL_EMBEDDER = ("router:\n  embed_provider: local\n  embed_model: Qwen/Qwen3-Embedding-0.6B\n"
                  "  embed_revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3\n  embed_dims: 1024\n")


def test_the_local_embedder_counts_as_an_embedder_for_every_faculty_that_reads_meaning(instance):
    """``router.embed_provider: local`` embeds with no endpoint: the report says semantic recall runs on the local
    model, and neither it nor the two faculties that read meaning say recall matches words only."""
    config = instance / "colonymind.yaml"
    config.write_text(config.read_text() + LOCAL_EMBEDDER)
    document = report(instance)
    recall = section(document, "Semantic recall")
    assert "no embedding endpoint is set" not in recall and "so recall matches words only" not in recall
    assert ("Recall by meaning runs on the local embedder, Qwen/Qwen3-Embedding-0.6B, which the sidecar loads "
            "itself.") in recall
    for title in ("Contradictions by meaning", "Earlier statements by meaning"):
        assert "no embedding endpoint is set" not in section(document, title), title
    assert not JARGON.search(recall)


def test_each_faculty_is_on_or_off_as_the_running_code_has_it(instance):
    """A switch is not the whole story: with the mind off, semantic recall, the claim-side learning rules, the
    memory blocks and the status block still run; goals need the drives, skills need lessons, wishes in the
    model's own words need wishes, and the feeling rules stand in for the running feelings."""
    config = (instance / "colonymind.yaml").read_text()
    (instance / "colonymind.yaml").write_text(config.replace("  enabled: true", "  enabled: false"))
    document = report(instance)
    assert "The mind is switched off, so only what runs without it acts (each section says)." in document.text()
    assert _heading(document, "Semantic recall") == "Semantic recall: on"
    assert _heading(document, "Learning rules") == "Learning rules: on"
    assert "only the rules for remembered statements apply" in section(document, "Learning rules")
    assert _heading(document, "Status block") == "Status block: on"
    assert _heading(document, "Memory blocks") == "Memory blocks: on"
    assert _heading(document, "Goals") == "Goals: off"
    assert "the mind is switched off" in section(document, "Goals")
    assert "To switch it on, turn the mind on" in section(document, "Goals")
    (instance / "colonymind.yaml").write_text(config.replace(
        "  faculties:\n", "  faculties:\n    drives: false\n    lessons: false\n    affect_rules: true\n").replace(
        "    affect_rules: false\n", "").replace("    wishes: true\n", "    wishes: false\n"))
    document = report(instance)
    for title, reason in (("Goals", "need the drives"), ("Skills", "lessons are off"),
                          ("Wishes in its own words", "wishes are off"), ("Feelings", "feeling rules stand in")):
        assert _heading(document, title) == f"{title}: off", title
        assert reason in section(document, title), title
        assert "to true in colonymind.yaml and turn" not in section(document, title)
    assert "To switch it on, set mind.faculties.drives to true in colonymind.yaml." in section(document, "Goals")
    assert "set mind.faculties.affect_rules to false" in section(document, "Feelings")
    assert _heading(document, "Feeling rules") == "Feeling rules: on"


def test_the_benchmark_evidence_sits_beside_each_faculty(instance, tmp_path):
    document = report(instance, bench_dir=write_bench_dir(tmp_path / "results"))
    assert "Skills measurably helped here: keep it on." in section(document, "Skills")
    assert "points to harm" in section(document, "Self-narrative")
    assert "Benchmark: not measured yet." in section(document, "Goals")
    assert "Benchmark: not measured yet." in section(document, "Wishes")


def test_a_shorter_window_counts_only_what_happened_in_it(instance):
    document = report(instance, since="12h")
    assert "This covers the 12 hours to 2026-09-20 12:00 UTC." in document.text()
    assert "It sent the owner nothing of its own in this window." in section(document, "Owner outreach")
    with pytest.raises(ValueError, match="since"):
        faculties.window("soon", now=NOW)
    assert faculties.window("2026-09-01", now=NOW)[0] == datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_a_record_that_cannot_be_read_is_said_and_the_rest_is_reported(instance, monkeypatch):
    def broken(reading, section):
        raise KeyError("an unexpected shape")
    monkeypatch.setitem(faculties.READERS, "opinions", broken)
    document = report(instance)
    assert "Its record could not be read here (KeyError)." in section(document, "Opinions")
    assert "It adopted 2 goals" in section(document, "Goals")


def test_a_row_in_a_shape_the_report_cannot_parse_is_left_out_and_counted(instance):
    conn = sqlite3.connect(instance / "initiatives.db")
    conn.execute("UPDATE initiatives SET result_metadata = '{not json' WHERE type = 'goal_step'")
    conn.execute("UPDATE initiatives SET created_at = 'yesterday-ish' WHERE type = 'upkeep_task'")
    conn.commit()
    conn.close()
    text = report(instance).text()
    assert "3 records of the mind's work could not be read and are left out." in text
    assert "It adopted 2 goals" in text


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads any file")
def test_a_store_the_report_may_not_open_is_reported_unreadable(instance):
    (instance / "initiatives.db").chmod(0)
    try:
        document = report(instance)
    finally:
        (instance / "initiatives.db").chmod(0o600)
    assert "1 store could not be read, so what it holds is missing below." in document.text()
    assert "5 concerns are open now" in section(document, "Broadcast")


def test_an_instance_with_no_records_is_reported_as_silent(tmp_path):
    home = tmp_path / "fresh"
    home.mkdir()
    (home / "colonymind.yaml").write_text("owner:\n  contact_id: owner-1\n")
    document = report(home)
    assert "It is on but formed nothing in this window." in section(document, "Initiative")
    assert "No record of the mind's work was found in this instance." in document.text()
    assert not any(path.name != "colonymind.yaml" for path in home.iterdir())


def _fingerprint(home: Path):
    """Every file's bytes and modification time. A WAL database's shared-memory index (``-shm``) is left out: every
    reader of a live WAL database records its read position there, which is SQLite's locking, not the data."""
    return {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in sorted(home.rglob("*")) if path.is_file() and not path.name.endswith("-shm")}


def test_the_report_only_reads_even_beside_a_writer(instance):
    before = _fingerprint(instance)
    writer = sqlite3.connect(instance / "initiatives.db")
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE initiatives SET priority = priority WHERE 0")
    try:
        done = threading.Event()
        result = {}

        def run():
            result["text"] = report(instance).text()
            done.set()
        thread = threading.Thread(target=run)
        thread.start()
        assert done.wait(30), "the report waited on the writer"
    finally:
        writer.rollback()
        writer.close()
    assert "It formed 23 intentions" in result["text"]
    # The writer made initiatives.db's WAL; nothing else changed or appeared.
    after = {name: value for name, value in _fingerprint(instance).items() if name != "initiatives.db-wal"}
    assert after == before


def test_a_database_nobody_has_open_is_read_without_creating_files(instance):
    """A WAL database with no writer (no -wal or -shm beside it) is read as an immutable file: a read-only
    connection would otherwise create both."""
    assert not (instance / "mind.db-shm").exists()
    before = _fingerprint(instance)
    report(instance)
    assert _fingerprint(instance) == before
    assert not (instance / "mind.db-shm").exists() and not (instance / "initiatives.db-wal").exists()


def test_a_store_a_writer_changes_after_the_report_opened_it_is_read_afresh(tmp_path):
    """A WAL database nobody had open is read as an immutable file, which SQLite never checks for change. When a
    writer opens it, writes and checkpoints while the report still holds it, the report must not read a mix of
    the old pages it kept and the new ones on disk: it reads the store again."""
    from colonymind.reports.instance import Instance
    home = tmp_path / "home"
    home.mkdir()
    path = home / "mind.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
    conn.executemany("INSERT INTO notes (body) VALUES (?)", [("a" * 500,)] * 2000)
    conn.commit()
    conn.close()
    assert not path.with_name("mind.db-shm").exists()
    instance = Instance(home)
    try:
        assert instance.query("mind.db", "SELECT length(body) AS n FROM notes WHERE id = 1")[0]["n"] == 500
        writer = sqlite3.connect(path)
        writer.execute("DELETE FROM notes")
        writer.commit()
        writer.execute("VACUUM")
        writer.executemany("INSERT INTO notes (body) VALUES (?)", [("b" * 700,)] * 2500)
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.close()
        rows = instance.query("mind.db", "SELECT count(*) AS n, sum(length(body)) AS size FROM notes")
        assert (rows[0]["n"], rows[0]["size"]) == (2500, 2500 * 700)
        assert instance.unreadable == []
    finally:
        instance.close()


def test_the_cli_writes_markdown_or_html(tmp_path):
    home = write_instance(tmp_path / "live", datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=1))
    results = write_bench_dir(tmp_path / "results")
    out = tmp_path / "faculties.md"
    command = [sys.executable, "-m", "colonymind", "--instance", str(home), "report", "faculties", "--since", "7d",
               "--bench", str(results), "--out", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    text = out.read_text()
    assert text.startswith("# Faculty report") and "What helped: skills" in text
    done = subprocess.run([sys.executable, "-m", "colonymind", "--instance", str(home), "report", "faculties",
                           "--format", "html"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.startswith("<!doctype html>")
    bad = subprocess.run([sys.executable, "-m", "colonymind", "--instance", str(home), "report", "faculties",
                          "--since", "whenever"], capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "since" in bad.stderr


def test_the_views_it_holds_are_the_ones_the_running_mind_lists(instance):
    """A view whose supporting claim was superseded keeps its stored status, but the running mind no longer lists it
    (its premise is unsupported); another owner's view is not this owner's either. The report counts what the
    running mind lists for the configured owner."""
    import gc
    from colonymind.self_model.judgments import SelfJudgments
    from colonymind.turns import TurnIdempotencyLedger
    from test_self_judgments import proposal, source
    ledger = TurnIdempotencyLedger(instance / "turn-idempotency.db")
    mine, theirs = SelfJudgments(ledger, owner_id="owner-1"), SelfJudgments(ledger, owner_id="contact-9")
    source(mine, "first", contact_id="owner-1")
    source(theirs, "second", contact_id="contact-9")
    premise, = mine.admitted_premises("first")
    assert mine.form(proposal([premise])).disposition == "formed"
    premise, = theirs.admitted_premises("second")
    assert theirs.form(proposal([premise], source_ref="turn:second", session_id="session-second")).disposition \
        == "formed"
    assert len(mine.revisions()) == 1
    gc.collect()
    assert "It holds 1 view now." in section(report(instance), "Opinions")
    with ledger._connect() as conn:
        conn.execute("UPDATE source_claims SET superseded_by='later' WHERE turn_id='first'")
    assert mine.revisions() == []
    gc.collect()
    assert "It holds" not in section(report(instance), "Opinions")


@pytest.mark.parametrize("skills, kept", [({}, "It keeps 0 skills"), ({"colonymind-a": "L1"}, "It keeps 1 skill;"),
                                         ({"colonymind-a": "L1", "colonymind-b": "L2", "colonymind-c": "L3"},
                                          "It keeps 3 skills"),
                                         ({"colonymind-a": "L1", "someone-elses": "L9"}, "It keeps 1 skill;")])
def test_the_skills_it_keeps_are_counted_as_the_mind_reads_its_manifest(instance, skills, kept):
    """The mind writes ``{"skills": {name: lesson id}}`` and owns only the names with its prefix: the count is of
    that mapping, never of the file's outer keys (after every skill retired, and with three, both read "1 skill")."""
    import json
    from colonymind.mind.skills import MANIFEST, Skills
    directory = instance / "skills"
    (directory / MANIFEST).write_text(json.dumps({"skills": skills}, indent=1, sort_keys=True) + "\n")
    assert len(Skills(directory, mind_state=None).owned()) == int(kept.split()[2])
    assert kept in section(report(instance), "Skills")


def test_a_window_past_what_the_instance_keeps_is_cut_to_it_and_says_so(instance):
    """Daily retention deletes the mind's finished work, its history and the owner gate's record after 90 days,
    while old open work stays: over 180 days the finished work would be missing and the open work counted, under
    a line saying the report covers 180 days. The window is cut to what the instance keeps, and says so."""
    from colonymind.initiatives.store import InitiativeStore
    store = InitiativeStore(state_dir=instance)
    store.create_intention(kind="task", type="research", title="old open work", drive="curiosity", cls="internal",
                           decision="ask", decision_reason="fixture", status="proposed", dedup_key=None,
                           created_at=NOW - timedelta(days=120))
    del store
    kept = section(report(instance, since="90d"), "Initiative")
    document = report(instance, since="180d")
    text = document.text()
    assert "This covers the 90 days to 2026-09-20 12:00 UTC" in text
    assert ("You asked for the 180 days, but the instance deletes the mind's finished work, its history and the "
            "owner gate's record after 90 days") in text
    assert section(document, "Initiative") == kept       # the old open work is outside what is covered
    assert faculties.collect(instance, since="2026-01-01", now=NOW).start == NOW - timedelta(days=90)
    # Resolved concerns go after 30 days: a longer window says its concern counts are partial.
    broadcast = section(report(instance, since="60d"), "Broadcast")
    assert "resolved concerns are deleted after 30 days" in broadcast
    assert "deleted after" not in section(report(instance, since="7d"), "Broadcast")
    assert "You asked for" not in report(instance, since="90d").text()


def test_a_learning_log_that_dropped_its_oldest_entries_says_its_counts_are_partial(instance):
    import sqlite3
    conn = sqlite3.connect(instance / "turn-idempotency.db")
    conn.execute("DELETE FROM learning_log WHERE id = (SELECT min(id) FROM learning_log)")   # as prune does
    conn.commit()
    conn.close()
    learning = section(report(instance, since="30d"), "Learning rules")
    assert "The learning log keeps only its newest entries; the oldest left is from 2026-09-18" in learning
    assert "keeps only its newest" not in section(report(instance, since="1d"), "Learning rules")


def test_an_owners_answer_kept_past_the_bound_does_not_hide_that_the_log_dropped_entries(instance):
    """``learning.prune`` keeps the owner's answers about claims memory still holds past its bound, so the oldest
    row left can be one of them, from long before the entries the log dropped: the counts are still partial."""
    conn = sqlite3.connect(instance / "turn-idempotency.db")
    conn.execute("DELETE FROM learning_log WHERE id < 3")            # as prune does: the oldest entries go
    conn.execute("INSERT INTO learning_log (id, at, store, action, stage, reason, item_id) "
                 "VALUES (1, ?, 'claim', 'ratified', 'owner_answer', 'the owner answered first', 'claim:kept')",
                 ((NOW - timedelta(days=60)).timestamp(),))
    conn.commit()
    conn.close()
    learning = section(report(instance, since="30d"), "Learning rules")
    assert "The learning log keeps only its newest entries; the oldest left is from 2026-09-18" in learning
    assert "keeps only its newest" not in section(report(instance, since="1d"), "Learning rules")
    # An answer kept past the bound with nothing dropped is not a gap.
    conn = sqlite3.connect(instance / "turn-idempotency.db")
    conn.execute("INSERT INTO learning_log (id, at, store, action, stage, reason) "
                 "VALUES (2, ?, 'claim', 'rejected', 'fixture', 'fixture')", ((NOW - timedelta(days=2)).timestamp(),))
    conn.commit()
    conn.close()
    assert "keeps only its newest" not in section(report(instance, since="30d"), "Learning rules")
