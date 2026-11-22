"""The recorder's watch: one alert to the owner per condition until it clears, the digest's daily line and the
pruning of old lines, all on the mind's tick."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from colonymind.config import DEFAULTS
from colonymind.recorder import watch
from colonymind.recorder.store import DB_FILE, INSERT, new_line, store_at
from test_mind_consolidate import fx  # noqa: F401

NOW = datetime(2030, 5, 15, 12, 0, tzinfo=timezone.utc)
HOUR, DAY = 3600, 86400


class Outbox:
    """Keeps the notices it is handed; ``broken`` makes the next ones fail."""

    def __init__(self):
        self.notices, self.broken = [], False

    def notice(self, **notice):
        if self.broken:
            raise RuntimeError("the outbox is down")
        self.notices.append(notice)


class Bench:
    def __init__(self, state_dir):
        self.store = store_at(state_dir)
        self.outbox = Outbox()
        self.mind = SimpleNamespace(state_dir=state_dir, outbox=self.outbox)
        self.now = NOW

    def put(self, kind, source, *, ago, result="done", **labels):
        """One line begun ``ago`` seconds before now."""
        line = new_line(kind, source, result=result, began_at=self.now.timestamp() - ago, **labels)
        assert self.store.write(INSERT, line) == 1
        return line[0]

    def run(self, **later):
        self.now += timedelta(**later)
        watch.upkeep(self.mind, self.now)
        return [notice["title"] for notice in self.outbox.notices]

    def alerts(self):
        return [(row["source"], row["detail"], row["result"]) for row in self.store.rows(
            "SELECT source, detail, result FROM lines WHERE kind = 'alert' ORDER BY rowid")]


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "settings", lambda: copy.deepcopy(DEFAULTS["recorder"]))
    return Bench(tmp_path)


def test_a_source_that_goes_quiet_is_told_once_until_it_clears(bench):
    for day in range(8):
        bench.put("event", "whatsapp", ago=37 * HOUR + day * DAY)        # every day, then nothing for 37 hours
        bench.put("event", "sms", ago=day * 2 * DAY + HOUR)              # every other day: not a daily source
    for day in range(3):
        bench.put("check", "check.new", ago=37 * HOUR + day * DAY)       # three days old: not yet a daily source
    assert bench.run() == ["Recorder: whatsapp has gone quiet"]
    notice = bench.outbox.notices[0]
    assert notice["type"] == "health_notice" and notice["dedup_key"].startswith("recorder:")
    assert notice["text"] == ("Nothing has been recorded from whatsapp for 37 hours. "
                              "In the week before, it recorded on 7 days.")
    assert bench.run(minutes=1) == bench.run(hours=3) == ["Recorder: whatsapp has gone quiet"]
    assert bench.alerts() == [("recorder.quiet", "whatsapp", "")]

    bench.put("event", "whatsapp", ago=0)                                # it records again: the alert clears
    assert len(bench.run(minutes=1)) == 1
    assert bench.alerts() == [("recorder.quiet", "whatsapp", "done")]
    assert len(bench.run(hours=37)) == 2                                 # quiet again: a new alert
    assert bench.outbox.notices[1]["dedup_key"] != notice["dedup_key"]
    assert len(bench.run(days=6)) == 2 and bench.alerts()[-1] == ("recorder.quiet", "whatsapp", "")
    assert len(bench.run(days=2)) == 2                                   # quiet for a week more: no longer watched
    assert bench.alerts()[-1] == ("recorder.quiet", "whatsapp", "done")


def test_a_check_that_keeps_failing_is_told_once_until_it_passes(bench):
    bench.put("check", "check.mail", ago=5 * HOUR)
    bench.put("check", "check.mail", ago=4 * HOUR, result="failed")
    bench.put("check", "check.mail", ago=3 * HOUR, result="failed")
    bench.put("check", "check.feed", ago=3 * HOUR, result="failed")
    assert bench.run() == []                                             # two failed runs are not three
    bench.put("check", "check.mail", ago=0, result="failed")
    assert bench.run(minutes=1) == ["Recorder: check.mail keeps failing"]
    assert bench.outbox.notices[0]["text"] == "check.mail has failed its last 3 runs."
    bench.put("check", "check.mail", ago=0, result="failed")
    assert len(bench.run(minutes=1)) == 1                                # still the same condition
    bench.put("check", "check.mail", ago=0)
    assert len(bench.run(minutes=1)) == 1
    assert bench.alerts() == [("recorder.failing", "check.mail", "done")]


def test_a_line_that_never_finished_is_told_once_until_it_clears(bench):
    bench.put("event", "sms", ago=31 * 60, result="")
    bench.put("event", "sms", ago=45 * 60, result="")
    bench.put("event", "whatsapp", ago=29 * 60, result="")              # still inside its 30 minutes
    bench.put("card", "mind.task", ago=3 * DAY, result="")              # a card may stay open for 7 days
    assert bench.run() == ["Recorder: sms has unfinished event lines"]
    assert bench.outbox.notices[0]["text"] == ("2 events from sms began more than 30 minutes ago "
                                               "and never got a result.")
    assert bench.run(minutes=2) == ["Recorder: sms has unfinished event lines",
                                    "Recorder: whatsapp has unfinished event lines"]
    assert len(bench.run(hours=2)) == 2
    assert len(bench.run(days=2)) == 2                                   # no new lost line for a day: cleared
    assert {alert[2] for alert in bench.alerts()} == {"done"}
    bench.put("event", "sms", ago=31 * 60, result="")
    assert len(bench.run()) == 3                                         # the next lost line is a new alert


def test_a_notice_that_fails_is_tried_again(bench):
    bench.put("event", "sms", ago=31 * 60, result="")
    bench.outbox.broken = True
    assert bench.run() == []
    assert bench.alerts() == [("recorder.unfinished", "event:sms", "failed")]
    bench.outbox.broken = False
    assert bench.run(minutes=1) == ["Recorder: sms has unfinished event lines"]


def test_nothing_happens_without_a_store(bench, tmp_path):
    assert bench.run() == [] and not (tmp_path / DB_FILE).exists()
    assert watch.digest_lines(bench.outbox, NOW - timedelta(days=1), NOW) == []


def test_old_lines_are_dropped_once_a_day(bench):
    old = bench.put("event", "sms", ago=181 * DAY)
    kept = bench.put("event", "sms", ago=170 * DAY)
    bench.run()
    assert [row["id"] for row in bench.store.rows("SELECT id FROM lines")] == [kept]
    late = bench.put("event", "sms", ago=200 * DAY)
    bench.run(minutes=5)                                                 # the same day: nothing is dropped
    assert {row["id"] for row in bench.store.rows("SELECT id FROM lines")} == {kept, late}
    bench.run(days=1)
    assert old != late and [row["id"] for row in bench.store.rows("SELECT id FROM lines")] == [kept]


def test_the_digest_line_counts_kinds_choices_results_failures_and_open_lines(bench):
    since = NOW - timedelta(days=1)
    for choice, result, count in (("ignore", "silent", 3), ("reply", "sent", 2), ("react", "reacted", 1),
                                  ("reply", "failed", 1)):
        for _ in range(count):
            bench.put("event", "whatsapp", ago=HOUR, choice=choice, result=result)
    bench.put("event", "whatsapp", ago=600, result="")                           # in flight
    bench.put("check", "check.mail", ago=2 * HOUR)
    bench.put("check", "check.mail", ago=3 * HOUR, result="failed")
    bench.put("check", "check.mail", ago=4 * HOUR, result="failed")
    bench.put("card", "mind.task", ago=5 * HOUR, choice="ask_owner", result="")
    bench.put("card", "mind.task", ago=3 * DAY, choice="act", result="")         # before this digest, still open
    bench.put("event", "sms", ago=2 * DAY, choice="reply", result="sent")        # before this digest
    assert watch.digest_lines(bench.outbox, since, NOW) == []                    # this mind has not ticked yet
    bench.run()
    assert watch.digest_lines(bench.outbox, since, NOW) == [
        "Recorded: 8 events (3 ignore, 1 react, 3 reply; 2 sent, 1 reacted, 3 silent, 1 failed, 1 open), "
        "1 card (1 ask owner; 1 open), 3 checks (2 failed, 1 done). "
        "Failed: 3 (2 check.mail, 1 whatsapp). Still open: 1 event, 2 cards."]
    assert watch.digest_lines(bench.outbox, since - timedelta(days=30), since - timedelta(days=20)) == []


def test_settings_fall_back_to_the_defaults(monkeypatch):
    defaults = DEFAULTS["recorder"]
    assert watch.settings()["keep_days"] == defaults["keep_days"]
    found = {"quiet_hours": 12, "failing_runs": "many", "keep_days": -1, "open_minutes": {"event": 5, "card": 0}}
    monkeypatch.setattr(watch, "load_config", lambda: {"recorder": found})
    assert watch.settings() == {"quiet_hours": 12, "quiet_active_days": defaults["quiet_active_days"],
                                "failing_runs": defaults["failing_runs"], "keep_days": defaults["keep_days"],
                                "open_minutes": {"event": 5}}


async def test_the_tick_tells_the_owner_and_the_digest_carries_the_line(fx):  # noqa: F811
    store = store_at(fx.mind.state_dir)
    for hours in (3, 2, 1):
        assert store.write(INSERT, new_line("check", "check.mail", result="failed", chooser="rule",
                                            began_at=fx.now.timestamp() - hours * HOUR)) == 1
    await fx.mind.tick(force=True)
    await fx.mind.tick(force=True)
    notices = fx.messages("health_notice")
    assert [row.context["concern"] for row in notices] == ["Recorder: check.mail keeps failing"]
    assert notices[0].context["text"] == "check.mail has failed its last 3 runs."
    digest = await fx.mind.outbox.build_digest(since=fx.now - timedelta(days=1), level=fx.mind.level)
    assert ("- Recorded: 3 checks (3 failed), 1 alert (1 open). Failed: 3 (3 check.mail). "
            "Still open: 1 alert.") in digest
