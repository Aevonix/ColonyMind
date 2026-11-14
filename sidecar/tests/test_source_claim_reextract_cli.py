"""``colonymind memory reextract [--person <id> | --all] [--dry-run] [--limit N]``, ``--status`` and ``--cancel``.

The command works on the instance's memory ledger directly: it queues the re-extraction for the sidecar's background
lane, reports progress and cancels what still waits. Before it starts anything it prints the plan: the count and the
estimate at the per-message rate measured on this instance. ``--dry-run`` opens the ledger read-only: it prints the
same plan and writes nothing.
"""
import argparse
from contextlib import closing
import json
import sqlite3
import sys

import pytest

from colonymind.beliefs import reextract_cli
from colonymind.beliefs.reextract import Reextraction
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.turns import TurnIdempotencyLedger
from test_source_claim_reextract import LAKE, OFFICE, OLD, RIVER, TEA, Scripted, drain, history

LEDGER = "turn-idempotency.db"


def parse(*argv):
    parser = argparse.ArgumentParser(prog="colonymind")
    reextract_cli.add_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(list(argv))


@pytest.fixture
def instance(tmp_path, monkeypatch):
    home = tmp_path / "instance"
    home.mkdir()
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(home))
    return home


async def stocked(home):
    ledger = TurnIdempotencyLedger(home / LEDGER)
    await history(ledger, [("t1", RIVER, 9), ("t2", TEA, 10), ("t3", LAKE, 11)], OFFICE)
    return ledger


def snapshot(path):
    with closing(sqlite3.connect(path)) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {table: sorted(map(repr, conn.execute(f'SELECT * FROM "{table}"'))) for table in tables}


async def test_the_dry_run_writes_nothing_and_prints_counts_and_the_estimate(instance, capsys):
    await stocked(instance)
    database = instance / LEDGER
    before = snapshot(database)
    raw = database.read_bytes()
    assert reextract_cli.run(parse("memory", "reextract", "--all", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert database.read_bytes() == raw and snapshot(database) == before
    assert "nothing was written" in out
    assert "1 person, 3 sources, 3 messages" in out and "3 claims" in out and OLD in out
    assert "6 model calls" in out and "not measured yet" in out

    assert reextract_cli.run(parse("memory", "reextract", "--all", "--dry-run", "--json")) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan["sources"], plan["messages"], plan["estimate"]["calls"]) == (3, 3, 6)
    assert database.read_bytes() == raw


async def test_the_estimate_uses_the_rate_measured_by_a_small_run(instance, capsys):
    ledger = await stocked(instance)
    assert reextract_cli.run(parse("memory", "reextract", "--all", "--limit", "1")) == 0
    out = capsys.readouterr().out
    assert "Queued 1 source" in out
    await drain(Reextraction(SourceClaimProjection(ledger)), Scripted(OFFICE))
    assert reextract_cli.run(parse("memory", "reextract", "--all", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "2 sources" in out and "measured on 1 message" in out
    assert "4,000 tokens" in out                              # two messages at 2,000 a message


async def test_start_prints_the_plan_first_then_queues_and_status_and_cancel_follow(instance, capsys):
    ledger = await stocked(instance)
    assert reextract_cli.run(parse("memory", "reextract", "--person", "owner", "--limit", "2")) == 0
    out = capsys.readouterr().out
    assert out.index("model calls") < out.index("Queued 2 sources")
    with closing(ledger._connect()) as conn:
        assert conn.execute("SELECT count(*) FROM source_reextract_jobs WHERE status='pending'").fetchone()[0] == 2
    assert reextract_cli.run(parse("memory", "reextract", "--status")) == 0
    assert "2 waiting" in capsys.readouterr().out
    assert reextract_cli.run(parse("memory", "reextract", "--status", "--json")) == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["jobs"] == {"pending": 2}
    assert reextract_cli.run(parse("memory", "reextract", "--cancel")) == 0
    assert "Cancelled 2" in capsys.readouterr().out
    with closing(ledger._connect()) as conn:
        assert conn.execute("SELECT count(*) FROM source_reextract_jobs WHERE status='pending'").fetchone()[0] == 0


async def test_a_plan_with_everything_queued_says_so(instance, capsys):
    await stocked(instance)
    assert reextract_cli.run(parse("memory", "reextract", "--all")) == 0
    assert "Queued 3 sources" in capsys.readouterr().out
    assert reextract_cli.run(parse("memory", "reextract", "--all", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "3 sources already queued" in out and "Nothing to re-extract" not in out


@pytest.mark.parametrize("extra", [("--dry-run",), ("--status",), ("--person", "owner"), ("--limit", "1")])
async def test_cancel_takes_no_other_option_and_writes_nothing_with_one(instance, capsys, extra):
    """``--cancel`` stops every waiting source; asked with a dry run, a status, a person or a limit it would do
    something else than was asked, so it refuses and the ledger stays as it was."""
    await stocked(instance)
    assert reextract_cli.run(parse("memory", "reextract", "--all")) == 0
    capsys.readouterr()
    database = instance / LEDGER
    before = snapshot(database)
    assert reextract_cli.run(parse("memory", "reextract", "--cancel", *extra)) == 2
    assert "--cancel" in capsys.readouterr().err
    assert snapshot(database) == before


async def test_a_person_or_all_must_be_named_and_a_missing_ledger_is_not_created(instance, capsys):
    with pytest.raises(SystemExit):
        parse("memory", "reextract", "--person", "owner", "--all")
    assert reextract_cli.run(parse("memory", "reextract", "--dry-run")) == 2
    assert "--person" in capsys.readouterr().err
    assert reextract_cli.run(parse("memory", "reextract", "--all", "--dry-run")) == 1
    assert "no memory ledger" in capsys.readouterr().err
    assert not (instance / LEDGER).exists()


def test_the_memory_group_is_shared_with_the_other_memory_commands():
    parser = argparse.ArgumentParser(prog="colonymind")
    sub = parser.add_subparsers(dest="command")
    memory = sub.add_parser("memory")
    memory.add_subparsers(dest="memory_command").add_parser("inventory")
    reextract_cli.add_parser(sub)
    assert parser.parse_args(["memory", "inventory"]).memory_command == "inventory"
    assert parser.parse_args(["memory", "reextract", "--all"]).memory_command == "reextract"


async def test_the_colonymind_command_runs_it(instance, capsys, monkeypatch):
    from colonymind import cli
    await stocked(instance)
    monkeypatch.setattr(sys, "argv", ["colonymind", "memory", "reextract", "--all", "--dry-run"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 0 and "3 sources" in capsys.readouterr().out
