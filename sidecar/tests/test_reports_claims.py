"""``colonymind report claims``: the claims formed from the messages captured over a window, in counts only, and what
became of those whose subject an earlier message of the conversation named (extraction version 19).

A claim named from an earlier message is measured live, not in shadow: the report is its line. Everything runs the
real ledger, projection and commit with the scripted extractor of the extraction tests; the report reads the store
read-only and prints no word, name or id of it.
"""
import argparse
import json
import sqlite3

import pytest

from colonymind.reports import claims as report
from colonymind.reports.cli import add_parser, run
from colonymind.reports.text import render
from test_source_claim_earlier_subjects import (CORRECT, LIVES, MOVED, Scripted, extract, fact, remove, say, sister,
                                                stored)

PLAIN = "My brother is a nurse in Leeds."


async def formed(tmp_path, **outputs):
    """The sister conversation, and a plain claim that names its own subject."""
    ledger, _ = await sister(tmp_path, **outputs)
    say(ledger, "plain", PLAIN, minute=5)
    await extract(ledger, Scripted({PLAIN: [fact(PLAIN, "My brother", "occupation", "nurse")]}))
    return ledger


@pytest.mark.asyncio
async def test_a_claim_named_from_an_earlier_message_is_counted_with_the_claims_formed(tmp_path):
    await formed(tmp_path)
    collected = report.collect(tmp_path, since="7d")
    assert collected.found and not collected.unreadable
    assert collected.claims == {"formed": 2, "by_version": {"source-claims-v19": 2}}
    assert collected.earlier == {"formed": 1, "standing": 1, "hidden": 0, "corrected": 0, "replaced": 0,
                                 "refused": {"total": 0, "by_reason": {}}}
    assert collected.sources["captured"] == 3


@pytest.mark.asyncio
async def test_a_basis_the_extraction_refused_is_counted_by_its_reason(tmp_path):
    await formed(tmp_path, **{LIVES: [fact(LIVES, "My sister", "location", "Denver", basis=(MOVED, "my sister"))]})
    collected = report.collect(tmp_path, since="7d")
    assert collected.claims["formed"] == 1
    assert collected.earlier["formed"] == 0
    assert collected.earlier["refused"] == {"total": 1, "by_reason": {"subject_basis_not_exact": 1}}


@pytest.mark.asyncio
async def test_an_annotated_naming_message_hides_the_claim_and_a_correction_retracts_it(tmp_path):
    ledger = await formed(tmp_path)
    remove(ledger, "annotate")
    assert report.collect(tmp_path, since="7d").earlier | {"refused": None} == {
        "formed": 1, "standing": 0, "hidden": 1, "corrected": 0, "replaced": 0, "refused": None}

    ledger = await formed(tmp_path / "corrected")
    say(ledger, "correct", CORRECT, minute=0, hour=20, session="later-chat")
    await extract(ledger, Scripted({CORRECT: [fact(CORRECT, "My sister", "location", "Dallas", operation="correct",
                                                   link="location")]}))
    assert stored(ledger)["lives"]["retracted_by"]
    earlier = report.collect(tmp_path / "corrected", since="7d").earlier
    # The correction inherits the basis: it is one of them, and standing; the claim it corrected is not.
    assert earlier | {"refused": None} == {"formed": 2, "standing": 1, "hidden": 0, "corrected": 1, "replaced": 0,
                                           "refused": None}




@pytest.mark.asyncio
async def test_the_window_is_when_the_message_was_captured(tmp_path):
    await formed(tmp_path)
    with sqlite3.connect(tmp_path / "turn-idempotency.db") as conn:
        conn.execute("UPDATE turn_sources SET ingested_at='2020-01-01T00:00:00.000Z' WHERE turn_id='lives'")
    collected = report.collect(tmp_path, since="7d")
    assert collected.claims["formed"] == 1 and collected.earlier["formed"] == 0
    assert report.collect(tmp_path, since="2019-12-31").earlier["formed"] == 1


def test_an_instance_with_no_ledger_reads_as_nothing_formed(tmp_path):
    collected = report.collect(tmp_path, since="24h")
    assert not collected.found
    assert collected.claims == {"formed": 0, "by_version": {}}
    assert "No claim was formed" in render(report.document(collected), "md")


def test_a_ledger_that_cannot_be_read_is_unknown_never_zero(tmp_path):
    (tmp_path / "turn-idempotency.db").write_bytes(b"not a database, " * 64)
    with pytest.raises(report.Unreadable):
        report.collect(tmp_path, since="7d")


@pytest.mark.asyncio
async def test_the_report_says_it_in_words_and_as_json_without_any_word_of_memory(tmp_path, monkeypatch, capsys):
    await formed(tmp_path)
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command"))
    assert run(parser.parse_args(["report", "claims", "--since", "7d"])) == 0
    text = capsys.readouterr().out
    assert "2 claims were formed" in text and "source-claims-v19: 2 claims" in text
    assert "1 claim took its subject from an earlier message" in text and "1 stands" in text
    assert run(parser.parse_args(["report", "claims", "--json"])) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["earlier_message_subjects"]["formed"] == 1 and data["claims"]["formed"] == 2
    assert set(data) == {"window", "found", "sources", "claims", "earlier_message_subjects"}
    printed = text + json.dumps(data)
    for words in ("sister", "Denver", "brother", "nurse", "owner", "chat", "lives", "moved"):
        assert words not in printed


@pytest.mark.asyncio
async def test_a_claim_erased_with_its_naming_message_is_gone_and_not_counted(tmp_path):
    ledger = await formed(tmp_path)
    remove(ledger, "erase")
    collected = report.collect(tmp_path, since="7d")
    assert collected.earlier["formed"] == 1 and collected.claims["formed"] == 2
