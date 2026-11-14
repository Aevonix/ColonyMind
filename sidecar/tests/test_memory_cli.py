"""``colonymind memory inventory <who>`` and ``colonymind memory history <claim>``: the inventory and a claim's history
from the terminal, through the running sidecar as the instance key (the owner's credential)."""
from __future__ import annotations

import argparse
import importlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from onekey import KEY
from colonymind.config import DEFAULTS, save_config
from test_memory_inventory import id_of, make_app, office_chain, say, state  # noqa: F401 (fixture)


def memory_cli():
    return importlib.import_module("colonymind.memory_cli")


def run(argv, capsys):
    """The command as ``colonymind`` parses and runs it, its HTTP client answered by the real routes (only while it
    runs: the claims the tests seed are formed without it)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=None)
    sub = parser.add_subparsers(dest="command")
    memory_cli().add_parser(sub)
    app, real = make_app(), httpx.Client
    httpx.Client = lambda **kwargs: TestClient(app)
    try:
        code = memory_cli().run(parser.parse_args(argv))
    finally:
        httpx.Client = real
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def sidecar(state, monkeypatch):
    """The CLI's instance: its configuration and the instance key, apart from the test's state."""
    home = state / "instance"
    home.mkdir()
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    save_config({**DEFAULTS, "owner": {"contact_id": "owner-1"}}, home)
    (home / "api.key").write_text(KEY + "\n")
    return home


@pytest.mark.asyncio
async def test_the_inventory_prints_each_claim_with_its_state_quote_and_source(sidecar, state, capsys):
    river, lake, hill = await office_chain(state)
    code, out, _ = run(["memory", "inventory", "contact-a"], capsys)
    assert code == 0
    assert out.index("[current] office location: Hill") < out.index("[retracted] office location: Lake") \
        < out.index("[superseded] office location: River")
    assert '"Correction: my office is in Hill."' in out
    for claim_id, source in ((river, "o-1"), (lake, "o-2"), (hill, "o-3")):
        assert claim_id in out and f"source {source}" in out
    assert "unverified" in out and "end of the listing" in out


@pytest.mark.asyncio
async def test_the_inventory_pages_and_all_follows_every_page(sidecar, state, capsys):
    for n in range(5):
        await say(state, f"h-{n}", f"My hobby number {n} is fun.", at=f"2026-03-0{n + 1}T09:00:00+00:00")
    code, out, _ = run(["memory", "inventory", "contact-a", "--limit", "2"], capsys)
    assert code == 0 and out.count("[current] hobby") == 2
    token = out.split("--page ")[1].split()[0]
    code, more, _ = run(["memory", "inventory", "contact-a", "--limit", "2", "--page", token], capsys)
    assert code == 0 and more.count("[current] hobby") == 2
    code, every, _ = run(["memory", "inventory", "contact-a", "--limit", "2", "--all"], capsys)
    assert code == 0 and every.count("[current] hobby") == 5 and "--page" not in every
    code, raw, _ = run(["memory", "inventory", "contact-a", "--json"], capsys)
    assert code == 0 and len(json.loads(raw)["claims"]) == 5


@pytest.mark.asyncio
async def test_an_unknown_claim_is_an_error_not_a_trace(sidecar, state, capsys):
    await say(state, "a-1", "My office is in River.")
    code, out, err = run(["memory", "history", "claim:" + "0" * 64], capsys)
    assert code == 1 and "no readable claim" in err and "Traceback" not in err


def test_the_colonymind_command_registers_memory(monkeypatch):
    from colonymind import cli
    seen = []
    monkeypatch.setattr(memory_cli(), "run", lambda args: seen.append(args) or 0)
    monkeypatch.setattr("sys.argv", ["colonymind", "memory", "inventory", "contact-a"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 0
    assert seen[0].memory_command == "inventory" and seen[0].person == "contact-a"


def test_the_memory_group_holds_the_re_extraction_too(monkeypatch, capsys):
    """``memory reextract`` joins the group the inventory makes, and a bare ``colonymind memory`` names all three."""
    from colonymind import cli
    monkeypatch.setattr("sys.argv", ["colonymind", "memory"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 2
    assert "{inventory,history,reextract}" in capsys.readouterr().err


# A claim whose own label says what its value is (formed before the validator refused such a label, or carried from
# an earlier build): neither field alone looks like a credential, the two together do, and its quotation and a later
# correction hold the value without the label. A passphrase of several words is one value too. Both values are made up.
WIFI = "Hunter" + "22!" + "wifi"
PASSPHRASE = "blue heron" + " at dawn 42"
LABELLED = [
    {"claim_id": "claim:wifi", "state": "current", "subject": "lake house", "predicate": "wifi password",
     "value": WIFI, "quote": f"The wifi at the lake house is {WIFI}, on the router's sticker.",
     "verification": "unverified", "source": {"source_id": "o-1", "source_version": "c" * 64, "session_id": "s1"},
     "said_at": "2026-09-02T09:00:00+00:00", "corrections": 1},
    {"claim_id": "claim:boathouse", "state": "current", "subject": "boathouse", "predicate": "wifi password",
     "value": PASSPHRASE, "quote": f"At the boathouse the network wants {PASSPHRASE} now.",
     "verification": "unverified", "source": {"source_id": "o-2", "source_version": "d" * 64, "session_id": "s1"},
     "said_at": "2026-09-03T09:00:00+00:00", "corrections": 0},
]


class LabelledSidecar:
    """The inventory routes, canned: contact-b's two labelled claims, and each one's history with a correction that
    repeats the value."""

    def __init__(self, home=None):
        pass

    def call(self, method, path, *, params=None, json_body=None, timeout=30.0):
        for card in LABELLED:
            if path.endswith(f"/claims/{card['claim_id']}/history"):
                return {"person_id": "contact-b", "claim_id": card["claim_id"], "complete": True, "gaps": [],
                        "limit_reached": False, "steps": [
                            {"event": "stated", "claim_id": card["claim_id"], "claim": card},
                            {"event": "corrected", "claim_id": card["claim_id"], "corrects": "o-2",
                             "by": {"correction": f"Still {card['value']}, the sticker is right.",
                                    "author": "contact-b", "source": {"source_id": "o-5"}}}]}
        assert path == "/v1/host/memory/inventory", path
        return {"person_id": "contact-b", "claims": LABELLED, "limit": 50, "next_page": None, "complete": True}


@pytest.mark.parametrize("argv", [
    ["memory", "inventory", "contact-b"], ["memory", "inventory", "contact-b", "--all"],
    ["memory", "inventory", "contact-b", "--json"],
    ["memory", "history", "claim:wifi"], ["memory", "history", "claim:boathouse", "--json"],
])
def test_a_value_a_claims_own_label_calls_a_credential_never_reaches_the_terminal(argv, monkeypatch, capsys):
    """The terminal reads each claim as the console does, its predicate beside its value, so a value its own label
    calls a credential is withheld wherever the listing or the history prints it: the value, the quotation and a
    correction's words, in the text and in ``--json`` alike."""
    monkeypatch.setattr("colonymind.mind.cli.Sidecar", LabelledSidecar)
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=None)
    memory_cli().add_parser(parser.add_subparsers(dest="command"))
    assert memory_cli().run(parser.parse_args(argv)) == 0
    out = capsys.readouterr().out
    for value in (WIFI, PASSPHRASE):
        assert value not in out and json.dumps(value)[1:-1] not in out
    assert "wifi password" in out and "(a private value, withheld)" in out
    if "--json" not in argv and argv[1] == "inventory":
        assert "[current] wifi password: (a private value, withheld)  (unverified)" in out
        assert '"The wifi at the lake house is (a private value, withheld), on the router\'s sticker."' in out
        assert '"At the boathouse the network wants (a private value, withheld) now."' in out
    if "--json" not in argv and argv[1] == "history":
        assert "wifi password: (a private value, withheld) [current]" in out
        assert 'contact-b: "Still (a private value, withheld), the sticker is right."' in out


# Legacy claims whose label names a credential however it is written (underscores, a token's name), and one whose
# value the detector would not seal under any label (a weak password). Every value is made up.
TOKEN = "Maple-7731" + "-Cedar"
REFRESH = "Birch-4410" + "-Aspen"
WEAK = "sun" + "shine"
AUTH = "Zq7xK9mP" + "2vL4Bn8w"
SPELLED = [
    {"claim_id": "claim:underscored", "subject": "lake house", "predicate": "wifi_password", "value": WIFI,
     "quote": f"The wifi at the lake house is {WIFI}, on the router's sticker."},
    {"claim_id": "claim:access", "subject": "build server", "predicate": "access_token", "value": TOKEN,
     "quote": f"The build server uses {TOKEN} for its deploys."},
    {"claim_id": "claim:refresh", "subject": "build server", "predicate": "refresh_token", "value": REFRESH,
     "quote": f"The build server renews with {REFRESH}."},
    {"claim_id": "claim:weak", "subject": "cabin", "predicate": "wifi password", "value": WEAK,
     "quote": f"The cabin's wifi is {WEAK}, all lower case."},
    {"claim_id": "claim:auth", "subject": "mail relay", "predicate": "authKey", "value": AUTH,
     "quote": f"The mail relay takes {AUTH}."},
]
SPELLED = [{"state": "current", "verification": "unverified", "said_at": "2026-09-02T09:00:00+00:00",
            "corrections": 1, "source": {"source_id": "o-" + card["claim_id"], "source_version": "e" * 64,
                                          "session_id": "s1"}, **card} for card in SPELLED]


class SpelledSidecar:
    """The inventory routes, canned: contact-b's claims under spelled credential labels, and each one's history with
    a correction that repeats the value."""

    def __init__(self, home=None):
        pass

    def call(self, method, path, *, params=None, json_body=None, timeout=30.0):
        for card in SPELLED:
            if path.endswith(f"/claims/{card['claim_id']}/history"):
                return {"person_id": "contact-b", "claim_id": card["claim_id"], "complete": True, "gaps": [],
                        "limit_reached": False, "steps": [
                            {"event": "stated", "claim_id": card["claim_id"], "claim": card},
                            {"event": "corrected", "claim_id": card["claim_id"], "corrects": card["claim_id"],
                             "by": {"correction": f"Still {card['value']}; nothing changed.",
                                    "author": "contact-b", "source": {"source_id": "o-9"}}}]}
        assert path == "/v1/host/memory/inventory", path
        return {"person_id": "contact-b", "claims": SPELLED, "limit": 50, "next_page": None, "complete": True}


@pytest.mark.parametrize("argv", [
    ["memory", "inventory", "contact-b"], ["memory", "inventory", "contact-b", "--json"],
    *[["memory", "history", card["claim_id"]] for card in SPELLED],
    *[["memory", "history", card["claim_id"], "--json"] for card in SPELLED],
])
def test_a_value_under_a_credential_label_is_withheld_however_the_label_is_spelled(argv, monkeypatch, capsys):
    """A label that names a credential (``wifi_password``, ``access_token``, ``refresh_token``, ``authKey``) is read
    as the validator reads it, so its whole value is withheld wherever the listing or the history prints it (the value,
    the quotation, a correction), a weak value the detector would never seal included, in text and ``--json``."""
    monkeypatch.setattr("colonymind.mind.cli.Sidecar", SpelledSidecar)
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=None)
    memory_cli().add_parser(parser.add_subparsers(dest="command"))
    assert memory_cli().run(parser.parse_args(argv)) == 0
    out = capsys.readouterr().out
    for value in (WIFI, TOKEN, REFRESH, WEAK, AUTH):
        assert value not in out and json.dumps(value)[1:-1] not in out, value
    assert "(a private value, withheld)" in out
    if argv[1] == "inventory" and "--json" not in argv:
        assert "[current] refresh_token: (a private value, withheld)  (unverified)" in out
        assert '"The cabin\'s wifi is (a private value, withheld), all lower case."' in out
    if argv[1] == "history" and "--json" not in argv:
        assert "Still (a private value, withheld); nothing changed." in out


def test_a_value_under_a_credential_label_is_withheld_whole_but_not_inside_other_words():
    """The whole value under a credential's label is withheld wherever it stands as itself, and never inside another
    word: a PIN of ``yes`` is withheld as a value and in a quotation, and the ``eyes`` beside it are left alone."""
    from colonymind.secrets.claims import withheld_claims
    cards = [{"subject": "gate", "predicate": "pin", "value": "yes", "quote": "The gate pin is yes, she said with her eyes."},
             {"subject": "gate", "predicate": "colour", "value": "green", "quote": "Yes, the gate is green."}]
    shown = withheld_claims(cards)
    assert shown[0]["value"] == "(a private value, withheld)"
    assert shown[0]["quote"] == "The gate pin is (a private value, withheld), she said with her eyes."
    assert shown[1] == cards[1]


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_the_history_prints_the_chain_in_order_with_erased_steps_as_markers(sidecar, state, capsys):
    river, lake, hill = await office_chain(state)
    code, out, _ = run(['memory', 'history', river], capsys)
    assert code == 0
    assert out.index('stated') < out.index('superseded') < out.index('retracted')
    assert '"Starting 2026-03-12, my office is now in Lake."' in out
    from colonymind.turns import get_turn_idempotency_ledger
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(get_turn_idempotency_ledger(state))
    curated.forget_sources(get_turn_idempotency_ledger(state), 'contact-a', source_ids=['o-2'])
    assert historical_bytes(get_turn_idempotency_ledger(state)) == _native_before_withdrawal
    code, out, _ = run(['memory', 'history', hill, '--as', 'contact-a'], capsys)
    assert code == 0 and '(erased)' in out and ('now in Lake' not in out)
    code, raw, _ = run(['memory', 'history', hill, '--json'], capsys)
    assert json.loads(raw)['claim_id'] == hill and json.loads(raw)['complete'] is True
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(get_turn_idempotency_ledger(state))
    curated.forget_sources(get_turn_idempotency_ledger(state), 'contact-a', source_ids=['o-3'])
    assert historical_bytes(get_turn_idempotency_ledger(state)) == _native_before_withdrawal
    code, out, _ = run(['memory', 'history', river], capsys)
    assert code == 0 and f'(what came after {lake} was erased with it: the chain may go on)' in out
    assert 'bound' not in out
