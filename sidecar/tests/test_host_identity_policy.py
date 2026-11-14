"""Host language policies stay optional; verification requires captured receipts."""
from contextlib import closing
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from colonymind.config import ConfigError, load_config
from colonymind.mind import drift, status
from colonymind.memory.notice import _preview, recall_notice_details
from colonymind.memory.recall import render_memory_context
from colonymind.turns.idempotency import source_message_hash
from test_mind_drift import T0, world
from test_hands_controller import c, running


LONG = ("The returned record includes the requested entry and the corresponding recorded time. "
        "This reply describes the available evidence and leaves the next action for a later turn.")
HOST_VALIDATORS = {
    "hedge": [r"\bperhaps\b"], "agency": [r"\bi decided\b"],
    "causal": [r"\bbecause\b"], "apology": [r"\bsorry\b"],
    "adjustment": [r"\bcorrected\b"], "tone": [r"\bstock service phrase\b"],
}
# Authored fixture policy with the same transformation contract as the prior renderer.
STATUS_POLICY = [
    {"pattern": r"\s+[\u2010-\u2015\u2e3a\u2e3b\ufe58\ufe63\uff0d]+\s+", "replacement": ", "},
    {"pattern": r"[\u2010-\u2015\u2e3a\u2e3b\ufe58\ufe63\uff0d]", "replacement": "-"},
]
NOTICE_POLICY = [{"pattern": r"[\u2013\u2014]", "replacement": "-"}]


def replies(text=LONG, *, receipts=()):
    return [drift.Reply(f"reply-{i}", "session", T0 + timedelta(minutes=i), "Status?", text, receipts)
            for i in range(3)]


def test_another_host_has_no_persona_or_punctuation_policy(tmp_path):
    config = load_config(tmp_path, environ={})
    assert config.get("mind.drift.validators") == {}
    assert config.get("mind.status_replacements") == []
    assert config.get("hands.notices.forbidden_patterns") == []
    assert config.get("context.recall_notice_replacements") == []
    reading = drift.signals(replies("As an AI language model, I'd be happy to help. " + LONG), [T0])
    for name in drift.SIGNALS:
        if name != "missing_verification":
            assert reading[name] == {"value": None, "n": 0, "turns": []}
    assert reading["missing_verification"]["value"] == 1.0
    assert _preview("café — suite–notes") == "café — suite–notes"


@pytest.mark.parametrize("phrase", ["Let me check.", "I checked the logs.", "According to the docs.",
                                  "The output shows this.", "I can't verify this."])
def test_verification_words_cannot_substitute_for_a_tool_receipt(phrase):
    reading = drift.signals(replies(phrase + " " + LONG), [])
    assert reading["missing_verification"]["value"] == 1.0
    assert reading["missing_verification"]["receipts"] == {}


def test_returned_tools_count_without_any_claimed_verification_wording():
    reading = drift.signals(replies(receipts=("tool:observed-call",)), [])
    assert reading["missing_verification"] == {
        "value": 0.0, "n": 3, "turns": [],
        "receipts": {f"reply-{i}": ["tool:observed-call"] for i in range(3)},
    }


def test_host_validators_measure_only_the_supplied_language():
    steady = drift.signals(replies("I decided because the evidence was clear. " + LONG), [],
                           validators=HOST_VALIDATORS)
    changed = drift.signals(replies("Perhaps this holds. " + LONG), [], validators=HOST_VALIDATORS)
    assert steady["agency_fading"]["value"] == 0.0
    assert steady["bare_assertions"]["value"] == 0.0
    assert changed["agency_fading"]["value"] == 1.0
    assert changed["bare_assertions"]["value"] == 1.0
    turns = [*replies(), *[drift.Reply(f"after-{i}", "session", T0 + timedelta(hours=2, minutes=i),
                                     "Go", "Stock service phrase.") for i in range(3)]]
    assert drift.signals(turns, [T0 + timedelta(hours=1)], validators=HOST_VALIDATORS)[
        "tone_after_compression"]["value"] == 1.0
    assert drift.signals(turns, [T0 + timedelta(hours=1)], validators={"tone": ["different phrase"]})[
        "tone_after_compression"]["value"] == 0.0
    corrected = [drift.Reply("wrong", "session", T0, "That was wrong.", "Sorry."),
                 drift.Reply("fixed", "session", T0, "That was wrong.", "Sorry, corrected.")]
    assert drift.signals(corrected, [], validators=HOST_VALIDATORS)["apology_not_correction"]["value"] == 0.5


@pytest.mark.parametrize("policy", [
    {"mind": {"drift": {"validators": {"tone": ["("]}}}},
    {"mind": {"drift": {"validators": {"tone": "pattern"}}}},
    {"mind": {"drift": {"validators": {"unsupported": []}}}},
    {"mind": {"drift": []}},
    {"mind": {"status_replacements": [{"pattern": "x", "replacement": r"\9"}]}},
    {"context": {"recall_notice_replacements": [{"pattern": "", "replacement": "x"}]}},
    {"hands": {"notices": {"forbidden_patterns": ["("]}}},
])
def test_invalid_host_policies_fail_configuration_loading(tmp_path, policy):
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump(policy))
    with pytest.raises(ConfigError, match="host text policy"):
        load_config(tmp_path, environ={})


async def test_drift_reads_host_settings_and_audits_the_policy(world):
    world.mind.config["drift"] = {"validators": deepcopy(HOST_VALIDATORS)}
    for i in range(3):
        world.turn(f"hedge-{i}", "Status?", "Perhaps this holds. " + LONG)
    result = await world.mind.drift.run()
    assert result["signals"]["agency_fading"]["value"] == 1.0
    assert result["validators"] == HOST_VALIDATORS
    assert world.mind.drift.history()[-1]["signal_version"] == drift.SIGNAL_VERSION
    assert world.mind.drift.history()[-1]["validators"] == HOST_VALIDATORS
    world.mind.config["drift"] = {"validators": {}}
    world.shift(hours=1)
    assert (await world.mind.drift.run())["signals"]["agency_fading"]["value"] is None


def test_policy_and_receipt_semantics_changes_start_a_new_signal_baseline():
    entry = {"version": 1, "signal_version": drift.SIGNAL_VERSION, "validators": HOST_VALIDATORS,
             "signals": {"missing_verification": {"value": 1.0}}}
    prior = {"version": 1, "signals": {"missing_verification": 0.0}}
    assert drift.assess(entry, [prior, prior])[1] == []
    same = {**prior, "signal_version": drift.SIGNAL_VERSION, "validators": HOST_VALIDATORS}
    assert drift.assess(entry, [same, same])[1]
    other = {**same, "validators": {}}
    assert drift.assess(entry, [other, other])[1] == []


@pytest.mark.parametrize("messages,expected", [
    ([{"role": "user", "content": "Check"}, {"role": "tool", "content": "Returned bytes", "tool_call_id": "call"},
      {"role": "assistant", "content": LONG}], []),
    ([{"role": "user", "content": "Check"}, {"role": "assistant", "content": "tool:call " + LONG}], []),
    ([{"role": "tool", "content": "Earlier result", "tool_call_id": "call"},
      {"role": "user", "content": "New instruction"}, {"role": "assistant", "content": LONG}], []),
    ([{"role": "user", "content": "Check"},
      {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "function": {"name": "read"}}]},
      {"role": "tool", "content": "", "tool_call_id": "call"},
      {"role": "assistant", "content": LONG}], ["tool:call"]),
    ([{"role": "user", "content": "Check"}, {"role": "tool", "content": "Returned bytes"},
      {"role": "assistant", "content": LONG}], []),
    ([{"role": "user", "content": "Check"}, {"role": "assistant", "content": LONG},
      {"role": "tool", "content": "Later result", "tool_call_id": "call"}], []),
])
def test_captured_receipts_require_a_returned_tool_before_the_reply(world, messages, expected):
    world.ledger.record_source("receipt-turn", contact_id=world.mind.owner_id, session_id="session",
                               messages=messages, occurred_at=world.now.isoformat(), derive_claims=False)
    assert world.mind.drift._replies(world.now)[0].tool_receipts == expected


def native_receipt(world, *, at=None):
    owner = world.mind.owner_id
    world.ledger.record_source("instruction", contact_id=owner, session_id="native",
        messages=[{"role": "user", "content": "Inspect the source."}], occurred_at=world.now.isoformat(),
        derive_claims=False)
    origin = world.ledger.source_references(["instruction"], contact_id=owner, session_id="native")[0]
    content = "Observed source bytes."
    native = {'profile_id': 'a' * 64, 'session_id': 'native', 'task_id': 'task', 'turn_id': 'instruction',
              'tool_call_id': 'observed-call', 'api_request_id': 'request', 'tool_name': 'read', 'message_id': 1,
              'timestamp': (at or world.now).timestamp(), 'result_sha256': hashlib.sha256(content.encode()).hexdigest()}
    ident = 'native-observation:' + hashlib.sha256(json.dumps(native, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    world.ledger.record_source(ident, contact_id=owner, session_id='native', derive_claims=False,
        messages=[{'role': 'tool', 'content': content, '_native_tool_observation': 'native-tool-observation-v1',
                   '_observation_sources': [origin], 'provenance': {'kind': 'native-tool-observation-v1',
                       'native': native, 'selection': {'author': 'model', 'reason': 'Relevant to this instruction.'}}}],
        occurred_at=(at or world.now).isoformat())
    message_hash = source_message_hash("native", {"role": "user", "content": "Inspect the source."})
    world.ledger.record_source("response", contact_id=owner, session_id="native",
        messages=[{"role": "assistant", "content": LONG,
                   "_supplied_inputs": [{"source_id": "instruction", "input_message_hash": message_hash}]}],
        occurred_at=world.now.isoformat(), derive_claims=False)
    return ident, origin


def test_native_receipt_after_the_reply_is_not_verification(world):
    native_receipt(world, at=world.now + timedelta(minutes=1))
    assert world.mind.drift._replies(world.now)[0].tool_receipts == []


def test_native_receipt_from_a_different_session_is_not_verification(world):
    native_receipt(world)
    with closing(world.ledger._connect()) as conn:
        conn.execute("UPDATE turn_sources SET session_id='other' WHERE turn_id='response'")
        conn.commit()
    assert world.mind.drift._replies(world.now)[0].tool_receipts == []


def test_native_receipt_tampered_content_is_not_verification(world):
    ident, _ = native_receipt(world)
    with closing(world.ledger._connect()) as conn:
        messages = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=?", (ident,)).fetchone()[0])
        messages[0]["content"] = "Tampered result"
        conn.execute("UPDATE turn_sources SET messages_json=? WHERE turn_id=?", (json.dumps(messages), ident))
        conn.commit()
    assert world.mind.drift._replies(world.now)[0].tool_receipts == []


async def test_host_status_policy_preserves_original_bytes_and_neutral_text():
    mind = SimpleNamespace(level="standard", enabled=False, off_reason="café — queue–held", faculties={}, config={})
    neutral = status.build(mind)
    assert "café — queue–held" in neutral
    mind.config["status_replacements"] = STATUS_POLICY
    expected = ("Mind: off (café, queue-held): nothing runs on its own. Autonomy when on: standard.").encode()
    assert status.build(mind).encode() == expected
    assert (await status.collect(mind)).encode() == expected


def test_recall_notice_policy_is_read_from_host_settings(tmp_path, monkeypatch):
    rows = [{"title": "café — suite–notes", "id": "fixture", "content": "café — suite–notes"}]
    content = render_memory_context(rows)
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    assert "café — suite–notes" in recall_notice_details(rows, content=content, contact_id="person")[0]
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"context": {"recall_notice_replacements": NOTICE_POLICY}}))
    assert recall_notice_details(rows, content=content, contact_id="person")[0].encode() == (
        "• café - suite-notes (date unknown)").encode()


async def test_neutral_hand_notices_accept_punctuation_and_host_policy_rejects_it(c):
    plan = await running(c)
    line = "Progress — source–notes"
    accepted = await c.notice("ntc:neutral", plan["hand_id"], "routine", line)
    assert accepted["line"].encode() == line.encode()
    c.config["notices"]["forbidden_patterns"] = [r"[\u2013\u2014]"]
    with pytest.raises(ValueError):
        await c.notice("ntc:host", plan["hand_id"], "routine", line)
    for i, malformed in enumerate(("x" * 201, "", "one\ntwo", "one\rtwo")):
        with pytest.raises(ValueError):
            await c.notice(f"ntc:transport-{i}", plan["hand_id"], "routine", malformed)

from colonymind.memory import curated

def test_historical_observations_are_not_receipts_and_keep_erasure_lineage(world):
    ident, origin = native_receipt(world)
    reading = world.mind.drift._replies(world.now)
    assert reading[0].tool_receipts == []
    world.turn("unrelated", "Other question", LONG, session="native")
    reading = {item.turn_id: item.tool_receipts for item in world.mind.drift._replies(world.now)}
    assert reading["unrelated"] == [] and reading["response"] == []
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(world.ledger)
    _source_forget_receipt = curated.forget_sources(world.ledger, world.mind.owner_id, source_ids=[origin['source_id']])
    assert historical_bytes(world.ledger) == _native_before_forget
    assert all(curated.is_forgotten(world.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert all(ident not in item.tool_receipts for item in world.mind.drift._replies(world.now))
