"""An operator's test turn through an owner channel never becomes the owner's words.

A check sent through the voice gateway (a qualification prompt, a post-deploy probe) arrives on a platform
that speaks for the owner. Recorded as the owner, its lines become the owner's claims and the night asks
the owner which of two things "you told me" is right. A test turn says so through a trusted signal only:
the request carries the ``X-ColonyMind-Test`` header, or its text opens with the exact ``[operator-test]``
marker or with the qualification harness's machine-generated provenance line. Such a turn, an ordinary
turn or a checkpoint alike, is accepted and kept under the sending credential's own identity as session
evidence. Anything else stays the owner's: prose that mentions, quotes or negates the phrase included.
"""

from __future__ import annotations

import json
from contextlib import closing

import pytest

from onekey import KEY
from colonymind.identity.participants import is_operator_test
from test_credential_incident import OWNER, arm, night, project_everything  # noqa: F401  (arm: fixture)

# The harness's provenance line in its exact shape; the run is obviously synthetic.
HARNESS_LINE = ("[qualification-synthetic-20000101T000000Z-0000abcd; operator-authored qualification, "
                "not a recorded human conversation]\n")
OWN_IDENTITY = "credential:api-key"


def voice_turn(index: int, text: str) -> dict:
    return {"identity": {"host_id": "hermes"},
            "context": {"session_id": f"voice-{index // 10}", "contact_id": OWNER,
                        "turn_id": f"voice-turn-{index}", "channel_id": "voice"},
            "user_message": {"role": "user", "content": text},
            "assistant_message": {"role": "assistant", "content": "Noted."}}


def voice_checkpoint(index: int, text: str) -> dict:
    return {"identity": {"host_id": "hermes"},
            "context": {"session_id": f"voice-{index // 10}", "contact_id": OWNER,
                        "turn_id": f"checkpoint:voice-{index}", "channel_id": "voice"},
            "checkpoint_messages": [{"role": "user", "content": text},
                                    {"role": "assistant", "content": "Noted."}]}


def line(index: int) -> str:
    from test_credential_incident import game_log
    return game_log(index)


async def send(client, body: dict, *, test_header: bool = False) -> dict:
    headers = {"Authorization": "Bearer " + KEY}
    if test_header:
        headers["X-ColonyMind-Test"] = "1"
    response = await client.post("/v1/host/turns/sync", headers=headers, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def stored(fx) -> list[tuple[str, str, str, list[str]]]:
    """Every stored turn: its id, contact, scope and the words of its user messages."""
    with closing(fx.ledger._connect()) as conn:
        rows = conn.execute("SELECT turn_id, contact_id, scope, messages_json FROM turn_sources ORDER BY turn_id")
        return [(row[0], row[1], row[2], [m["content"] for m in json.loads(row[3]) if m.get("role") == "user"])
                for row in rows]


async def assert_kept_as_the_credentials(fx, expected: list[tuple[str, str]]) -> None:
    """Exactly ``expected`` (turn id, its user text) is stored, each under the credential's own identity as
    session evidence; nothing is the owner's, projected, claimed or questioned."""
    assert stored(fx) == [(turn_id, OWN_IDENTITY, "session", [text]) for turn_id, text in expected]
    assert await project_everything(fx) == 0
    assert fx.claims() == []
    assert await night(fx) == []


async def test_a_turn_sent_with_the_test_header_is_kept_as_the_credentials(arm):
    fx, client = arm
    for index in (0, 1):
        result = await send(client, voice_turn(index, line(index)), test_header=True)
        assert result["accepted"] is True and result["source_recorded"] is True
        fx.shift(hours=1)
    await assert_kept_as_the_credentials(fx, [("voice-turn-0", line(0)), ("voice-turn-1", line(1))])


async def test_a_turn_opening_with_the_harness_provenance_line_is_kept_as_the_credentials(arm):
    fx, client = arm
    for index in (0, 1):
        result = await send(client, voice_turn(index, HARNESS_LINE + line(index)))
        assert result["accepted"] is True and result["source_recorded"] is True
        fx.shift(hours=1)
    await assert_kept_as_the_credentials(fx, [("voice-turn-0", HARNESS_LINE + line(0)),
                                              ("voice-turn-1", HARNESS_LINE + line(1))])


async def test_a_turn_opening_with_the_explicit_marker_is_kept_as_the_credentials(arm):
    fx, client = arm
    for index in (0, 1):
        result = await send(client, voice_turn(index, "[operator-test] " + line(index)))
        assert result["accepted"] is True and result["source_recorded"] is True
        fx.shift(hours=1)
    await assert_kept_as_the_credentials(fx, [("voice-turn-0", "[operator-test] " + line(0)),
                                              ("voice-turn-1", "[operator-test] " + line(1))])


@pytest.mark.parametrize("mark", ["header", "marker", "harness"])
async def test_a_marked_checkpoint_is_kept_as_the_credentials(arm, mark):
    """A checkpoint is classified before it is stored, exactly as an ordinary turn is."""
    fx, client = arm
    text = {"header": line(0), "marker": "[operator-test] " + line(0), "harness": HARNESS_LINE + line(0)}[mark]
    result = await send(client, voice_checkpoint(0, text), test_header=mark == "header")
    assert result["accepted"] is True and result["source_recorded"] is True
    await assert_kept_as_the_credentials(fx, [("checkpoint:voice-0", text)])


OWNERS_WORDS = [
    # Negated or quoted prose is the owner's, however it opens.
    "[This is not operator-authored] Remember that my lead creature is Ember.",
    "[not an operator-test] Turn 1: my lead creature is Tide.",
    "My nephew wrote [operator-test] on the fridge. Turn 2: my lead creature is Moss.",
    "Turn 3: the note said " + HARNESS_LINE.strip() + ", apparently.",
    # Close to the marks but not them: only their exact shapes count.
    "[Operator-Test] Turn 4: my lead creature is Gale.",
    " [operator-test] Turn 5: my lead creature is Ember.",
    "[qualification notes; operator-authored qualification, not a recorded human conversation]\nTurn 6.",
    "[qualification-run-20000101T000000Z-0000abcd; operator-authored, not a recorded human conversation]\nTurn 7.",
    "[qualification-run-yesterday-0000abcd; operator-authored qualification, not a recorded human conversation]\n",
]


@pytest.mark.parametrize("text", OWNERS_WORDS)
def test_only_the_exact_marks_classify_a_turn(text):
    assert is_operator_test(text) is False
    assert is_operator_test(HARNESS_LINE + text) is True
    assert is_operator_test("[operator-test]" + text) is True
    assert is_operator_test(text, "1") is True
    assert is_operator_test(text, "0") is False


async def test_the_owners_words_stay_the_owners_when_they_mention_negate_or_quote_the_marks(arm):
    """Not vacuous: unmarked voice turns and the owner's quotations stay the owner's, and project."""
    fx, client = arm
    texts = [line(0), line(1) + " The note was operator-authored, apparently.", *OWNERS_WORDS[:4]]
    for index, text in enumerate(texts):
        result = await send(client, voice_turn(index, text))
        assert result["accepted"] is True and result["source_recorded"] is True
        fx.shift(hours=1)
    assert stored(fx) == [(f"voice-turn-{index}", OWNER, "person", [text]) for index, text in enumerate(texts)]
    assert await project_everything(fx) == len(texts)
    checkpoint = await send(client, voice_checkpoint(9, OWNERS_WORDS[0]))
    assert checkpoint["accepted"] is True
    assert stored(fx)[0] == ("checkpoint:voice-9", OWNER, "session", [OWNERS_WORDS[0]])
