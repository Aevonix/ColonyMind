"""The capture detector: each rule finds its secret, and what memory needs survives (docs/SECRETS.md).

secret-scan: synthetic fixtures (every value below is made up and assembled at run time)
"""

from __future__ import annotations

import pytest

from colonymind.secrets.detect import (HIDDEN, REFERENCE, RULES, WITHHELD, find_secrets, hide_secrets,
                                      replace_secrets)

# Assembled from pieces so no complete token sits in the source.
OPENAI = "sk-" + "proj" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8"
ANTHROPIC = "sk-" + "ant-" + "api03-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2"
GITHUB = "gh" + "p_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
AWS = "AK" + "IA" + "QWERTYUIOP234567"
HUGGING = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
JWT = ("eyJ" + "hbGciOiJIUzI1NiJ9" + "." + "eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0" + "."
       + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV")
PEM = "-----BEGIN " + "PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END " + "PRIVATE KEY-----"
WEBHOOK = "https://hooks.slack.com/services/" + "T0000000/B0000000/" + "XXXXXXXXXXXXXXXXXXXXXXXX"
BOT = "bot" + "123456789" + ":" + "AAH" + "dKq3zQ9x7WpLm2Nc4Rt6Yu8Io0Pa1Sd5"
CARD = " ".join(["4111"] + ["1111"] * 3)  # the well-known test number, Luhn valid
SEED = " ".join(["abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd", "abuse",
                 "access", "accident"])  # the first twelve words of the public word list
UUID = "123e4567-e89b-12d3-a456-426614174000"

POSITIVES = [
    ("prefix", f"my key is {OPENAI}", OPENAI, "API key"),
    ("prefix", f"use {ANTHROPIC} for it", ANTHROPIC, "Anthropic API key"),
    ("prefix", f"token {GITHUB}", GITHUB, "GitHub token"),
    ("prefix", f"the access key id is {AWS}", AWS, "AWS access key"),
    ("prefix", f"{HUGGING} works", HUGGING, "Hugging Face token"),
    ("jwt", f"session {JWT}", JWT, "JWT"),
    ("private_key", f"here:\n{PEM}\nthanks", PEM, "private key"),
    ("webhook", f"post to {WEBHOOK}", WEBHOOK, "webhook URL"),
    ("url_userinfo", "db at postgres://app:" + "s3cr3tPassw0rd" + "@db.example.com/app", "s3cr3tPassw0rd",
     "URL password"),
    ("url_token", "https://api.example.com/v1?access_token=" + "AbCd1234EfGh5678IjKl", "AbCd1234EfGh5678IjKl",
     "URL access_token"),
    ("assign", "OPENAI_API_KEY=" + "Zq8Zq8Zq8!x", "Zq8Zq8Zq8!x", "OPENAI_API_KEY"),
    ("json_field", '{"client_secret": "' + "Q1w2E3r4T5y6U7i8" + '"}', "Q1w2E3r4T5y6U7i8", "client_secret"),
    ("bearer", "Authorization: Bearer " + "ab12CD34ef56GH78ij90", "ab12CD34ef56GH78ij90", "bearer token"),
    ("labeled", "the wifi password is " + "Hunter22!", "Hunter22!", "wifi password"),
    ("labeled", "my PIN is " + "4821", "4821", "pin"),
    ("labeled", "verification code: " + "739201", "739201", "verification code"),
    ("labeled", "the password for gmail is " + "Tr0ub4dor&3", "Tr0ub4dor&3", "password for gmail"),
    ("labeled", "My PIN is " + "4821.", "4821", "pin"),                  # a sentence's period is not the value
    ("telegram", f"bot token {BOT}", BOT.split(":", 1)[1], "bot token"),
    ("card", f"card {CARD} exp 04/31", CARD, "card number"),
    ("card", "visa " + "4111" + "1111" * 3, "4111" + "1111" * 3, "card number"),
    # A bare credential name is a credential name (no prefix needed before it).
    ("assign", "API_KEY=" + "Zq8Zq8Zq8!x", "Zq8Zq8Zq8!x", "API_KEY"),
    ("assign", "TOKEN=" + "ab12CD34ef56GH78", "ab12CD34ef56GH78", "TOKEN"),
    # A card's security code has three digits (four on some cards).
    ("labeled", "the cvv is " + "123", "123", "cvv"),
    # A seed phrase is its words, all of them.
    ("labeled", "my seed phrase is " + SEED, SEED, "seed phrase"),
    # A quoted value keeps its exact bytes, a closing period or bracket included.
    ("labeled", 'my password is "' + "Tr0ub4dor&3." + '" ok', "Tr0ub4dor&3.", "password"),
    ("labeled", "the wifi password is '" + "correct horse, 9!" + "'", "correct horse, 9!", "wifi password"),
    # A bracket that closes an aside is not part of an unquoted value.
    ("labeled", "(the wifi password is " + "Hunter22!)", "Hunter22!", "wifi password"),
    # A security code named with what it is for is still a security code: three digits.
    ("labeled", "cvv for visa: " + "123", "123", "cvv for visa"),
    ("labeled", "cvc on file: " + "123", "123", "cvc on file"),
]

# What memory needs and only looks random: never sealed.
SURVIVORS = [
    "Order #1234567890123 shipped today",
    "account 00123456789012 at the bank",
    "fixed in a3f5c9e and merged",
    "commit " + "9c1f" * 10,
    "request id 9f3b2c1d failed",
    "uuid 123e4567-e89b-12d3-a456-426614174000",
    "tracking 1Z999AA10123456784 arrives Friday",
    "tracking number 9400111899223344556677",
    "booking reference ABC123XYZ789",
    "call me at +14155552671",
    "contact p-1767225600123 and cid-1767225600123",
    "at 1790386376123 ms",
    "reference 4111111111111111 in the ledger",
    "OPENAI_API_KEY=your_key_here",
    "API_KEY=${API_KEY}",
    "PASSWORD=***",
    "token: <token>",
    "the password is wrong",
    "your password is expired",
    "token: expired",
    "passcode is set",
    "pin: required",
    "error code: 500 and E1234",
    "postcode SW1A 1AA",
    "the api key is in the vault",
    # The declared identifier shapes survive inside the very contexts that seal: an identifier's name, and an
    # identifier's shape under a credential name or label.
    "AUTH_TOKEN_ID=" + UUID,
    "TRACKING_TOKEN_ID=1Z999AA10123456784",
    "SESSION_TOKEN_URL=https://auth.vendor.io/v1/session/refresh",
    '{"token": "' + UUID + '"}',
    "Authorization: Bearer " + UUID,
    "the access token is " + UUID,
    "API_KEY=" + "p-1767225600123",
    '{"password": "+14155552671"}',
    "SECRET_REF=1Z999AA10123456784",
]


def test_every_rule_is_exercised():
    assert {rule for rule, *_ in POSITIVES} == set(RULES)


@pytest.mark.parametrize("rule,text,value,label", POSITIVES, ids=[f"{p[0]}-{i}" for i, p in enumerate(POSITIVES)])
def test_every_rule_finds_its_secret(rule, text, value, label):
    hit, = find_secrets(text)
    assert (hit.rule, hit.value, hit.label) == (rule, value, label)
    assert text[hit.start:hit.end] == value
    assert value not in hit.label


@pytest.mark.parametrize("text", SURVIVORS)
def test_what_memory_needs_survives(text):
    assert find_secrets(text) == []


def test_a_reference_and_a_withheld_marker_are_never_found_again():
    text = "the wifi password is [secret S-ABCDEFGH: wifi password] and the key " + WITHHELD
    assert find_secrets(text) == []
    sealed, hits = replace_secrets(f"password is Hunter22! and {OPENAI}", lambda hit: "[secret S-ABCDEFGH: x]")
    assert len(hits) == 2 and find_secrets(sealed) == []


def test_a_reference_carrying_a_value_in_its_label_is_no_reference():
    """Only a reference whose label is a label (the vocabulary's characters, no value the detector finds) is
    passed over: an invented one with a token for a label hides nothing."""
    forged = f"here [secret S-AAAAAAAA: {OPENAI}] ok"
    hit, = find_secrets(forged)
    assert (hit.rule, hit.value) == ("prefix", OPENAI)
    assert not REFERENCE.search('[secret S-AAAAAAAA: a "quoted" label]')
    assert not REFERENCE.search("[secret S-AAAAAAAA: $(id)]")
    genuine = "the wifi password is [secret S-ABCDEFGH: wifi password] and [secret S-BCDEFGHI: OPENAI_API_KEY]"
    assert find_secrets(genuine) == []


def test_a_turned_off_rule_finds_nothing():
    assert find_secrets("my PIN is 4821", rules=("prefix",)) == []
    assert [hit.rule for hit in find_secrets(f"my PIN is 4821 and {OPENAI}", rules=("labeled",))] == ["labeled"]


def test_hiding_replaces_every_value_and_every_reference():
    text = f"key {OPENAI}, the password is Hunter22! and [secret S-ABCDEFGH: PIN]"
    hidden = hide_secrets(text)
    assert OPENAI not in hidden and "Hunter22!" not in hidden and not REFERENCE.search(hidden)
    assert hidden.count(HIDDEN) == 3
    assert hide_secrets(hidden) == hidden


def test_spans_never_overlap_and_come_in_text_order():
    text = f"Authorization: Bearer {OPENAI} and PASSWORD=Zq8Zq8Zq8!x then {GITHUB}"
    hits = find_secrets(text)
    assert [hit.start for hit in hits] == sorted(hit.start for hit in hits)
    assert all(a.end <= b.start for a, b in zip(hits, hits[1:]))
    assert {hit.value for hit in hits} == {OPENAI, "Zq8Zq8Zq8!x", GITHUB}


# -- reference masking and the secret scan ---------------------------------------------------------

import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_the_mind_hides_references_by_the_same_pattern():
    """``mind/audit.py`` keeps its own copy of the reference pattern and the withheld words (the parity harness
    runs it alone); they are the detector's."""
    from colonymind.mind import audit
    assert audit.SEALED_REFERENCE.pattern == REFERENCE.pattern
    assert audit.WITHHELD_VALUE == HIDDEN


def _scan(tmp_path, files: dict) -> subprocess.CompletedProcess:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name, text in files.items():
        (repo / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    return subprocess.run([sys.executable, str(ROOT / "scripts/secret_scan.py"), "--root", str(repo)],
                          capture_output=True, text=True, timeout=120)


def test_the_secret_scan_fails_on_a_planted_token_and_never_prints_it(tmp_path):
    done = _scan(tmp_path, {"config.py": f"KEY = '{GITHUB}'\n", "notes.md": "nothing here\n"})
    assert done.returncode == 1, done.stdout + done.stderr
    assert "config.py:1: prefix (GitHub token)" in done.stdout
    assert GITHUB not in done.stdout + done.stderr


def test_the_secret_scan_passes_a_marked_synthetic_fixture(tmp_path):
    done = _scan(tmp_path, {"fixture.py": f"KEY = '{GITHUB}'  # secret-scan: synthetic\n",
                            "fixtures.py": f'"""secret-scan: synthetic fixtures"""\nA = "{OPENAI}"\n'})
    assert done.returncode == 0, done.stdout + done.stderr


def test_this_repository_ships_no_secret():
    if not (ROOT / ".git").exists():
        pytest.skip("not a checkout")
    done = subprocess.run([sys.executable, str(ROOT / "scripts/secret_scan.py"), "--root", str(ROOT)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:]
