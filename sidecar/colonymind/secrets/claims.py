"""A claim's words, read for secrets (docs/SECRETS.md).

A claim is shown as a card: its predicate beside its value (``wifi password: ...``), its quotation, and in its history
a correction's words. A predicate can say what its value is though neither field alone looks like a credential, so a
reader that scans each field alone shows the value. Every place that shows claims (the owner console's memory pages,
``colonymind memory inventory`` and ``history``) reads them here, the same way: ``withheld_claims``.

A claim is never formed with a label that names a credential (``names_a_credential``, which the claims validator
applies to the subject and the predicate), so the reading here matters for claims formed before that rule, and for a
value the capture detector would seal in any of a claim's words. The capture detector, the validator and the reading
here share one vocabulary of credential names: the labels the detector reads before a secret's value, and the names
it seals a value under in an assignment or a JSON field (``API_KEY``, ``refresh_token``), with the names it reads as
about a credential, never one (``TOKEN_COUNT``, ``AUTH_TOKEN_ID``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from colonymind.secrets.detect import _NOT_SECRET_NAME, _SECRET_NAME, HIDDEN, REFERENCE, WITHHELD, find_secrets

#: A claim's words, as its card and its history show them: what the claim is about, what it says the value is, the
#: value, the words it was said in, and a correction's words.
CLAIM_WORDS = ("subject", "predicate", "value", "quote", "correction", "excerpt")

#: The words that name a credential in a claim's label: every label the capture detector reads before a secret's
#: value (its ``labeled`` rule and the seed phrase), and the claims validator's own credential words (a test keeps the
#: detector's labels here). Matched on a label's words (``label_words``), spaced.
CREDENTIAL_LABEL = re.compile(
    r"\b(?:(?:wi-?fi\s+)?password|passwords|passwd|passphrase|passcode|pin(?:\s+code)?|otp|one[- ]time code|"
    r"2fa code|verification code|security code|backup codes?|recovery codes?|recovery phrase|seed phrase|mnemonic|"
    r"api.?keys?|secret key|access key|private key|auth token|access token|cvv|cvc|credentials?|secrets?)\b",
    re.IGNORECASE)


def label_words(label: str) -> List[str]:
    """A label's words, however it is written: ``wifi_password``, ``refreshToken``, ``AUTH_KEY`` and ``wifi
    password`` are all two words."""
    return re.findall(r"[A-Za-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label))


def names_a_credential(label: Any) -> bool:
    """Whether a claim's subject or predicate names a credential: a label the capture detector reads before a
    secret's value (``CREDENTIAL_LABEL``), or a name it seals a value under in an assignment or a JSON field
    (``API_KEY``, ``refresh_token``), read word by word (a secretary is no secret). A name the detector reads as
    about a credential, never one (``TOKEN_COUNT``, ``AUTH_TOKEN_ID``, ``password length``), names none."""
    if not isinstance(label, str):
        return False
    words = label_words(label)
    if not words or _NOT_SECRET_NAME.search("_".join(words).upper()):
        return False
    if CREDENTIAL_LABEL.search(" ".join(words)):
        return True
    runs = ([word.upper() for word in words[start:start + size]]
            for size in (1, 2, 3) for start in range(len(words) - size + 1))
    return any(_SECRET_NAME.fullmatch("_".join(run)) or _SECRET_NAME.fullmatch("".join(run)) for run in runs)


def claim_secrets(claim: Dict[str, Any]) -> Set[str]:
    """Every value the capture detector would seal in ``claim``'s words, read each alone and read as the card shows
    its predicate beside its value: ``predicate: value``, and with the value quoted, since the value field is one
    value however many words it has (the detector reads a quoted value as the exact characters between the quotes).
    The predicate is read by its words (``wifi_password`` as ``wifi password``), as the validator reads it. A
    predicate that names a credential ("wifi password") so marks its value though neither field alone looks like
    one; ``labelled_value`` withholds such a value whole, whatever the detector makes of it."""
    words = {name: str(claim[name]) for name in CLAIM_WORDS if claim.get(name) is not None}
    readings = list(words.values())
    predicate, value = " ".join(label_words(words.get("predicate", ""))), words.get("value", "")
    if predicate and value:
        readings.append(f"{predicate}: {value}")
        mark = next((mark for mark in "\"'`" if mark not in value), None)
        if mark and "\n" not in value:
            readings.append(f"{predicate}: {mark}{value}{mark}")
    return {hit.value for reading in readings for hit in find_secrets(reading)}


def labelled_value(claim: Dict[str, Any]) -> Optional[str]:
    """The whole value of a claim whose own subject or predicate names a credential (``names_a_credential``), as the
    validator would have refused it: a value that label calls a credential, whatever it looks like (a weak password
    too). None for any other claim, and for a value that is already a sealed reference or a withheld marker."""
    value = claim.get("value")
    if value is None or not (names_a_credential(claim.get("subject")) or names_a_credential(claim.get("predicate"))):
        return None
    value = str(value).strip()
    if not value or REFERENCE.fullmatch(value) or value in (HIDDEN, WITHHELD):
        return None
    return value


def _alone(value: str) -> str:
    """``value`` as a pattern that finds it standing as itself, never inside a longer word (``yes`` in ``eyes``)."""
    return (r"(?<![A-Za-z0-9])" if value[0].isalnum() else "") + re.escape(value) + \
        (r"(?![A-Za-z0-9])" if value[-1].isalnum() else "")


def withhold(text: str, secrets: Iterable[str], whole: Iterable[str] = ()) -> str:
    """``text`` with every one of ``secrets`` (values the detector finds, wherever they stand) and of ``whole``
    (values a credential's label marks, where they stand as themselves) replaced by the withheld marker, in one pass
    (the longest first)."""
    patterns = {value: re.escape(value) for value in whole}
    patterns.update((value, re.escape(value)) for value in secrets)
    for value in set(whole) - set(secrets):
        patterns[value] = _alone(value)
    ordered = sorted(patterns, key=len, reverse=True)
    return re.sub("|".join(patterns[value] for value in ordered), HIDDEN, text) if ordered else text


def withheld_claims(claims: List[Any]) -> List[Any]:
    """``claims`` (cards, or a history's steps holding them) as they are shown together: a value the detector would
    seal in any claim's words (``claim_secrets``), and the whole value of a claim whose own label names a credential
    (``labelled_value``), is withheld from the words of every one, so such a value never stands in its card, its
    quotation, a correction or another card shown with it."""
    def cards(item: Any) -> List[Dict[str, Any]]:
        if not isinstance(item, dict):
            return []
        return [item] + [item[key] for key in ("claim", "by") if isinstance(item.get(key), dict)]

    shown = [card for item in claims for card in cards(item)]
    secrets: Set[str] = set().union(*(claim_secrets(card) for card in shown))
    whole: Set[str] = {value for value in map(labelled_value, shown) if value}
    if not secrets and not whole:
        return claims

    def withheld(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        copy = {key: withheld(part) if key in ("claim", "by") else part for key, part in item.items()}
        for name in CLAIM_WORDS:
            if copy.get(name) is not None:
                copy[name] = withhold(str(copy[name]), secrets, whole)
        if labelled_value(item):
            copy["value"] = HIDDEN
        return copy

    return [withheld(item) for item in claims]


__all__ = ["CLAIM_WORDS", "CREDENTIAL_LABEL", "claim_secrets", "label_words", "labelled_value", "names_a_credential",
           "withheld_claims", "withhold"]
