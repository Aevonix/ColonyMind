"""The capture detector: which values in a message are secrets to seal (docs/SECRETS.md).

Calibrated on stored conversation data (the method is in docs/SECRETS.md): the rules below find credentials,
keys, passwords, codes and card numbers, and leave alone what memory needs and what only looks random: order and
account numbers, commit hashes, short hex ids, UUIDs, tracking and booking references, phone numbers, contact ids
and timestamps. A reference to a sealed value, and the marker of a withheld one, are never detected again, so
sealing is idempotent.

The sidecar and host adapters import this dependency-free detector. Both distributions package this module;
the Hermes adapter runs it on gateway messages before Hermes sees them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

# The rules, in the order they claim a span (an earlier rule's span wins an overlap).
RULES = ("prefix", "jwt", "private_key", "webhook", "url_userinfo", "url_token", "assign", "json_field", "bearer",
         "labeled", "telegram", "card")
# How a sealed value reads in stored words: an id from the vault and a label from a fixed vocabulary, in the
# characters ``clean_label`` keeps. Anything else in the brackets is text, and the detector reads it.
REFERENCE = re.compile(r"\[secret S-[A-Z2-7]{8}: [A-Za-z0-9 _.@+-]{1,60}\]")
# What stands where a value was found but could not be sealed (the store was unavailable).
WITHHELD = "[secret withheld: the store was unavailable; send it again]"
# What stands where a value or a reference is hidden from a message to someone else.
HIDDEN = "(a private value, withheld)"

_EDGE_L, _EDGE_R = r"(?<![A-Za-z0-9_-])", r"(?![A-Za-z0-9_-])"
# Vendor token prefixes with minimum tail lengths, and the label each is stored under.
_VENDORS: Tuple[Tuple[str, str], ...] = (
    (r"sk-ant-[A-Za-z0-9_-]{20,}", "Anthropic API key"),
    (r"sk-[A-Za-z0-9_-]{20,}", "API key"),
    (r"gh[pousr]_[A-Za-z0-9]{30,}", "GitHub token"),
    (r"github_pat_[A-Za-z0-9_]{30,}", "GitHub token"),
    (r"glpat-[A-Za-z0-9_-]{20,}", "GitLab token"),
    (r"xox[baprs]-[A-Za-z0-9-]{20,}", "Slack token"),
    (r"xapp-[A-Za-z0-9-]{20,}", "Slack token"),
    (r"AIza[A-Za-z0-9_-]{35}", "Google API key"),
    (r"ya29\.[A-Za-z0-9_-]{20,}", "Google access token"),
    (r"pplx-[A-Za-z0-9]{20,}", "Perplexity API key"),
    (r"fal_[A-Za-z0-9_-]{20,}", "Fal API key"),
    (r"fc-[A-Za-z0-9]{20,}", "Firecrawl API key"),
    (r"bb_live_[A-Za-z0-9_-]{20,}", "Browserbase API key"),
    (r"gAAAAA[A-Za-z0-9_=-]{40,}", "encrypted token"),
    (r"AKIA[A-Z0-9]{16}", "AWS access key"),
    (r"ASIA[A-Z0-9]{16}", "AWS access key"),
    (r"sk_live_[A-Za-z0-9]{20,}", "Stripe key"),
    (r"sk_test_[A-Za-z0-9]{20,}", "Stripe key"),
    (r"rk_live_[A-Za-z0-9]{20,}", "Stripe key"),
    (r"rk_test_[A-Za-z0-9]{20,}", "Stripe key"),
    (r"SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}", "SendGrid API key"),
    (r"hf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"r8_[A-Za-z0-9]{30,}", "Replicate token"),
    (r"npm_[A-Za-z0-9]{30,}", "npm token"),
    (r"pypi-[A-Za-z0-9_-]{40,}", "PyPI token"),
    (r"dop_v1_[A-Za-z0-9]{40,}", "DigitalOcean token"),
    (r"doo_v1_[A-Za-z0-9]{40,}", "DigitalOcean token"),
    (r"sk_[A-Za-z0-9]{30,}", "API key"),
    (r"tvly-[A-Za-z0-9]{20,}", "Tavily API key"),
    (r"exa_[A-Za-z0-9]{20,}", "Exa API key"),
    (r"xai-[A-Za-z0-9]{40,}", "xAI API key"),
    (r"shpat_[a-f0-9]{32}", "Shopify token"),
)
# Each vendor pattern alone, anchored at a word edge: the same patterns Hermes' redactor takes from the plugin.
VENDOR_PATTERNS = tuple(pattern for pattern, _ in _VENDORS)
_VENDOR_LABELS = tuple((re.compile(pattern), label) for pattern, label in _VENDORS)
_PREFIX = re.compile(_EDGE_L + r"(" + "|".join(VENDOR_PATTERNS) + r")" + _EDGE_R)
_JWT = re.compile(_EDGE_L + r"(eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{16,})" + _EDGE_R)
_PRIVATE_KEY = re.compile(r"(-----BEGIN[A-Z ]*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
                          r"-----END[A-Z ]*PRIVATE KEY(?: BLOCK)?-----)")
_WEBHOOK = re.compile(r"(https://(?:hooks\.slack\.com/services|discord(?:app)?\.com/api/webhooks)/[A-Za-z0-9/_-]{20,})")
_URL_USERINFO = re.compile(r"\b[a-z][a-z0-9+.-]{1,15}://[^\s:/@\"'<>]{1,64}:([^\s@/\"'<>]{4,128})@[A-Za-z0-9.-]+")
_URL_TOKEN = re.compile(r"[?&](access_token|api_key|apikey|token|secret|password|passwd|pwd|auth|client_secret|sig)"
                        r"=([^&\s\"'<>#]{16,})", re.IGNORECASE)
_ASSIGN = re.compile(r"\b([A-Z][A-Z0-9_]{1,80})\s*[=:]\s*([\"']?)([^\s\"']{8,512})\2")
# A credential's name (``API_KEY``, ``GITHUB_TOKEN``, ``DB_PASSWORD``), unless it names something about one: its id,
# its address, its kind or its lifetime (``AUTH_TOKEN_ID``, ``TOKEN_URL``, ``TOKEN_TYPE``, ``TOKEN_TTL``).
_SECRET_NAME = re.compile(r"API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY|AUTH_KEY")
_NOT_SECRET_NAME = re.compile(r"(?:^|_)(?:IDS?|UUID|URL|URI|ENDPOINT|HOST|PORT|PATH|FILE|DIR|NAME|TYPE|KIND|REF|"
                              r"EXPIRY|EXPIRES(?:_AT|_IN)?|TTL|LENGTH|LEN|COUNT|VERSION|HEADER|FORMAT|MODE)$")
_JSON_FIELD = re.compile(r'"(api_?key|apiKey|token|access_token|refresh_token|auth_token|id_token|client_secret|'
                         r'secret|password|passwd|private_key)"\s*:\s*"([^"]{8,2048})"', re.IGNORECASE)
_BEARER = re.compile(r"\b(?:Authorization:\s*)?Bearer\s+([A-Za-z0-9._~+/=-]{16,})")
# A secret label, what it is for, then the value: quoted (its exact bytes between the quotes) or bare (one word).
_LABELED = re.compile(r"\b((?:wi-?fi\s+)?password|passwd|passphrase|passcode|pin(?:\s+code)?|otp|one[- ]time code|"
                      r"2fa code|verification code|security code|backup codes?|recovery codes?|recovery phrase|"
                      r"seed phrase|api key|secret key|access key|private key|auth token|access token|cvv|cvc)\b"
                      r"((?:\s+(?:for|to|of|on)\s+[\w.@-]{1,40})?)\s*(?:is|was|=|:|->)\s*"
                      r"(?:([\"'`])((?:(?!\3)[^\n]){3,128})\3|[\"'`]?([^\s\"'`,;]{3,128}))", re.IGNORECASE)
# A seed or recovery phrase is its words, all of them: twelve to twenty-four words of three to eight letters.
_PHRASE = re.compile(r"\b(seed phrase|recovery phrase|mnemonic(?: phrase)?)\b\s*(?:is|was|=|:|->)?\s*[\"'`]?"
                     r"((?:[a-z]{3,8}[ \t]+){11,23}[a-z]{3,8})\b", re.IGNORECASE)
# The labels a card's three or four digit security code comes under.
_CODE_LABELS = frozenset({"cvv", "cvc", "security code"})
_CARD = re.compile(r"(?<![A-Za-z0-9_-])((?:\d{4}[ -]){3}\d{1,7}|\d{15,16})(?![A-Za-z0-9_-])")
_CARD_LABEL = re.compile(r"\b(?:card|visa|mastercard|amex|credit|debit)\b", re.IGNORECASE)
_TELEGRAM = re.compile(r"(?<![A-Za-z0-9_-])(?:bot)?\d{8,10}:([A-Za-z0-9_-]{35})(?![A-Za-z0-9_-])")

_PLACEHOLDER = re.compile(r"^(?:\*+|x{3,}|\.{3,}|<[^>]*>|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_]+%|\[[^\]]*\]|"
                          r"(?:your|my|the|an?)[_-].*|.*(?:here|example|placeholder|redacted|changeme|dummy|sample).*|"
                          r"none|null|nil|true|false|yes|no|empty|unset|required|optional|secret|password|token)$",
                          re.IGNORECASE)
_STOPWORDS = frozenset("wrong incorrect expired invalid valid correct required missing changed reset same different "
                       "saved stored set updated needed ready sent there here this that what which your mine".split())
_LABEL_CHARS = re.compile(r"[^A-Za-z0-9 _.@+-]+")
# Identifiers memory needs, never a secret whatever names them: a UUID, a contact id, a phone number, a long run
# of digits (an order or account number).
_IDENTIFIER = re.compile(r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|"
                         r"[a-z]{1,4}-\d{6,}|\+\d{8,15}|\d{13,})$")
# A tracking or booking reference (upper case letters and digits): an identifier, unless a secret's label (a
# password, a code) calls it one.
_TRACKING = re.compile(r"^(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{8,22}$")


@dataclass(frozen=True)
class Hit:
    """One detected value: the rule that found it, its span in the text, the value and the label it is stored
    under (never the value itself)."""
    rule: str
    start: int
    end: int
    value: str
    label: str


def clean_label(text: str) -> str:
    """A label as a reference may carry it: letters, digits and a few separators, at most 60 characters."""
    label = " ".join(_LABEL_CHARS.sub(" ", str(text or "")).split())[:60].strip()
    return label or "secret"


def _trim(value: str) -> str:
    """An unquoted value without what closes the sentence around it: trailing periods, and a closing bracket
    that has no opening one in the value (the end of an aside)."""
    while value and (value[-1] == "." or (value[-1] == ")" and value.count(")") > value.count("("))):
        value = value[:-1]
    return value


def _credential_like(value: str, *, labeled: bool, label: str = "") -> bool:
    value = value.strip()
    if not value or _PLACEHOLDER.match(value) or value.lower() in _STOPWORDS:
        return False
    if _IDENTIFIER.match(value) or (not labeled and _TRACKING.match(value)):
        return False
    classes = sum(bool(re.search(pattern, value)) for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    if labeled:
        # A short PIN or code is a secret when a secret label names it; a lone word is not.
        if label in _CODE_LABELS and value.isdigit() and 3 <= len(value) <= 4:
            return True
        return (value.isdigit() and 4 <= len(value) <= 12) or (len(value) >= 6 and classes >= 2) or len(value) >= 12
    return len(value) >= 8 and (classes >= 3 or (classes >= 2 and len(value) >= 16) or len(value) >= 32)


def _luhn(digits: str) -> bool:
    total, flip = 0, False
    for char in reversed(digits):
        value = int(char) * (2 if flip else 1)
        total += value - 9 if value > 9 else value
        flip = not flip
    return total % 10 == 0


def _card_like(value: str, text: str, start: int) -> bool:
    digits = re.sub(r"\D", "", value)
    brand = ((re.match(r"4", digits) and len(digits) in (13, 16, 19))
             or (re.match(r"5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720)", digits) and len(digits) == 16)
             or (re.match(r"3[47]", digits) and len(digits) == 15)
             or (re.match(r"6(?:011|5)", digits) and len(digits) == 16))
    grouped = not value.isdigit()
    labeled = bool(_CARD_LABEL.search(text[max(0, start - 40):start]))
    return bool(brand) and _luhn(digits) and (grouped or labeled)


def _vendor(value: str) -> str:
    return next((label for pattern, label in _VENDOR_LABELS if pattern.fullmatch(value)), "API key")


def _candidates(text: str, rule: str) -> Iterable[Tuple[int, int, str, str]]:
    """``(start, end, value, label)`` of every match of ``rule``, its value checks applied."""
    if rule == "prefix":
        for match in _PREFIX.finditer(text):
            yield match.start(1), match.end(1), match.group(1), _vendor(match.group(1))
    elif rule in {"jwt", "private_key", "webhook"}:
        pattern, label = {"jwt": (_JWT, "JWT"), "private_key": (_PRIVATE_KEY, "private key"),
                          "webhook": (_WEBHOOK, "webhook URL")}[rule]
        for match in pattern.finditer(text):
            yield match.start(1), match.end(1), match.group(1), label
    elif rule == "url_userinfo":
        for match in _URL_USERINFO.finditer(text):
            if _credential_like(match.group(1), labeled=True):
                yield match.start(1), match.end(1), match.group(1), "URL password"
    elif rule == "url_token":
        for match in _URL_TOKEN.finditer(text):
            if _credential_like(match.group(2), labeled=False):
                yield match.start(2), match.end(2), match.group(2), f"URL {match.group(1).lower()}"
    elif rule == "assign":
        for match in _ASSIGN.finditer(text):
            name = match.group(1)
            if (_SECRET_NAME.search(name) and not _NOT_SECRET_NAME.search(name)
                    and _credential_like(match.group(3), labeled=False)):
                yield match.start(3), match.end(3), match.group(3), name
    elif rule == "json_field":
        for match in _JSON_FIELD.finditer(text):
            if _credential_like(match.group(2), labeled=False):
                yield match.start(2), match.end(2), match.group(2), match.group(1)
    elif rule == "bearer":
        for match in _BEARER.finditer(text):
            if _credential_like(match.group(1), labeled=False):
                yield match.start(1), match.end(1), match.group(1), "bearer token"
    elif rule == "labeled":
        for match in _PHRASE.finditer(text):
            yield match.start(2), match.end(2), match.group(2), match.group(1).lower()
        for match in _LABELED.finditer(text):
            label = (match.group(1) + match.group(2)).lower().replace("wi-fi", "wifi")
            if match.group(4) is not None:
                value, start = match.group(4), match.start(4)   # quoted: exactly the bytes between the quotes
            else:
                # Unquoted: a sentence's closing period, or the bracket closing an aside, is not the value, so the
                # same value ends a sentence and sits mid-sentence under one reference.
                value, start = _trim(match.group(5)), match.start(5)
            # What the label says the value is (a card's security code, whatever card it is for) decides its
            # shape; the label it is stored under keeps what it is for.
            if _credential_like(value, labeled=True, label=match.group(1).lower()):
                yield start, start + len(value), value, label
    elif rule == "telegram":
        for match in _TELEGRAM.finditer(text):
            yield match.start(1), match.end(1), match.group(1), "bot token"
    elif rule == "card":
        for match in _CARD.finditer(text):
            if _card_like(match.group(1), text, match.start(1)):
                yield match.start(1), match.end(1), match.group(1), "card number"


def find_secrets(text: str, *, rules: Optional[Iterable[str]] = None) -> List[Hit]:
    """Every secret the capture detector finds in ``text``, in text order, spans never overlapping. A reference to
    a sealed value and a withheld marker are never found (nor anything inside them)."""
    if not isinstance(text, str) or not text:
        return []
    enabled = tuple(RULES if rules is None else (rule for rule in RULES if rule in set(rules)))
    # A reference is passed over only when its label carries no value: an invented one with a token for a label
    # is read like any other text.
    taken: List[Tuple[int, int]] = [match.span() for match in REFERENCE.finditer(text)
                                    if not any(True for rule in enabled
                                               for _ in _candidates(match.group(0)[len("[secret S-XXXXXXXX: "):-1],
                                                                    rule))]
    start = text.find(WITHHELD)
    while start >= 0:
        taken.append((start, start + len(WITHHELD)))
        start = text.find(WITHHELD, start + 1)
    found: List[Hit] = []
    for rule in enabled:
        for begin, end, value, label in _candidates(text, rule):
            if any(left < end and begin < right for left, right in taken):
                continue
            taken.append((begin, end))
            found.append(Hit(rule, begin, end, value, clean_label(label)))
    return sorted(found, key=lambda hit: hit.start)


def replace_secrets(text: str, replacement: Callable[[Hit], str], *,
                    rules: Optional[Iterable[str]] = None) -> Tuple[str, List[Hit]]:
    """``text`` with each detected value replaced by ``replacement(hit)``, and the hits."""
    hits = find_secrets(text, rules=rules)
    if not hits:
        return text, []
    parts, cursor = [], 0
    for hit in hits:
        parts += [text[cursor:hit.start], replacement(hit)]
        cursor = hit.end
    parts.append(text[cursor:])
    return "".join(parts), hits


def hide_secrets(text: str, *, rules: Optional[Iterable[str]] = None) -> str:
    """``text`` with every detected value and every reference replaced by ``HIDDEN``: what a message to someone
    else may carry of a secret."""
    hidden, _ = replace_secrets(str(text or ""), lambda hit: HIDDEN, rules=rules)
    return REFERENCE.sub(HIDDEN, hidden)


__all__ = ["HIDDEN", "Hit", "REFERENCE", "RULES", "VENDOR_PATTERNS", "WITHHELD", "clean_label", "find_secrets",
           "hide_secrets", "replace_secrets"]
