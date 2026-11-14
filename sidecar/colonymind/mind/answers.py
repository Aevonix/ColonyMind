"""The owner's words as their answer to an ask: a fixed phrase reader (docs/MIND.md, Asks).

Pure: no store, no model, no clock, no I/O. ``bind`` decides which ask a message answers from structure alone (what
it reacts or replies to, and what the chat showed when it arrived); ``read`` decides whether one message of the
owner's says yes, no, later or always plainly; ``plan`` what the two together do; ``states_change`` whether the
owner's own recent words state a change to who may be contacted; ``ask_again`` what the owner asks to be asked about
again ("ask me again about Sam"), which only ever takes a standing allow back.

**Where words decide, and where they never do.** Nothing in ColonyMind is authorized by matching words, with this
module as the one declared exception, and only inside three limits. It reads only the owner's own message (never
the model's text, a tool result or a page); only after something structural has already tied that message to one
ask (``bind``: a reaction or a quoted reply on the ask's own message, or the ask's own message being the last thing
in the owner's chat when the message arrived); and only its fixed tables count. What those tables cannot place is
never guessed:

- ``yes``, ``no``, ``later`` (the ask stays open, no verdict), ``always`` (a yes that also stops the asking, only
  where the ask offers it), and against a numbered list ``all_yes``, ``all_no`` or ``pick`` (numbers, yes or no);
- ``unclear``: an answer word with a hedge ("but", "if", "wait", "maybe" ...), a question in the first sentence
  (anywhere, for a floor or widening ask read ``strict``), a negation within three words of the yes ("not sure"),
  a yes and a no in one message, or words beside the answer the tables cannot place. One plain question back,
  never a guess;
- ``none``: no answer word in the first clause ("Here is what Sam said: yes" is someone else's yes). It is
  conversation, not an answer.

Someone else's words the owner passes on (a quotation, "Sam said ...", a chat line naming its speaker) are blanked
before anything is read (``reactions.own_words``). A typed code is never a reference to the words: "yes K7F" is a
plain yes, and which ask it answers is decided by structure alone (``bind``); a code in the owner's own words only
ever shows its ask (``named``).

The tables follow the approvals plan; a few natural forms are added beyond it ("of course", "perfect", "that
works", "no way", the 1.11 words "deny", "reject" and "decline", and a no before a yes verb such as "don't do it").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import reactions

# -- readings ---------------------------------------------------------------------------------------------------

YES, NO, LATER, ALWAYS = "yes", "no", "later", "always"
ALL_YES, ALL_NO, PICK = "all_yes", "all_no", "pick"
UNCLEAR, NONE = "unclear", "none"
VERDICTS = (YES, NO, LATER, ALWAYS, ALL_YES, ALL_NO, PICK, UNCLEAR, NONE)


@dataclass(frozen=True)
class Reading:
    """What the owner's words say. ``said`` is ``yes`` or ``no`` for an answer with a polarity (a pick, always,
    all), empty otherwise; ``picks`` the list numbers of a pick; ``hedge`` the first hedge found; ``reason`` a
    plain phrase for the model and the settling history. ``answer_word``: the first clause holds a yes or no word;
    ``hard_hedge``: "but", "except", "only if", "unless", "instead" or a question mark appears (a later typed
    check reads these, never the tables)."""
    verdict: str
    said: str = ""
    picks: Tuple[int, ...] = ()
    hedge: str = ""
    reason: str = ""
    answer_word: bool = False
    hard_hedge: bool = False

    @property
    def positive(self) -> bool:
        """Whether this reading would approve something."""
        return self.verdict in {YES, ALWAYS, ALL_YES} or (self.verdict == PICK and self.said == YES)

    def as_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "said": self.said, "picks": list(self.picks), "hedge": self.hedge,
                "reason": self.reason}


# -- the tables (canonical: lowercase, one space between words, typographic apostrophes plain, emoji bare) ---------

YES_PHRASES = frozenset({
    "yes", "y", "yeah", "yea", "yep", "yup", "sure", "ok", "okay", "alright", "all right", "fine", "go ahead",
    "go for it", "do it", "do that", "please do", "yes please", "sounds good", "approve", "approved", "confirm",
    "confirmed", "send it", "ship it", "👍", "✅", "👌", "✔",
    # beyond the plan's table
    "absolutely", "definitely", "of course", "perfect", "great", "sure thing", "sounds great", "that's fine",
    "thats fine", "that works", "works for me", "go on",
})
# Yes only for an ordinary ask: too short to carry a floor or widening change.
WEAK_YES = frozenset({"k", "go"})
_YES_VERBS = ("do it", "do that", "send it", "ship it", "go ahead", "go for it", "approve it", "approve")
NO_PHRASES = frozenset({
    "no", "n", "nope", "nah", "don't", "dont", "do not", "no thanks", "no thank you", "skip", "skip it", "cancel",
    "cancel it", "leave it", "never mind", "nevermind", "👎", "❌", "🚫",
    # beyond the plan's table: the 1.11 no words, and a no in front of a yes verb ("don't do it")
    "deny", "denied", "reject", "rejected", "decline", "declined", "no way", "absolutely not", "definitely not",
    *(f"{no} {verb}" for no in ("don't", "dont", "do not") for verb in _YES_VERBS),
})
LATER_PHRASES = frozenset({"later", "not now", "not yet", "hold off", "ask me later", "remind me later", "tomorrow",
                           "in a bit", "not today", "ask me tomorrow", "remind me tomorrow"})
# "always" on its own is the reply an ask that offers it invites; the longer forms count only after a yes word.
ALWAYS_PHRASES = frozenset({"always", "always allow", "yes always"})
ALWAYS_AFTER_YES = frozenset({
    "don't ask again", "don't ask me again", "dont ask again", "dont ask me again", "do not ask again",
    "do not ask me again", "stop asking", "no need to ask"})
ALL_YES_PHRASES = frozenset({"all", "all of them", "yes to all", "yes all", "both", "both of them", "yes to both",
                             "yes both"})
ALL_NO_PHRASES = frozenset({"none", "none of them", "no to all", "neither", "neither of them", "no to both"})
PAIR_PHRASES = frozenset({"both", "both of them", "yes to both", "yes both", "neither", "neither of them",
                          "no to both"})
HEDGES = frozenset({"but", "except", "only if", "if", "unless", "instead", "wait", "hold on", "maybe", "perhaps",
                    "first", "🤔"})
HARD_HEDGES = frozenset({"but", "except", "only if", "unless", "instead"})
QUESTIONS = frozenset({"what", "why", "which", "how", "tell me more", "more details"})
NEGATIONS = frozenset({"not", "never", "isn't", "wasn't", "won't", "can't", "cannot", "shouldn't", "wouldn't",
                       "didn't", "doesn't", "haven't", "hasn't", "hardly", "nor"})
POLITE = frozenset({"thanks", "thank you", "thanks a lot", "many thanks", "thx", "ty", "cheers", "please", "pls",
                    "plz", "🙏", "🙂", "😊", "😀", "😄", "😁", "☺",
                    "thanks so much", "thank you so much", "thanks very much", "thank you very much", "thanks again",
                    "thank you again", "much appreciated", "appreciated", "appreciate it",
                    # plainly glad emoji: they carry no condition, so beside a yes they change nothing
                    "🎉", "🙌", "💯", "🔥", "❤", "👏", "😍", "🥳", "✨", "💪", "😃", "🤝"})
# Words that change nothing about an answer.
FILLERS = frozenset({"and", "&", "oh", "ah", "then", "just", "now", "right now", "right away", "no problem",
                     "no worries", "no rush", "no hurry"})
# Only against a list: "yes to 2", "no to 1".
LIST_FILLERS = frozenset({"to"})
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10}

# Labels, in the order a phrase found in two tables takes them.
_ALWAYS_AFTER, _ALWAYS, _NO, _LATER, _ALL_YES, _ALL_NO, _YES, _WEAK = (
    "always_after", "always", "no", "later", "all_yes", "all_no", "yes", "weak")
_HEDGE, _QUESTION, _NEG, _POLITE, _FILLER, _LIST_FILLER, _NUMBER, _OTHER = (
    "hedge", "question", "neg", "polite", "filler", "list_filler", "number", "other")
_TABLES: Tuple[Tuple[str, frozenset], ...] = (
    (_ALWAYS_AFTER, ALWAYS_AFTER_YES), (_ALWAYS, ALWAYS_PHRASES), (_NO, NO_PHRASES), (_LATER, LATER_PHRASES),
    (_ALL_YES, ALL_YES_PHRASES), (_ALL_NO, ALL_NO_PHRASES), (_YES, YES_PHRASES), (_WEAK, WEAK_YES),
    (_HEDGE, HEDGES), (_QUESTION, QUESTIONS), (_NEG, NEGATIONS), (_POLITE, POLITE), (_FILLER, FILLERS),
    (_LIST_FILLER, LIST_FILLERS))
_PHRASES: Dict[str, str] = {}
for _label, _table in reversed(_TABLES):
    for _phrase in _table:
        _PHRASES[_phrase] = _label
_LONGEST = max(len(phrase.split()) for phrase in _PHRASES)
_ANSWER_LABELS = frozenset({_ALWAYS_AFTER, _ALWAYS, _NO, _LATER, _ALL_YES, _ALL_NO, _YES, _WEAK})
_YES_LABELS, _NO_LABELS = frozenset({_YES, _WEAK, _ALL_YES}), frozenset({_NO, _ALL_NO})
_NEUTRAL = frozenset({_POLITE, _FILLER})

# -- normalising ----------------------------------------------------------------------------------------------------

_TYPOGRAPHIC = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "′": "'", "`": "'", "“": '"', "”": '"', "„": '"',
                              "«": '"', "»": '"'})
_QUOTE_POINTER = re.compile(r'^\s*\[Replying to[^:\]\n]{0,80}:\s*"', re.IGNORECASE)
_SKIN_TONES = re.compile("[\U0001F3FB-\U0001F3FF]")
_SELECTORS = re.compile("[\uFE0E\uFE0F\u200D]")
_EMOJI = "[\U0001F300-\U0001FAFF\u2600-\u27BF]"
_EMOTICON = re.compile(r"(?<!\S)[:;]-?[)(dDpP](?!\S)")
_MENTION = re.compile(r"^\s*@[\w.\-]{1,64}[\s,:]*")
_LEADING_WORD = re.compile(r"^\s*([^\W\d_][\w'\-]{0,30})\s*[,:]\s+(?=\S)")
_LEADING_ANSWER_COLON = re.compile(r"^(\s*[^\W\d_][\w' ]{0,20}?)\s*:\s*(?=\S)")
_CODE_SHAPE = re.compile(r"(?:[A-HJKMNP-Z2-9]{3}|[A-HJKMNP-Z2-9]{6})", re.IGNORECASE)
_DIGITS_THEN_LETTERS = re.compile(r"\d+[a-z]+", re.IGNORECASE)     # 2nd, 3pm: never a code
_EDGE_PUNCTUATION = ".!,;:~\u2026-\u2013\u2014 \t\n"
_TOKEN = re.compile(rf"[a-z0-9]+(?:'[a-z]+)*|[?!.]+|[,;:\u2014\u2013]|\n|&|{_EMOJI}")


def canonical(phrase: Any) -> str:
    """``phrase`` as the tables hold it: lowercase, typographic quotes plain, emoji bare, one space between
    words, no punctuation at either end."""
    text = unicodedata.normalize("NFC", str(phrase or "")).translate(_TYPOGRAPHIC)
    text = _SELECTORS.sub("", _SKIN_TONES.sub("", text)).lower()
    text = re.sub(rf"({_EMOJI})", r" \1 ", text)
    return " ".join(text.split()).strip(_EDGE_PUNCTUATION)


def _unquote_pointer(text: str) -> str:
    """The owner's own words after the gateway's quote pointer (``[Replying to: "..."]``, which a quoted reply
    puts before them); nothing when the pointer does not end (the whole message is the quotation)."""
    while True:
        match = _QUOTE_POINTER.match(text)
        if match is None:
            return text
        end = text.find('"]', match.end())
        if end < 0:
            return ""
        text = text[end + 2:]


def _is_word(token: str) -> bool:
    """Whether the tables read ``token`` as a word ("yes", "but", "nah", "ten")."""
    return _label(token) != _OTHER or canonical(token) in NUMBER_WORDS


def _code_of(token: str, known: Iterable[str], code: str) -> Optional[str]:
    """What a typed token is: an open ask's code (``known``, or ``code``, the ask the message is read for), ``""``
    for a code no open ask holds, None for a word. Only where it stands as a code: with a digit (no word has
    one), or in capitals as the ask shows it. Their own ask's code also counts in ordinary case, unless the tables
    read it as a word: a code spelled "BUT" or "YES" never erases their hedge or their answer word."""
    upper = token.upper()
    digit = any(char.isdigit() for char in token)
    if upper in known or upper == code:
        if digit or token == upper or (upper == code and not _is_word(token)):
            return upper
        return None
    if digit and _CODE_SHAPE.fullmatch(token) is not None and _DIGITS_THEN_LETTERS.fullmatch(token) is None:
        return ""
    return None


# One private-use mark per distinct open code a message holds: never a word character, so a code set aside reads
# as a gap to every rule that follows, while where it stood (their own words, or someone else's) stays visible.
_MARK_BASE = 0xE100
_MARKS = re.compile("[\ue100-\uf8ff]")


def _marked(text: str, codes: Iterable[str], code: str = "") -> Tuple[str, List[str]]:
    """``text`` with each typed code set aside (``_code_of``): an open ask's code replaced by its mark
    (``found[i]`` is the code of mark ``chr(_MARK_BASE + i)``), a code no open ask holds blanked, words kept."""
    known = {str(item).strip().upper() for item in codes if str(item or "").strip()}
    code = str(code or "").strip().upper()
    found: List[str] = []
    text = _MARKS.sub(" ", text)           # a private-use character they typed never stands for a code

    def mark(match: re.Match) -> str:
        token = match.group(0)
        typed = _code_of(token, known, code)
        if typed is None:
            return token
        if not typed:
            return " "
        if typed not in found:
            found.append(typed)
        return chr(_MARK_BASE + found.index(typed))
    return re.sub(r"(?<![\w'])[A-Za-z0-9]{3,6}(?![\w'])", mark, text), found


def _unmarked(text: str) -> str:
    return _MARKS.sub(" ", text)


def _label(phrase: str) -> str:
    return _PHRASES.get(canonical(phrase), _OTHER)


# Every phrase of more than one word in the tables ("tell me more", "don't ask again"): kept whole while
# ``reactions.own_words`` blanks someone else's words, whose reporting verbs ("tell", "ask") some of them hold.
_PROTECTED = sorted((phrase for phrase in _PHRASES if " " in phrase), key=len, reverse=True)
_PROTECT = re.compile(r"(?<![\w'])(" + "|".join(re.escape(phrase) for phrase in _PROTECTED) + r")(?![\w'])",
                      re.IGNORECASE)
_HELD = re.compile("\ue000(\\d+)\ue001")


def _own(value: str) -> str:
    """``reactions.own_words`` over ``value``, with the tables' own phrases kept whole."""
    kept: List[str] = []

    def hold(match: re.Match) -> str:
        kept.append(match.group(0))
        return f"\ue000{len(kept) - 1}\ue001"
    own = reactions.own_words(_PROTECT.sub(hold, value), strict=True)
    return _HELD.sub(lambda match: kept[int(match.group(1))], own)


def _tidy(value: str) -> str:
    value = _EMOTICON.sub(" ", value)
    match = _LEADING_WORD.match(value)
    if match is not None and _label(match.group(1)) == _OTHER:
        value = value[match.end():]        # a leading name or interjection: "Sam, yes"; never a table word
    value = _SELECTORS.sub("", _SKIN_TONES.sub("", value)).lower()
    value = re.sub(rf"({_EMOJI})", r" \1 ", value)
    value = "\n".join(" ".join(line.split()) for line in value.split("\n"))
    value = re.sub(r"\n{2,}", "\n", value).strip()
    return value.rstrip(_EDGE_PUNCTUATION).lstrip(_EDGE_PUNCTUATION)


def _prepared(text: Any, codes: Iterable[str], code: str = "") -> Tuple[str, str, Tuple[str, ...]]:
    """``(full, own, named)``: the message ready for the tables, the same with someone else's words blanked, and
    the open codes (of ``codes`` and ``code``) their own words name, in the order they first appear."""
    value = unicodedata.normalize("NFC", str(text or "")).translate(_TYPOGRAPHIC)
    value = reactions.strip_prefix(_unquote_pointer(value)).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"' and '"' not in value[1:-1]:
        value = value[1:-1].strip()        # a voice note's transcript, rendered as the whole message in quotes
    value, found = _marked(value, codes, code)
    value = _MENTION.sub("", value)
    value = value.lstrip(_EDGE_PUNCTUATION)
    # "Yes: go ahead" is the owner's answer, not a chat line whose speaker is "Yes".
    value = _LEADING_ANSWER_COLON.sub(lambda m: m.group(1) + ", " if _label(m.group(1)) != _OTHER else m.group(0),
                                      value)
    own = _own(value)
    named = tuple(code for index, code in enumerate(found) if chr(_MARK_BASE + index) in own)
    return _tidy(_unmarked(value)), _tidy(_unmarked(own)), named


def normalise(text: Any, *, codes: Iterable[str] = (), code: str = "") -> str:
    """The owner's own words, ready for the tables: the gateway's prefixes and quote pointer removed, a voice
    note's surrounding quotes removed, typed codes removed, someone else's words blanked, a leading name or mention
    set aside (never a word the tables read), emoji bare, lowercase, no punctuation at the ends but a question
    mark."""
    return _prepared(text, codes, code)[1]


# -- tokens, clauses and sentences ------------------------------------------------------------------------------------

@dataclass
class _Token:
    text: str
    label: str
    at: int                    # the position of its first word in the message
    value: int = 0             # a number's value


@dataclass
class _Clause:
    tokens: List[_Token]
    joined: bool               # a conjunction introduced it ("yes, and remind me tomorrow"): a request of its own

    def labels(self) -> set:
        return {token.label for token in self.tokens}

    def says(self) -> bool:
        return any(token.label not in _NEUTRAL for token in self.tokens)


@dataclass
class _Sentence:
    clauses: List[_Clause]
    asked: bool                # it holds a question mark or a question phrase

    def tokens(self) -> List[_Token]:
        return [token for clause in self.clauses for token in clause.tokens]


def _label_words(words: List[Tuple[str, int]], *, listed: bool) -> List[_Token]:
    tokens: List[_Token] = []
    i = 0
    while i < len(words):
        for size in range(min(_LONGEST, len(words) - i), 0, -1):
            label = _PHRASES.get(" ".join(word for word, _ in words[i:i + size]))
            if label is not None:
                tokens.append(_Token(" ".join(word for word, _ in words[i:i + size]),
                                     _OTHER if label == _LIST_FILLER and not listed else label, words[i][1]))
                i += size
                break
        else:
            word, at = words[i]
            if word.isdigit() or word in NUMBER_WORDS:
                tokens.append(_Token(word, _NUMBER if listed else _OTHER, at,
                                     int(word) if word.isdigit() else NUMBER_WORDS[word]))
            else:
                tokens.append(_Token(word, _OTHER, at))
            i += 1
    return tokens


def _sentences(text: str, *, listed: bool) -> List[_Sentence]:
    """``text`` as sentences of clauses of labelled phrases. A comma, a colon, a dash or a conjunction ends a
    clause (never the comma or conjunction between two list numbers: "1, 3", "1 and 3")."""
    sentences: List[_Sentence] = []
    clauses: List[_Clause] = []
    words: List[Tuple[str, int]] = []
    position = 0

    def close_clause() -> None:
        nonlocal words
        tokens = _label_words(words, listed=listed)
        words = []
        part: List[_Token] = []
        joined = False
        for index, token in enumerate(tokens):
            between = (0 < index < len(tokens) - 1 and tokens[index - 1].label == _NUMBER
                       and tokens[index + 1].label == _NUMBER)
            if token.text in {"and", "&"} and not between:
                if part:
                    clauses.append(_Clause(part, joined))
                part, joined = [], True
                continue
            part.append(token)
        if part:
            clauses.append(_Clause(part, joined))

    def close_sentence(asked: bool) -> None:
        nonlocal clauses
        close_clause()
        if clauses or asked:
            asked = asked or any(token.label == _QUESTION for clause in clauses for token in clause.tokens)
            sentences.append(_Sentence(clauses, asked))
        clauses = []

    raw = _TOKEN.findall(text)
    for index, token in enumerate(raw):
        if token == "\n" or re.fullmatch(r"[?!.]+", token):
            close_sentence("?" in token)
        elif token in {",", ";", ":", "\u2014", "\u2013"}:
            neighbours = (raw[index - 1] if index else "", raw[index + 1] if index + 1 < len(raw) else "")
            if listed and token == "," and all(item.isdigit() or item in NUMBER_WORDS for item in neighbours):
                continue                       # "1, 3" is one pick
            close_clause()
        else:
            words.append((token, position))
            position += 1
    close_sentence(False)
    return sentences


def _first_saying(sentences: List[_Sentence]) -> Tuple[Optional[_Clause], Optional[_Sentence]]:
    """The first clause that says something ("Thanks, go ahead": the thanks says nothing about the ask)."""
    for sentence in sentences:
        for clause in sentence.clauses:
            if clause.says():
                return clause, sentence
    return None, None


def _unplaced(sentences: List[_Sentence], placed: set, *, after: Optional[int]) -> List[str]:
    """Words the tables cannot place after the word at position ``after`` (the answer), read in everything they sent;
    with ``after`` None (strict), in the whole message. A separate request (a clause a conjunction introduces:
    "yes, and remind me tomorrow") and a question in a later sentence ("Yes. How long will it take?") reach the
    agent as usual and are not read, except strict, where nothing rides with the yes."""
    stray: List[str] = []
    past = False                   # in a sentence after the answer's own
    for sentence in sentences:
        holds = after is not None and any(token.at == after for token in sentence.tokens())
        for clause in sentence.clauses:
            own_clause = after is not None and any(token.at == after for token in clause.tokens)
            if after is not None and not own_clause and (clause.joined or (past and sentence.asked)):
                continue
            stray += [token.text for token in clause.tokens
                      if (after is None or token.at > after) and token.label not in placed]
        past = past or holds
    return stray


# -- reading ----------------------------------------------------------------------------------------------------------

def read(text: Any, *, offers_always: bool = False, list_size: int = 0, strict: bool = False,
         codes: Iterable[str] = (), code: str = "") -> Reading:
    """What one message of the owner's says about an ask. ``strict`` for a floor or widening ask (``k`` and ``go``
    alone do not count, a question anywhere voids a yes, and so does anything beside it the tables cannot place,
    a separate request included); ``offers_always`` where the ask offers "always"; ``list_size`` when it answers a
    numbered list; ``codes`` the open asks' codes and ``code`` the code of the ask it is read for, set aside
    where they stand as codes (``_code_of``), never where an ordinary word spells one.

    The answer word must stand first in their own words (someone else's words blanked); everything that voids a yes
    (a hedge, a negation, a no, a question) is looked for in everything they sent, the blanked words included. So are
    words the tables cannot place after the answer word, in any clause, quoted or not: extra words pass only as a
    separate request introduced by a conjunction, or a question in a later sentence."""
    size = max(0, int(list_size or 0))
    listed = size > 0
    full, own, _ = _prepared(text, codes, code)
    if not own:
        return Reading(NONE, reason="nothing of their own to read" if full else "nothing to read")
    sentences = _sentences(full, listed=listed)
    tokens = [token for sentence in sentences for token in sentence.tokens()]
    hard = "?" in full or any(token.label == _HEDGE and token.text in HARD_HEDGES for token in tokens)
    hedge = next((token.text for token in tokens if token.label == _HEDGE), "")
    own_sentences = _sentences(own, listed=listed)
    first, first_sentence = _first_saying(own_sentences)
    if first is None or first_sentence is None:
        return Reading(NONE, reason="no answer word", hard_hedge=hard)
    labels = first.labels()
    answer_word = bool(labels & (_ANSWER_LABELS - {_LATER}))

    def unclear(reason: str) -> Reading:
        return Reading(UNCLEAR, hedge=hedge, reason=reason, answer_word=answer_word, hard_hedge=hard)

    if not labels & (_ANSWER_LABELS | {_NUMBER}):
        later_words = [token for sentence in own_sentences for token in sentence.tokens()
                       if token.at > first.tokens[-1].at]
        if labels <= {_HEDGE, _QUESTION, _NEG} and any(token.label in _ANSWER_LABELS for token in later_words):
            answer_word = True
            return unclear(f"'{first.tokens[0].text}' before the answer")       # "Wait, yes", "What? ok"
        return Reading(NONE, reason="no answer word first", hard_hedge=hard)
    # A question up to the first sentence that says something voids it ("K7F? sure": they asked about it first).
    leading = []
    for sentence in sentences:
        leading.append(sentence)
        if any(clause.says() for clause in sentence.clauses):
            break
    if first_sentence.asked or any(sentence.asked for sentence in leading):
        return unclear("a question in the first sentence")
    placed = {_HEDGE, _QUESTION, _NEG, _POLITE, _FILLER, *_ANSWER_LABELS} | ({_NUMBER, _LIST_FILLER} if listed
                                                                                 else set())
    stray = [token.text for token in first.tokens if token.label not in placed]
    if stray:
        if hedge and not labels & _NO_LABELS:
            return unclear(f"a condition or hesitation ('{hedge}')")
        return unclear(f"words the tables cannot place ('{' '.join(stray)}')")
    if any(token.label in _YES_LABELS for token in tokens) and any(token.label in _NO_LABELS for token in tokens):
        return unclear("a yes and a no in one message")
    said = NO if labels & _NO_LABELS else YES
    if said == YES and strict and _WEAK in labels and not labels & {_YES, _ALWAYS, _ALL_YES}:
        weak = next(token.text for token in first.tokens if token.label == _WEAK)
        return unclear(f"'{weak}' alone does not answer a change of this weight")

    numbers = [token.value for token in first.tokens if token.label == _NUMBER]
    pairs = [token.text for token in first.tokens if token.text in PAIR_PHRASES]
    if numbers:
        if any(number < 1 or number > size for number in numbers):
            return unclear(f"a number the list does not hold (it has {size})")
        reading = Reading(PICK, said=said, picks=tuple(sorted(set(numbers))), reason=f"numbers from the list ({said})",
                          answer_word=answer_word, hard_hedge=hard)
    elif labels & {_ALL_YES, _ALL_NO}:
        if not listed:
            return unclear("'all' or 'none' answers only a list")
        if pairs and size != 2:
            return unclear(f"'{pairs[0]}' answers a list of two (this one has {size})")
        verdict = ALL_NO if _ALL_NO in labels else ALL_YES
        reading = Reading(verdict, said=NO if verdict == ALL_NO else YES, reason="the whole list",
                          answer_word=answer_word, hard_hedge=hard)
    elif said == NO:
        return Reading(NO, said=NO, reason="a plain no", answer_word=answer_word, hard_hedge=hard)
    else:
        # A later word beside the yes: in its first clause, or a clause of later words alone that is not a request
        # of its own ("yes, tomorrow" is later; "yes, and remind me tomorrow" is a yes and a request).
        beside = _LATER in labels or any(
            not clause.joined and clause.says() and clause.labels() <= _NEUTRAL | {_LATER}
            for sentence in sentences for clause in sentence.clauses)
        if beside:
            return Reading(LATER, reason="later: the ask stays open", answer_word=answer_word, hard_hedge=hard)
        yes_seen, always = False, ""
        for token in tokens:
            if token.label in {_YES, _WEAK}:
                yes_seen = True
            elif token.label == _ALWAYS and not always:
                always = token.text
            elif token.label == _ALWAYS_AFTER and not always:
                if not yes_seen:
                    return unclear(f"'{token.text}' counts only after a yes")
                always = token.text
        if always and not offers_always:
            return unclear(f"'always' is not offered for this ask ('{always}')")
        reading = Reading(ALWAYS if always else YES, said=YES, reason="yes, and stop asking" if always else "a plain yes",
                          answer_word=answer_word, hard_hedge=hard)

    if reading.positive:
        if hedge:
            return unclear(f"a condition or hesitation ('{hedge}')")
        yes_at = [token.at for token in tokens if token.label in {_YES, _WEAK, _ALWAYS}]
        negation = next((token.text for token in tokens if token.label == _NEG
                         and any(abs(token.at - at) <= 3 for at in yes_at)), "")
        if negation:
            return unclear(f"a negation beside the yes ('{negation}')")
        if strict and ("?" in full or any(token.label == _QUESTION for token in tokens)):
            return unclear("a question, which a change of this weight does not take with a yes")
        # A comma, a full stop, a new line or quotation marks hide no condition: what follows the answer word is
        # read in everything they sent (someone else's words they pass on included). Read strict, the whole message
        # is, since nothing rides with the yes. The first answer word anywhere is the earliest theirs can be.
        answer_at = next((token.at for token in tokens if token.label in _ANSWER_LABELS | {_NUMBER}), None)
        beside = _unplaced(sentences, placed, after=None if strict else (answer_at if answer_at is not None else -1))
        if beside:
            return unclear(f"words beside the answer the tables cannot place ('{' '.join(beside)}')")
    return reading


# -- which ask a message answers: structure, never words (docs/MIND.md, Asks) -------------------------------------

# The paths an answer takes to its ask; the settling row records it (``answer.path``) and the asks report counts it.
REACTION, REPLY, FOLLOW_UP, LIST = "reaction", "reply", "follow_up", "list"
PATHS = (REACTION, REPLY, FOLLOW_UP, LIST)
# Not a binding: an owner channel's explicit answer naming the ask's immutable id and one of its offered choices
# (``Mind.answer_explicit``, ``POST /v1/mind/permit-asks/{id}/answer``), recorded only after the credential's owner
# grant on that channel's platform and the owner's identity were checked. ``bind`` never produces it.
EXPLICIT = "explicit"
# The plan's defaults, each an assumption the owner may change (APPROVALS-UX 6.2, 6.5, 6.6):
SETTLE_SECONDS = 3.0        # an ask delivered less than this before the message arrived is never what it answered
ORDER_SECONDS = 5.0         # two sends this close whose order the plugin did not fix bind nothing
FOLLOW_UP_HOURS = 12.0      # a floor or widening ask binds by follow-up only this long after its own message
LIST_HOURS = 24.0           # a list's numbers mean that list only this long
CLARIFY_HOURS = 24.0        # at most one plain question per ask in this long
UNDO_MINUTES = 10.0         # an approval can be undone in its chat this long after it
YES_REACTIONS = frozenset({"👍", "✅", "👌", "✔"})
NO_REACTIONS = frozenset({"👎", "❌", "🚫"})


@dataclass(frozen=True)
class Shown:
    """One message that put asks in front of the owner, as the sidecar recorded its delivery: the platform and
    the chat it was delivered in, its delivery id there (unique only within that chat on some platforms), when it
    was delivered, and the asks it showed in its order (one: that ask's own message; more: a list)."""
    platform: str
    chat: str
    message_id: str
    at: datetime
    asks: Tuple[str, ...]


@dataclass(frozen=True)
class Seen:
    """One thing the plugin saw reach the owner's chat (its ledger): when, and the asks it showed (a send the
    sidecar recorded; none for a model reply, an interim message, a model send, a cron delivery or anything else,
    each of which breaks the follow-up). ``ordered``: the plugin fixed its order among its own sends."""
    at: datetime
    asks: Tuple[str, ...] = ()
    ordered: bool = False


@dataclass(frozen=True)
class Evidence:
    """What structure a message carries, never its words: its platform and chat, when it arrived, what it replies
    or reacts to (a delivery id in that same chat), and the chat as the plugin saw it at that arrival (``ledger``;
    None when read without it, at ingest: then only a reaction or a quoted reply binds)."""
    platform: str
    arrived_at: datetime
    chat: str = ""
    reply_to: str = ""
    reaction: str = ""
    reaction_removed: bool = False
    ledger: Optional[Tuple[Seen, ...]] = None


@dataclass(frozen=True)
class Binding:
    """The asks a message answers, in the order the owner saw them (more than one: a list), and how it reached
    them; empty, with the reason, when nothing binds."""
    asks: Tuple[str, ...] = ()
    path: str = ""
    at: Optional[datetime] = None
    reason: str = ""

    @property
    def listed(self) -> bool:
        return len(self.asks) > 1

    def __bool__(self) -> bool:
        return bool(self.asks)


def _hours(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def _delivered(shown: Iterable[Shown], evidence: Evidence) -> Optional[Shown]:
    """The message ``evidence`` replies or reacts to, among ``shown``: the same delivery id, on the same platform,
    in the same chat, all three known. A delivery id alone names nothing: another chat may use it too."""
    return next((item for item in shown if item.message_id == evidence.reply_to and item.platform
                 and item.platform == evidence.platform and item.chat and item.chat == evidence.chat), None)


def bind(evidence: Evidence, shown: Iterable[Shown], *, weighty: Iterable[str] = ()) -> Binding:
    """Which ask (or list) one message answers, by structure alone; the first rule that applies decides.

    1. A reaction whose target is an ask's own message (its delivery id, platform and chat) binds to that ask; a
       reaction on anything else (a list included) is ignored.
    2. A quoted reply to an ask's own message binds to that ask, to a list binds to that list (for
       ``LIST_HOURS``). A quoted reply to anything else binds nothing: no later rule applies to it.
    3. The follow-up: the message binds to the asks of the last thing its chat showed when it arrived
       (``evidence.ledger``), when that was an ask's own message or a list, delivered at least
       ``SETTLE_SECONDS`` before the arrival, with no unordered send within ``ORDER_SECONDS`` of it; a floor or
       widening ask (``weighty``) only within ``FOLLOW_UP_HOURS``, a list only within ``LIST_HOURS``. Without the
       ledger it never binds.

    A message delivered after the arrival is never what it answered. Nothing else binds: words that name an ask
    only show it (``named``, the caller's), and a typed code is no reference."""
    weighty = set(weighty)
    arrived = evidence.arrived_at
    shown = [item for item in shown if item.at <= arrived and item.message_id]
    if evidence.reaction or evidence.reaction_removed:
        target = _delivered(shown, evidence)
        if target is None or len(target.asks) != 1:
            return Binding(reason="a reaction binds only on an ask's own message")
        return Binding(target.asks, REACTION, target.at, "a reaction on the ask's own message")
    if evidence.reply_to:
        target = _delivered(shown, evidence)
        if target is not None:
            if len(target.asks) > 1:
                if _hours(arrived, target.at) > LIST_HOURS:
                    return Binding(reason="a list's numbers mean it for a day only")
                return Binding(target.asks, LIST, target.at, "a quoted reply to a list")
            return Binding(target.asks, REPLY, target.at, "a quoted reply to the ask's own message")
        # The owner quoted something else (the model's message, a closed ask's, their own): that is what they
        # answered, whatever the chat showed last.
        return Binding(reason="a quoted reply to a message that shows no open ask binds nothing")
    if evidence.ledger is None:
        return Binding(reason="read without the chat's ledger: only a reaction or a quoted reply binds")
    before = [item for item in evidence.ledger if item.at <= arrived]
    if not before:
        return Binding(reason="nothing reached the chat before the message")
    last = max(before, key=lambda item: item.at)
    if not last.asks:
        return Binding(reason="the last thing the chat showed was not an ask")
    if (arrived - last.at).total_seconds() < SETTLE_SECONDS:
        return Binding(reason="the ask reached the chat moments before the message: not what it answered")
    if any(item is not last and abs((item.at - last.at).total_seconds()) < ORDER_SECONDS
           and not (item.ordered and last.ordered) for item in evidence.ledger):
        return Binding(reason="two messages reached the chat together in no known order")
    if len(last.asks) > 1:
        if _hours(arrived, last.at) > LIST_HOURS:
            return Binding(reason="a list's numbers mean it for a day only")
        return Binding(last.asks, LIST, last.at, "the list was the last thing the chat showed")
    if last.asks[0] in weighty and _hours(arrived, last.at) > FOLLOW_UP_HOURS:
        return Binding(reason=f"a change of this weight binds by follow-up only within {FOLLOW_UP_HOURS:g} hours")
    return Binding(last.asks, FOLLOW_UP, last.at, "the ask was the last thing the chat showed")


def reaction_reading(emoji: Any) -> Reading:
    """A reaction as an answer: 👍 ✅ 👌 yes, 👎 ❌ 🚫 no, anything else nothing."""
    value = canonical(emoji)
    if value in YES_REACTIONS:
        return Reading(YES, said=YES, reason=f"a {value} on the ask's own message", answer_word=True)
    if value in NO_REACTIONS:
        return Reading(NO, said=NO, reason=f"a {value} on the ask's own message", answer_word=True)
    return Reading(NONE, reason="a reaction that is no yes and no no")


@dataclass(frozen=True)
class Plan:
    """What one bound message does: the asks it approves (``always``: of those, the one whose "always" also stops
    the asking), declines or keeps open for later, the ones it puts in front of the owner again instead of approving
    (a weighty ask named in a list's yes), the one it asks a plain question about (``clarify``) and the list it asks
    "which one?" about (``which``)."""
    yes: Tuple[str, ...] = ()
    always: Tuple[str, ...] = ()
    no: Tuple[str, ...] = ()
    later: Tuple[str, ...] = ()
    show: Tuple[str, ...] = ()
    clarify: Tuple[str, ...] = ()
    which: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return any((self.yes, self.no, self.later, self.show, self.clarify, self.which))


def plan(reading: Reading, binding: Binding, *, open_asks: Iterable[str], weighty: Iterable[str] = ()) -> Plan:
    """What ``reading`` does to the asks ``binding`` found, of those still open. One ask: yes (and "always", which
    only an ask that offered it reads), no, later, one plain question when unclear, nothing when no answer word. A
    list (read with its size): the numbers, all or none it names, later for all, and "which one?" for a bare yes or
    no or an unclear answer. A floor or widening ask (``weighty``) never takes a list's yes: it is shown on its own
    instead; a list's no declines it."""
    open_set, weighty = set(open_asks), set(weighty)
    asks = tuple(ask for ask in binding.asks if ask in open_set)
    if not asks or reading.verdict == NONE:
        return Plan()
    if not binding.listed:
        return {YES: Plan(yes=asks), ALWAYS: Plan(yes=asks, always=asks), NO: Plan(no=asks),
                LATER: Plan(later=asks)}.get(reading.verdict, Plan(clarify=asks))
    if reading.verdict == LATER:
        return Plan(later=asks)
    if reading.verdict == ALL_NO or (reading.verdict == PICK and reading.said == NO):
        picked = asks if reading.verdict == ALL_NO else _picked(binding, reading, open_set)
        return Plan(no=picked)
    if reading.verdict in {ALL_YES, PICK}:
        picked = asks if reading.verdict == ALL_YES else _picked(binding, reading, open_set)
        return Plan(yes=tuple(ask for ask in picked if ask not in weighty),
                    show=tuple(ask for ask in picked if ask in weighty))
    return Plan(which=asks) if len(asks) > 1 else Plan(clarify=asks)


def _picked(binding: Binding, reading: Reading, open_set: set) -> Tuple[str, ...]:
    return tuple(binding.asks[number - 1] for number in reading.picks
                 if 0 < number <= len(binding.asks) and binding.asks[number - 1] in open_set)


_GENERIC_NAMES = frozenset({"you", "them", "a contact", "the owner", "owner", "me"})


def named(text: Any, names: Mapping[str, Iterable[str]], *, codes: Mapping[str, str] | None = None
          ) -> Tuple[str, ...]:
    """The asks the owner's own words name (someone else's words blanked): by one of its ``names`` (a contact's
    name, a file or folder, a payee; three characters or more, whole words, any case), or by an open code
    (``codes`` maps it to its ask) standing as a code in their own words. Naming an ask only ever shows it."""
    codes = {str(code).upper(): ask for code, ask in (codes or {}).items() if str(code or "").strip()}
    _, own, typed = _prepared(text, list(codes))
    found: Dict[str, None] = {codes[code]: None for code in typed if code in codes}
    for ask, values in names.items():
        for value in values or ():
            name = " ".join(str(value or "").lower().split())
            if len(name) < 3 or name in _GENERIC_NAMES:
                continue
            if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", own):
                found.setdefault(ask, None)
                break
    return tuple(found)


UNDO_PHRASES = frozenset({
    "undo", "undo that", "undo it", "undo this", "cancel that", "cancel this", "wait no", "i didn't mean that",
    "i did not mean that", "didn't mean that", "i didnt mean that", "that's not what i meant", "not what i meant",
    "take that back", "scratch that"})
_UNDO_LEADS = frozenset({"no", "oh", "ah", "oops", "sorry"})
# The owner saying a yes (or a no) read from their answer was never theirs: "I didn't say yes". Past the undo's
# minutes it takes back only a reading a model made of their words (``Mind._undo_answer``).
_NEVER_SAID = ("i didn't say", "i did not say", "i didnt say", "didn't say", "didnt say", "i never said")
REPUDIATIONS = {said: frozenset(f"{lead} {said}{tail}" for lead in _NEVER_SAID for tail in ("", " to that", " to this"))
                for said in ("yes", "no")}
_REPUDIATION = re.compile(r"(?<![\w'])(" + "|".join(re.escape(phrase) for phrase in sorted(
    set().union(*REPUDIATIONS.values()), key=len, reverse=True)) + r")(?![\w'])", re.IGNORECASE)


def repudiated(text: Any) -> str:
    """``yes`` when the owner's whole message says they never said yes ("I didn't say yes", "no, I never said
    yes"), ``no`` when it says they never said no, else "". Read as ``is_undo`` reads: polite words and a leading
    "no", "oh", "wait" aside, words about anything more are not it, a question asks, and only the owner's own words
    count: a message with a quotation or someone else's words in it is not their statement. The phrase is held whole
    while someone else's words are looked for, since its "say" also reports speech."""
    kept: List[str] = []

    def hold(match: re.Match) -> str:
        kept.append(match.group(0).lower())
        return f"\ue000{len(kept) - 1}\ue001"
    value = unicodedata.normalize("NFC", str(text or "")).translate(_TYPOGRAPHIC)
    held = _REPUDIATION.sub(hold, reactions.strip_prefix(_unquote_pointer(value)).strip())
    if not kept or reactions.own_words(held, strict=True).split() != held.split():
        return ""
    own = _HELD.sub(lambda match: kept[int(match.group(1))], held).lower()
    if "?" in own or "\u00bf" in own:
        return ""
    words = [word for word in re.findall(r"[a-z']+", own) if word not in {"please", "pls", "thanks", "now"}]
    for said, phrases in REPUDIATIONS.items():
        if " ".join(words) in phrases or (len(words) > 1 and words[0] in _UNDO_LEADS | {"wait"}
                                          and " ".join(words[1:]) in phrases):
            return said
    return ""


def is_undo(text: Any) -> bool:
    """Whether the owner's whole message asks to undo what they just approved ("undo", "cancel that", "wait,
    no", "I didn't mean that"), polite words aside; words about anything more ("undo the last edit") are not.
    Read as the answer reader reads: only the owner's own words count (a quoted line, a quotation or a line
    passed on from someone else is not their undo), and a question ("undo?") asks, it does not undo."""
    own = normalise(text)
    if "?" in own or "\u00bf" in own:
        return False
    words = [word for word in re.findall(r"[a-z']+", own) if word not in {"please", "pls", "thanks", "now"}]
    joined = " ".join(words)
    if joined in UNDO_PHRASES:
        return True
    return len(words) > 1 and words[0] in _UNDO_LEADS and " ".join(words[1:]) in UNDO_PHRASES


# -- "ask me again about <thing>": the owner taking a standing allow back --------------------------------------------

# "ask me again about", "ask me first before", "always ask me before", "start asking again about", "go back to asking
# me before", "check with me again before": the words that bring the asking back, and the thing after them.
_AGAIN_LEAD = (r"(?:\b(?:always|start|go back to)\s+ask(?:ing)?\s+(?:me\s+)?(?:again\s+)?(?:first\s+)?"
               r"|\bask(?:ing)?\s+(?:me\s+)?(?:again|first)\s+(?:first\s+)?"
               r"|\bcheck(?:ing)?\s+with\s+me\s+(?:again|first)\s+(?:first\s+)?)"
               r"(?:about|before|for|when|on)\b")
_ASK_AGAIN = re.compile(_AGAIN_LEAD + r"\s+(?P<thing>\S.*)$")
# The words are held whole while someone else's words are blanked: "ask" alone reads as a verb reporting speech.
_AGAIN_HELD = re.compile(_AGAIN_LEAD, re.IGNORECASE)
# A negation before those words says the opposite ("don't ask me again about Sam", "no need to ask me first").
_AGAIN_NEGATION = re.compile(r"\b(?:never|don't|dont|do not|does not|doesn't|didn't|did not|stop|stopped|not|no longer|"
                             r"no need|needn't|can't|cannot|shouldn't|won't|wouldn't)\b")
# A question asks to be asked again only as a request ("could you ask me again before ...?").
_REQUEST = re.compile(r"^(?:please\s+)?(?:can|could|would|will)\s+(?:you|u)\b|^please\b")
_AGAIN_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")


def ask_again(text: Any) -> Optional[str]:
    """The thing the owner's own message asks to be asked about again ("ask me again about Sam", "please ask me first
    before you use terminal", "could you check with me again before paying acme?"), lowercase, or None. Someone
    else's words the owner passes on are blanked first; a negation before the words ("don't ask me again about
    ...", "no need to ask me first"), a question that is not a request, and words with no thing after them name
    nothing. What it names only ever takes a standing allow back (``allows.named_by``): it never widens."""
    kept: List[str] = []

    def hold(match: re.Match) -> str:
        kept.append(match.group(0))
        return f"\ue000{len(kept) - 1}\ue001"
    value = unicodedata.normalize("NFC", str(text or "")).translate(_TYPOGRAPHIC)
    own = _HELD.sub(lambda match: kept[int(match.group(1))],
                    reactions.own_words(_AGAIN_HELD.sub(hold, value), strict=True))
    for found in _AGAIN_SENTENCE.finditer(own.lower()):
        sentence = " ".join(found.group(0).split())
        body = sentence.rstrip(".!?").strip()
        match = _ASK_AGAIN.search(body)
        if match is None or _AGAIN_NEGATION.search(body[:match.start()]):
            continue
        if sentence.endswith("?") and not _REQUEST.search(body):
            continue
        thing = match.group("thing").strip(" ,;:")
        if thing:
            return thing
    return None


# -- a change the owner states in their own words -----------------------------------------------------------------------

@dataclass(frozen=True)
class Statement:
    """Whether the owner's own words state a change, and why (or why not)."""
    stated: bool
    reason: str

    def __bool__(self) -> bool:
        return self.stated


_PERMISSION = re.compile(
    r"\b(?:any\s*time|whenever|without asking|without checking with me|(?:you|u) (?:can|may) "
    r"(?:message|text|call|email|contact|write to|reach out to|reach))\b")
_STATE_NEGATION = re.compile(r"\b(?:never|don't|dont|do not|stop|stopped|not|no longer|can't|cannot|shouldn't|"
                             r"mustn't|won't|nobody|no one)\b")
_EVERY = re.compile(r"\bevery\s+(?:(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|other)\s+)?"
                    r"(minute|hour|day|week|fortnight|month)s?\b")
_UNIT_MINUTES = {"minute": 1, "hour": 60, "day": 1440, "week": 10080, "fortnight": 20160, "month": 43200}
_SENTENCE = re.compile(r"[^.!?\n]+")
_PHONE = re.compile(r"\+?[\d\s().-]{7,}")


def _every_minutes(sentence: str) -> Optional[int]:
    match = _EVERY.search(sentence)
    if match is None:
        return None
    count, unit = match.group(1), match.group(2)
    if count in (None, "a", "an", "one"):
        number = 1
    elif count == "other":
        number = 2
    else:
        number = int(count) if count.isdigit() else NUMBER_WORDS[count]
    return number * _UNIT_MINUTES[unit]


def _names_in(sentence: str, names: Sequence[str], shared: Iterable[str]) -> List[str]:
    """The person's names and handles the sentence carries, longest first; a name another contact shares, or one
    inside a longer name already found, does not count."""
    shared_set = {" ".join(str(name).lower().split()) for name in shared}
    found: List[str] = []
    digits = re.sub(r"\D", "", sentence)
    spans: List[Tuple[int, int]] = []
    for name in sorted({str(name) for name in names if str(name or "").strip()}, key=len, reverse=True):
        key = " ".join(name.lower().split())
        if _PHONE.fullmatch(key):
            # A phone number is found by its digits, however the owner spaced them.
            if re.sub(r"\D", "", key) in digits and key not in shared_set:
                found.append(name)
            continue
        for match in re.finditer(rf"(?<![\w@.+-]){re.escape(key)}(?![\w@-])", sentence):
            if any(start <= match.start() and match.end() <= end for start, end in spans):
                continue
            spans.append((match.start(), match.end()))
            if key not in shared_set:
                found.append(name)
    return found


def states_change(texts: Sequence[Any], change: Mapping[str, Any], names: Sequence[str], *,
                  shared: Iterable[str] = ()) -> Statement:
    """Whether one of the owner's own recent messages (``texts``: the caller passes at most three, from the same
    chat, in the last ten minutes) states exactly ``change`` for this person: a sentence that names the person
    (``names``: its names and handles; one in ``shared`` belongs to another contact too and does not count) and
    says the change in the tables' words. ``set_permission`` to ``auto``: "any time", "whenever", "without asking",
    "you can message, text or call"; ``set_cadence``: "every <interval>" with exactly the call's minutes. A
    negation ("never", "don't", "stop", "not") states the narrowing it says, never the widening; a merge is never
    stated (it always confirms). Someone else's words they pass on do not count."""
    operation = str(change.get("operation") or "")
    if operation == "set_permission" and change.get("value") != "auto":
        return Statement(False, "only a raise to auto is read from their words")
    if operation not in {"set_permission", "set_cadence"}:
        return Statement(False, f"a {operation or 'change'} is always confirmed")
    minutes = change.get("minutes")
    if operation == "set_cadence" and not (isinstance(minutes, int) and not isinstance(minutes, bool) and minutes > 0):
        return Statement(False, "no cadence to compare")
    why = "their words do not name the person and the change"
    for text in list(texts or [])[:3]:
        own = reactions.own_words(unicodedata.normalize("NFC", str(text or "")).translate(_TYPOGRAPHIC), strict=True)
        for sentence in _SENTENCE.findall(own.lower()):
            if not _names_in(sentence, names, shared):
                continue
            said = (_PERMISSION.search(sentence) is not None if operation == "set_permission"
                    else _every_minutes(sentence) == minutes)
            if not said:
                continue
            if _STATE_NEGATION.search(sentence):
                why = "a negation: their words state a narrowing, never this widening"
                continue
            return Statement(True, "their own words name the person and the change")
    return Statement(False, why)


__all__ = ["ALL_NO", "ALL_YES", "ALWAYS", "ALWAYS_AFTER_YES", "ALWAYS_PHRASES", "Binding", "CLARIFY_HOURS", "Evidence",
           "EXPLICIT", "FOLLOW_UP", "FOLLOW_UP_HOURS", "HEDGES", "LATER", "LATER_PHRASES", "LIST", "LIST_HOURS", "NO",
           "NONE", "NO_PHRASES", "NO_REACTIONS", "ORDER_SECONDS", "PATHS", "PICK", "Plan", "REACTION", "REPLY",
           "Reading", "SETTLE_SECONDS", "Seen", "Shown", "Statement", "UNCLEAR", "UNDO_MINUTES", "UNDO_PHRASES",
           "VERDICTS", "WEAK_YES", "YES", "YES_PHRASES", "YES_REACTIONS", "ask_again", "bind", "canonical", "is_undo",
           "named", "normalise", "plan", "reaction_reading", "read", "states_change"]
