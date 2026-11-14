"""The mind's decision points, as data.

The five fast points keep the questions, labels and states they were measured with (docs/DECISIONS.md): the answer
of a decision model at each is a reading of words that can only do what the point's existing path could have done
with them. ``owner_verdict``'s fast tier ships live (it was at least as accurate as its existing path with the
fallback); the other four ship in shadow, asked where they would have been asked and never applied. Four of them also
have a reader tier (an LLM already served, read as option probabilities) in shadow, which may settle what its fast
tier may. ``outreach_reply`` has none yet: a topic-linked reply the phrases read nothing in may be a new request
rather than an answer, and its five reactions have no label for that, so a reader would be forced into one of them
(a stop, a mute). Its reader is registered with an out-of-scope label, measured together, never before.

The four typed calls on the mind's own model (``owner_instruction``, ``report_substance``, ``question_answer``,
``mind_wish_scope``) are points whose deliberate tier is that call, moved unchanged (the existing path): the
caller's own reader decides the clear cases first and asks only where it leaves the case open, and with no answer
the reader's conservative fallback stands. The first three gain a reader tier in shadow, which may settle only what
cannot act for the owner: never ``resume`` (it lifts the owner's own stop), never ``first``, ``second`` or
``neither`` (they retract a stored statement). The fast tier is left out of them: none is a literal yes/no, and a
decision model's accuracy on them is unmeasured.

``owner_send`` is the owner gate's decision: its prior is the rules tier (final at or above the gate's confidence,
except for a night-time send or a critical alert on a topic already sent, which are high stakes), the second
opinion is its deliberate tier, and a fast and a reader tier are registered in shadow on the gate's question. They
may only hold: no model tier sends where the prior did not.

``claim_relation`` is the night's relation read (``mind/consolidate.py``): does the newer of two statements one
person made replace the older, or add to it? Its deliberate tier is the night's own typed call, charged to the
night as before; a fast and a reader tier may settle only ``adds`` (the pair is then recorded compatible and never
asked). A ``replaces`` supersedes a stored claim, so only the night's own call may record one. The question is the
one the bake-off measures, so a record fitted there applies here.

The two pre-screens, ``opinion_none`` (``mind/opinions.py``) and ``appraisal_empty`` (``self_model/appraisals.py``),
ask before a whole model call whether the turn gives it anything to do; their effect is to route (a settled "no"
skips the call). They ship in shadow in this release, so the call always runs; a live screen skips only at a
calibrated "no" of at least ``PRESCREEN_FLOOR``, and a turn the opt-out phrases flag is never asked (the appraisal
is the second net for an opt-out).

``ask_answer`` reads the owner's own words as their answer to one ask (``mind/tick.py``, the answer route), with the
question the bake-off measures. The phrase tables read the answer first; the point is asked only where they read it
unclear with a yes or no word in its first clause and no hard hedge, for one ordinary ask (never a floor ask or one
that widens, never a list). A model tier may settle ``decline`` (a no) or the two readings that settle nothing
(``conditional_or_question``, ``unrelated``), never ``approve``: a model's yes, read at its bar, is withheld
(``Tier.withholds``: the fit calibrates it like the rest) and the caller only asks the owner back. With no answer the tables' reading stands (``conditional_or_question``: one plain
question). No deliberate tier: nothing generative is asked on the owner's turn.

Those earlier model tiers ship in shadow unless noted above: asked where eligible, logged, never applied until
promoted on evidence.

``percept_attend`` routes an admitted observation to ``ignore``, ``note``, ``look`` or ``tell``. Structural rules
decide first. Its reader and deliberate tiers are live, within the engine's calibration and escalation gates;
without a measured reader answer the fallback is a note. The caller still applies work authority and the owner
message gate, and habituation never settles a concern.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .registry import Point, RulesAnswer, Tier, register

FAST = ("outreach_reply", "opt_out", "no_reminders", "interest_settled", "owner_verdict")
#: Per point, the model tiers that read only its short states: the point's longer states are over the tier's limit
#: and skip it (never cut). ``owner_send`` with the house facts in full is longer than a decision server reads.
SHORT_STATES = {"owner_send": ("fast",)}
#: The least calibrated "no" at which a pre-screen may skip a whole model call (my choice): a wrong skip loses an
#: opinion or an appraisal silently, so a screen goes live only where it almost never misses one.
PRESCREEN_FLOOR = 0.99


def _nothing(fields: Mapping[str, Any], rules: Optional[RulesAnswer]) -> Optional[str]:
    return None


# -- the five fast points: the 1.11 questions, labels and states, unchanged ----------------------------------------

OUTREACH_REPLY = Point(
    name="outreach_reply", type="choice", effect="read",
    instructions="How does the owner answer the assistant's unprompted message?",
    labels={"engaged": "welcomes or acknowledges it and asks for nothing more",
            "dig_deeper": "asks for more on it: details, a follow-up or research",
            "not_interested": "does not want this topic, or found it useless",
            "not_now": "busy right now; it can wait until later",
            "stop": "wants no more unprompted messages at all"},
    template='The assistant messaged the owner unprompted about {topic}. The owner answered: "{text}"',
    # The measured 1.11 fast tier only: no reader until the question can say a reply is about something else.
    tiers=(Tier("fast", may_settle=("engaged", "dig_deeper", "not_interested", "not_now", "stop")),),
    fallback=_nothing, turn_path=True,
    caller_rules="the reactions phrase tables: asked only for a topic-linked reply they read nothing in")

OPT_OUT = Point(
    name="opt_out", type="yes_no", effect="filter", literal=True,
    instructions="Does the sender ask to receive no more messages from the assistant at all?",
    labels={"yes": "they want no more messages at all",
            "no": "anything else, including when, where or how to send something"},
    template='Message to the assistant: "{text}"',
    tiers=(Tier("fast", may_settle=("yes", "no")), Tier("reader", may_settle=("yes", "no"))), fallback=_nothing,
    caller_rules="the opt-out phrases: asked only for a contact's words they miss, never the owner's")

NO_REMINDERS = Point(
    name="no_reminders", type="yes_no", effect="filter", literal=True,
    instructions="Does the person ask not to be reminded about this item?",
    labels={"yes": "they want no reminders about it", "no": "reminders are wanted, or not mentioned"},
    template='Item: {item}\nThe person said: "{text}"',
    tiers=(Tier("fast", may_settle=("yes", "no")), Tier("reader", may_settle=("yes", "no"))), fallback=_nothing,
    caller_rules="the capture call: asked only for a new item it gave a reminder, never for a message to a contact")

INTEREST_SETTLED = Point(
    name="interest_settled", type="yes_no", effect="read", literal=True,
    instructions="Does the owner say their question about this topic is answered or no longer wanted?",
    labels={"yes": "answered, satisfied or dropped", "no": "still open, or not about this topic"},
    template='Topic: {topic}\nThe owner said: "{text}"',
    tiers=(Tier("fast", may_settle=("yes", "no")), Tier("reader", may_settle=("yes", "no"))), fallback=_nothing,
    caller_rules="the capture call: asked only for an open interest the turn names that the call left open")

OWNER_VERDICT = Point(
    name="owner_verdict", type="yes_no", effect="filter", literal=True,
    instructions="Does the owner judge or correct the assistant's earlier reply, rather than ask for something new?",
    labels={"yes": "a verdict or a correction on that reply", "no": "a new request or other talk"},
    template='The assistant replied: "{reply}"\nThe owner then said: "{text}"',
    tiers=(Tier("fast", may_settle=("yes", "no"), state="live"), Tier("reader", may_settle=("yes", "no"))),
    fallback=_nothing,
    caller_rules="the night's lesson checks: asked only for a verdict they admitted; a no withdraws it")


# -- the typed calls on the mind's model, moved: the deliberate tier is the existing path ---------------------------

OWNER_INSTRUCTION = Point(
    name="owner_instruction", type="choice", effect="read",
    instructions="Does the owner, in their own voice, tell the assistant what to do about its unprompted check-ins?",
    labels={"stop": "their own instruction to stop the check-ins until they say otherwise",
            "pause_today": "their own instruction to hold the check-ins for today",
            "resume": "their own permission to start the check-ins again",
            "none": "no such instruction of the owner's own"},
    template='Message: "{message}"',
    tiers=(Tier("reader", may_settle=("stop", "pause_today", "none")),
           Tier("deliberate", may_settle=("stop", "pause_today", "resume", "none"), on_path=True)),
    fallback=_nothing, turn_path=True,
    caller_rules="the reactions reader: asked only where a cue sits beside someone else's words")

REPORT_SUBSTANCE = Point(
    name="report_substance", type="choice", effect="route",
    instructions="Does this research report tell the owner something substantive about its topic, or only that "
                 "nothing was found or changed?",
    labels={"finding": "a fact, a change or a milestone the topic itself reached",
            "empty": "only that nothing was found or nothing changed"},
    template='Topic: {topic}\nReport: "{report}"',
    tiers=(Tier("reader", may_settle=("finding", "empty")),
           Tier("deliberate", may_settle=("finding", "empty"), on_path=True)), fallback=_nothing,
    caller_rules="the outreach reader: asked only where a null marker sits beside other words")

QUESTION_ANSWER = Point(
    name="question_answer", type="choice", effect="read",
    instructions="What does the owner's reply answer about the two statements the assistant asked them to choose "
                 "between?",
    labels={"first": "keep only the first statement", "second": "keep only the second statement",
            "both": "both hold", "neither": "neither holds any more",
            "unclear": "about this question, but it does not say which",
            "none": "about something else"},
    template='Question: "{question}"\nFirst: "{first}"\nSecond: "{second}"\nEarlier: "{earlier}"\n'
             'Reply: "{reply}"',
    tiers=(Tier("reader", may_settle=("both", "unclear", "none")),
           Tier("deliberate", may_settle=("first", "second", "both", "neither", "unclear", "none"), on_path=True)),
    fallback=_nothing,
    caller_rules="the question ledger: asked once per owner turn after the question was delivered")

MIND_WISH_SCOPE = Point(
    name="mind_wish_scope", type="choice", effect="filter",
    instructions="Does the text of this wish that code did not write ask for more than a yes to the wish allows?",
    labels={"within": "it only names or describes the capability and why it is missing",
            "beyond": "it asks for more than a yes allows",
            "unsure": "the model cannot tell"},
    template='Capability: {capability}\nTopic: {topic}\nWords: "{words}"',
    tiers=(Tier("deliberate", may_settle=("within", "beyond", "unsure"), on_path=True),), fallback=_nothing,
    caller_rules="the wish's shape checks: asked once for a valid wish, inside the night's budget")


CLAIM_RELATION = Point(
    name="claim_relation", type="choice", effect="filter",
    instructions="How does the newer statement stand to the older one the same person made?",
    labels={"adds": "both can hold together: a detail, a refinement, another item, or another fact",
            "replaces": "they cannot both hold now: a changed fact, a correction, one choice instead of another"},
    template='About: {about}\nOlder ({older_on}): "{older}"\nNewer ({newer_on}): "{newer}"',
    tiers=(Tier("fast", may_settle=("adds",)), Tier("reader", may_settle=("adds",)),
           Tier("deliberate", may_settle=("replaces", "adds"), on_path=True)),
    fallback=_nothing,
    caller_rules="the night's contradictions stage: asked for each undecided pair it takes, at most eight a night")



# -- the pre-screens: skip a whole model call when the turn gives it nothing to do -------------------------------

OPINION_NONE = Point(
    name="opinion_none", type="yes_no", effect="route",
    instructions="Does this turn give the assistant anything to form or change a view on?",
    labels={"yes": "it states, argues or reports something to hold a view on, or bears on a view",
            "no": "nothing to judge: small talk, thanks, logistics or a plain request"},
    template='The person said: "{said}"\nThe assistant replied: "{reply}"',
    tiers=(Tier("reader", may_settle=("yes", "no")),), fallback=_nothing, floors={"no": PRESCREEN_FLOOR},
    caller_rules="the opinion pass: asked only for a turn with admitted evidence, never for a finding, a request to "
                 "reconsider or a turn the opt-out phrases flag")

APPRAISAL_EMPTY = Point(
    name="appraisal_empty", type="yes_no", effect="route",
    instructions="Does this message carry anything to appraise: a feeling, an outcome, someone's mood, a wish to "
                 "hear no more, or an incident?",
    labels={"yes": "a feeling, an outcome, someone's mood, a wish to hear no more, or an incident",
            "no": "none of these: small talk, logistics, a plain request or a fact"},
    template='Message: "{text}"',
    tiers=(Tier("reader", may_settle=("yes", "no")),), fallback=_nothing, floors={"no": PRESCREEN_FLOOR},
    caller_rules="the appraisal: asked only for a first appraisal of a turn (never revision work: a retained view, "
                 "an owner's correction or an open incident), never for a turn the opt-out phrases flag")


def prescreen_eligible(texts: Any) -> bool:
    """Whether a pre-screen may be asked about a turn's words: never where the opt-out phrases flag one of them."""
    from colonymind.contacts.optout import detects_opt_out
    return not any(detects_opt_out(text) for text in texts if isinstance(text, str))


# -- the owner gate's decision -------------------------------------------------------------------------------------

def _owner_send_prior(fields: Mapping[str, Any]) -> Optional[RulesAnswer]:
    """The gate's deterministic prior as the rules tier: its choice speaks at or above the gate's confidence."""
    from colonymind.mind import gate
    message = gate.Message(type="", topic="", text="", urgency=str(fields.get("urgency") or "normal"),
                           source=str(fields.get("source") or "mind"))
    probabilities = gate.prior(message, night=bool(fields.get("night")))
    label = max(gate.CHOICES, key=lambda name: (float(probabilities.get(name, 0.0)), -gate.CHOICES.index(name)))
    bar = fields.get("confidence", 0.75)
    return RulesAnswer(label, probabilities, speaks=float(probabilities[label]) >= float(bar))


def _owner_send_high_stakes(fields: Mapping[str, Any], label: Optional[str]) -> bool:
    """A night-time send, or a critical alert on a topic the owner was already sent: asked whatever the confidence."""
    return bool(fields.get("raised")) or (bool(fields.get("night")) and label == "send_now")


def _owner_send_fallback(fields: Mapping[str, Any], rules: Optional[RulesAnswer]) -> Optional[str]:
    """Nothing settled it: the prior's choice, except that a night-time wake needs a yes it did not get (a
    critical alert still goes, unconfirmed)."""
    if rules is None or rules.label is None:
        return None
    if fields.get("night") and rules.label == "send_now" and fields.get("urgency") != "critical":
        return "digest"
    return rules.label


OWNER_SEND = Point(
    name="owner_send", type="choice", effect="filter",
    instructions="Should the assistant message its owner about this now, list it in tomorrow's digest, or drop it?",
    labels={"send_now": "worth interrupting the owner for now",
            "digest": "worth knowing, but it can wait for the daily digest",
            "drop": "already said, or not worth the owner's attention"},
    template='{facts}\nA {urgency} {kind} about {topic}: "{text}"',
    tiers=(Tier("fast", may_settle=("digest", "drop")), Tier("reader", may_settle=("digest", "drop")),
           Tier("deliberate", may_settle=("send_now", "digest", "drop"), on_path=True)),
    rules=_owner_send_prior, fallback=_owner_send_fallback, high_stakes=_owner_send_high_stakes,
    # Code floors (chosen values, not measured): a model drops only at 0.95 and holds for the digest at 0.80.
    floors={"drop": 0.95, "digest": 0.80})


# -- the owner's answer to an ask ------------------------------------------------------------------------------------

#: What a model tier may settle at ``ask_answer``: a no, or a reading that settles nothing. Never a yes.
ASK_LABELS = ("decline", "conditional_or_question", "unrelated")


def _ask_unclear(fields: Mapping[str, Any], rules: Optional[RulesAnswer]) -> Optional[str]:
    """Nothing settled it: the tables' own reading of an unclear answer, which asks one plain question."""
    return "conditional_or_question"


ASK_ANSWER = Point(
    name="ask_answer", type="choice", effect="read",
    instructions="What do the owner's own words answer to the assistant's ask?",
    labels={"approve": "the owner's own plain yes to this ask, with no condition",
            "decline": "the owner's own no to this ask",
            "conditional_or_question": "a yes or no with a condition or a change, not now, a question back, or "
                                       "hesitation",
            "unrelated": "not the owner's own answer to this ask: other talk, or someone else's words"},
    template='The assistant asked: "{ask}"\nThe owner wrote: "{words}"',
    # ``approve`` is read at its calibrated bar and withheld, never settled: the caller asks the owner back.
    tiers=(Tier("fast", may_settle=ASK_LABELS, withholds=("approve",)),
           Tier("reader", may_settle=ASK_LABELS, withholds=("approve",))),
    fallback=_ask_unclear, turn_path=True,
    # The code floors (chosen values, not measured): a model's yes asks the owner back only at 0.97, its no
    # settles at 0.90.
    floors={"approve": 0.97, "decline": 0.90},
    caller_rules="the answer tables: asked only for an unclear answer to one ordinary ask, with a yes or no word "
                 "first and no hard hedge")


# -- percept attention --------------------------------------------------------------------------------------------

def _percept_attend_rules(fields: Mapping[str, Any]) -> Optional[RulesAnswer]:
    """The observation reader supplies these flags and the operator's configured route, never a sender's route.

    Missing or inactive observations cannot raise work. Habituation and a missing original leave a note, which
    cannot close an existing concern. A configured route selects among the same bounded paths as model tiers;
    the caller still applies ordinary work authority and the owner gate.
    """
    if fields.get("enabled") is not True or fields.get("active") is not True:
        return RulesAnswer("ignore")
    if fields.get("habituated") is True or fields.get("has_original") is not True:
        return RulesAnswer("note")
    route = fields.get("rule_route")
    if isinstance(route, str) and route in {"ignore", "note", "look", "tell"}:
        return RulesAnswer(route)
    return None


def _percept_attend_fallback(fields: Mapping[str, Any], rules: Optional[RulesAnswer]) -> str:
    """Unsettled percepts retain a note without a task or message."""
    return "note"


PERCEPT_ATTEND = Point(
    name="percept_attend", type="choice", effect="route",
    instructions="How should the assistant attend to this unverified observation? Choose only a route; work "
                 "authority and the owner message gate still apply.",
    labels={"ignore": "take no further action on the retained surprise",
            "note": "retain a concern note without work or a message",
            "look": "ask a worker to reopen the original and inspect it",
            "tell": "offer the observation to the owner through the owner message gate"},
    template='Source: {source}\nSubject: {subject}\nFeature: {feature}\nSeverity: {severity}\n'
             'Unverified description: "{summary}"',
    # A reader needs the engine's matching calibration record. A new deliberate call may run only after a
    # calibrated reader abstains, within the ordinary escalation budget. No unmeasured fast choice is installed.
    tiers=(Tier("reader", may_settle=("ignore", "note", "look", "tell"), state="live"),
           Tier("deliberate", may_settle=("ignore", "note", "look", "tell"), state="live")),
    rules=_percept_attend_rules, fallback=_percept_attend_fallback,
    caller_rules="validated observation activity and original availability, habituation and the operator's route")


POINTS = (OUTREACH_REPLY, OPT_OUT, NO_REMINDERS, INTEREST_SETTLED, OWNER_VERDICT, OWNER_INSTRUCTION,
          REPORT_SUBSTANCE, QUESTION_ANSWER, MIND_WISH_SCOPE, CLAIM_RELATION, OPINION_NONE, APPRAISAL_EMPTY,
          OWNER_SEND, ASK_ANSWER, PERCEPT_ATTEND)


def register_all(registry=None) -> None:
    for point in POINTS:
        register(point, registry=registry)


__all__ = ["APPRAISAL_EMPTY", "ASK_ANSWER", "ASK_LABELS", "CLAIM_RELATION", "FAST", "INTEREST_SETTLED",
           "MIND_WISH_SCOPE", "NO_REMINDERS", "OPINION_NONE", "OPT_OUT", "OUTREACH_REPLY", "OWNER_INSTRUCTION",
           "OWNER_SEND", "OWNER_VERDICT", "PERCEPT_ATTEND", "POINTS", "PRESCREEN_FLOOR", "QUESTION_ANSWER", "REPORT_SUBSTANCE",
           "SHORT_STATES", "prescreen_eligible", "register_all"]
