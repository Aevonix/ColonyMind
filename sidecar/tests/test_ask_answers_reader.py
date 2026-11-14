"""The owner's words as an answer to an ask (``colonymind.mind.answers``): a fixed phrase reader.

The reader runs only on the owner's own message, and only once something structural has tied that message to an
ask; what it decides is whether the words say yes, no, later or always plainly. A hedge, a question, a yes beside a
no, or words its tables cannot place make it ``unclear`` (one plain question back, never a guess); a message with
no answer word first is ``none`` (conversation). A typed code is never a reference: ``yes K7F`` is a plain yes.
"""

from __future__ import annotations

import pytest

from colonymind.mind import answers
from colonymind.mind.answers import (ALL_NO, ALL_YES, ALWAYS, LATER, NO, NONE, PICK, UNCLEAR, YES, normalise, read,
                                    states_change)


def typed(text, *codes):
    """The open codes (``codes``) the owner's own words name, in their order: what shows an ask (``named``)."""
    return answers.named(text, {}, codes={code: code for code in codes})


def verdict(text, **kwargs):
    return read(text, **kwargs).verdict


# -- yes ------------------------------------------------------------------------------------------------------

YES_FORMS = ["yes", "y", "yeah", "yea", "yep", "yup", "sure", "ok", "okay", "k", "alright", "fine", "go", "go ahead",
             "go for it", "do it", "do that", "please do", "yes please", "sounds good", "approve", "approved",
             "confirm", "confirmed", "send it", "ship it", "👍", "✅", "👌", "✔️"]


@pytest.mark.parametrize("text", YES_FORMS)
def test_every_yes_form_alone_is_a_yes(text):
    assert verdict(text) == YES


@pytest.mark.parametrize("text", [f"{form} {tail}" for form in ("yes", "sure", "go ahead", "ok", "👍", "do it")
                                  for tail in ("thanks", "please")])
def test_a_yes_followed_by_thanks_or_please_is_a_yes(text):
    assert verdict(text) == YES


@pytest.mark.parametrize("text", ["YES", "Yes.", "Yes!!", "Sure!", "OK.", "Go ahead.", "  yep  ", "Yes, please.",
                                  "Sounds good!", "Ok, thanks!", "Yes yes", "Yeah sure", "Thanks, go ahead"])
def test_capitals_spacing_and_punctuation_do_not_matter(text):
    assert verdict(text) == YES


@pytest.mark.parametrize("text", ["👍🏽", "👍🏿", "👌🏻", "✔", "✅ thanks", "👍thanks", "👍 🙏"])
def test_emoji_are_read_with_their_skin_tones_and_selectors_set_aside(text):
    assert verdict(text) == YES


@pytest.mark.parametrize("text", ["Yes K7F!!! Thanks so much", "yes K7F, thank you so much", "Yes K7F. Much appreciated.",
                                  "yes K7F thanks very much", "Yes K7F. Thanks again!", "ok, thank you very much"])
def test_a_longer_thanks_beside_a_yes_changes_nothing(text):
    assert verdict(text, codes=("K7F",)) == YES
    assert verdict(text, codes=("K7F",), strict=True) == YES


@pytest.mark.parametrize("text", ["yes 🎉", "Yes K7F! 🎉", "Go ahead 🙌", "sure ❤️", "ok 💯", "Yes. 🔥🔥"])
def test_a_plainly_glad_emoji_beside_a_yes_changes_nothing(text):
    """Glad emoji carry no condition: beside the yes, or on their own after it, they change nothing about it."""
    assert verdict(text, codes=("K7F",)) == YES
    assert verdict(text, codes=("K7F",), strict=True) == YES


def test_a_glad_emoji_alone_is_no_answer_and_a_doubtful_one_still_voids_a_yes():
    assert verdict("🎉") == NONE
    assert verdict("yes 🤔") == UNCLEAR


@pytest.mark.parametrize("text", ["k", "go", "K", "Go."])
def test_k_and_go_alone_answer_only_an_ordinary_ask(text):
    assert verdict(text) == YES
    assert verdict(text, strict=True) == UNCLEAR


@pytest.mark.parametrize("text", ["yes", "go ahead", "okay", "👍", "confirmed"])
def test_a_clean_yes_answers_a_floor_or_widening_ask_too(text):
    assert verdict(text, strict=True) == YES


# -- no -------------------------------------------------------------------------------------------------------

NO_FORMS = ["no", "n", "nope", "nah", "don't", "do not", "no thanks", "skip", "skip it", "cancel", "leave it",
            "never mind", "👎", "❌", "🚫"]


@pytest.mark.parametrize("text", NO_FORMS)
def test_every_no_form_alone_is_a_no(text):
    assert verdict(text) == NO
    assert verdict(text, strict=True) == NO


@pytest.mark.parametrize("text", ["No.", "NOPE!", "No thanks.", "no, thank you", "👎🏿", "Don't.", "don't do it",
                                  "Do not send it", "Nah, leave it", "no, never mind"])
def test_no_forms_with_capitals_emoji_and_punctuation(text):
    assert verdict(text) == NO


def test_a_no_with_a_hedge_is_still_a_no():
    """Hedges void a yes: a no only narrows, so a condition beside it changes nothing."""
    assert verdict("no, but ask me again next week") == NO
    assert verdict("no, maybe another time") == NO


# -- later ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["later", "not now", "not yet", "hold off", "ask me later", "remind me later",
                                  "tomorrow", "in a bit", "Not now, thanks.", "Later!", "maybe later"])
def test_later_keeps_the_ask_open(text):
    reading = read(text)
    assert reading.verdict == LATER and not reading.positive


@pytest.mark.parametrize("text", ["yes, later", "sure, tomorrow", "yes tomorrow", "ok not now", "yes. later."])
def test_a_later_word_beside_a_yes_reads_as_later(text):
    assert verdict(text) == LATER


@pytest.mark.parametrize("text", ["no, later", "nope, not now", "no tomorrow"])
def test_a_later_word_beside_a_no_reads_as_no(text):
    assert verdict(text) == NO


def test_a_later_word_inside_a_separate_request_does_not_touch_the_yes():
    assert verdict("yes, and remind me tomorrow") == YES
    assert verdict("Yes and remind me about the invoice tomorrow") == YES


# -- hedges, negations and a yes beside a no ----------------------------------------------------------------------

@pytest.mark.parametrize("text", ["yes but", "yes, but not the payment", "Yes but only the first one",
                                  "yes if you must", "yes, if it is cheap", "yes except the second", "ok wait",
                                  "Wait, yes", "yes, hold on", "maybe yes", "yes unless it costs money",
                                  "sure, instead send it tomorrow morning", "yes, check the price first",
                                  "yes, only if it is free", "Yes. But ask Sam first.", "perhaps, go ahead"])
def test_a_hedge_anywhere_voids_a_yes(text):
    reading = read(text)
    assert reading.verdict == UNCLEAR and reading.hedge, text


@pytest.mark.parametrize("text", ["not sure", "yes, don't", "sure, not really", "I'm not sure, ok",
                                  "yes and no", "yes no", "no, yes", "Yes. Don't send it though."])
def test_a_negation_near_the_yes_or_a_no_beside_it_is_unclear(text):
    assert verdict(text) == UNCLEAR


# -- questions ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["yes?", "Yes??", "ok? ", "sure, tell me more", "ok, more details",
                                  "yes, why?", "go ahead? what does it cost", "Yes, how long will it take?",
                                  "K7F? sure", "What? ok"])
def test_a_question_in_the_first_sentence_voids_it(text):
    assert verdict(text, codes=("K7F",)) == UNCLEAR


def test_a_hedge_is_the_reason_given_even_beside_words_the_tables_cannot_place():
    reading = read("yes K7F but not the payment", codes=("K7F",))
    assert reading.verdict == UNCLEAR and reading.hedge == "but" and "'but'" in reading.reason


def test_a_question_after_a_clean_first_sentence_voids_only_a_strict_yes():
    text = "Yes. How long will it take?"
    assert verdict(text) == YES
    assert verdict(text, strict=True) == UNCLEAR
    assert verdict("Go ahead. What time is it there?", strict=True) == UNCLEAR
    assert verdict("No. Why did you even ask?", strict=True) == NO


@pytest.mark.parametrize("text", ["why?", "what is this", "which one", "how much?"])
def test_a_question_with_no_answer_word_is_conversation(text):
    assert verdict(text) == NONE


# -- always ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["always", "Always.", "yes always", "always allow", "yes, don't ask again",
                                  "yes, don't ask me again", "yes, stop asking", "yes, no need to ask",
                                  "Yes. Don't ask me again."])
def test_always_is_read_where_the_ask_offers_it(text):
    assert verdict(text, offers_always=True) == ALWAYS


@pytest.mark.parametrize("text", ["always", "yes always", "yes, don't ask again"])
def test_always_is_unclear_where_the_ask_does_not_offer_it(text):
    reading = read(text)
    assert reading.verdict == UNCLEAR and "always" in reading.reason


@pytest.mark.parametrize("text", ["don't ask again", "stop asking", "no need to ask"])
def test_the_long_always_phrases_count_only_after_a_yes(text):
    assert verdict(text, offers_always=True) == UNCLEAR


def test_a_hedge_voids_an_always_and_a_no_stays_a_no():
    assert verdict("yes always, but not on weekends", offers_always=True) == UNCLEAR
    assert verdict("no, always ask me first", offers_always=True) == NO


# -- lists ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, size, expected", [
    ("all", 3, ALL_YES), ("all of them", 3, ALL_YES), ("yes to all", 3, ALL_YES), ("both", 2, ALL_YES),
    ("Both, thanks", 2, ALL_YES), ("all", 1, ALL_YES),
    ("none", 3, ALL_NO), ("no to all", 3, ALL_NO), ("neither", 2, ALL_NO), ("none of them", 2, ALL_NO),
])
def test_all_and_none_answer_a_whole_list(text, size, expected):
    assert verdict(text, list_size=size) == expected


@pytest.mark.parametrize("text, size, picks, said", [
    ("1", 2, (1,), "yes"), ("2", 3, (2,), "yes"), ("1 and 3", 3, (1, 3), "yes"), ("1, 3", 3, (1, 3), "yes"),
    ("yes 2", 3, (2,), "yes"), ("no 1", 2, (1,), "no"), ("3 and 1", 3, (1, 3), "yes"), ("one and two", 2, (1, 2), "yes"),
    ("2 please", 2, (2,), "yes"), ("No to 2", 2, (2,), "no"),
])
def test_numbers_pick_from_a_list(text, size, picks, said):
    reading = read(text, list_size=size)
    assert (reading.verdict, reading.picks, reading.said) == (PICK, picks, said)


@pytest.mark.parametrize("text, size", [("4", 3), ("0", 3), ("both", 3), ("neither", 3), ("1 but not 2", 2),
                                        ("yes 2 no 1", 2)])
def test_a_list_answer_the_list_cannot_hold_is_unclear(text, size):
    assert verdict(text, list_size=size) == UNCLEAR


@pytest.mark.parametrize("text", ["all", "both", "yes 2", "1 and 3", "none"])
def test_list_words_without_a_list_do_not_answer(text):
    assert verdict(text) in {UNCLEAR, NONE}
    assert not read(text).positive


def test_a_bare_yes_against_a_list_is_a_yes_for_the_binding_to_place():
    assert verdict("yes", list_size=3) == YES
    assert verdict("no", list_size=2) == NO


# -- what is not their answer -------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Here is what Sam said: yes, go ahead", "Sam said yes, go ahead", "She wrote: yes", "Robin told me: go ahead",
    "The vendor's reply was \"yes, go ahead\" so tell them", "",
    "I pasted this from the thread. The team agreed yes to the plan and we move on Monday.",
    "What does K7F do? Do not approve it", "hm", "thanks", "no problem", "",
])
def test_words_that_are_not_his_answer_read_as_none_or_unclear_never_yes(text):
    reading = read(text, codes=("K7F",))
    assert reading.verdict in {NONE, UNCLEAR} and not reading.positive, text


@pytest.mark.parametrize("text", ["Here is what Sam said: yes, go ahead", "She wrote: yes", "hm", "thanks", "",
                                  "I pasted this from the thread. The team agreed yes to the plan."])
def test_no_answer_word_first_is_none(text):
    assert verdict(text) == NONE


def test_a_yes_followed_by_a_separate_request_is_a_yes():
    assert verdict("yes, and remind me tomorrow") == YES
    assert verdict("Sure. Also, what's the weather like?") == YES
    assert verdict("Go ahead, and send me the receipt") == YES
    assert verdict("Yes. And ping me when it is done.") == YES


@pytest.mark.parametrize("text", ["yes K7F, only for today", "yes, for today only", "Yes K7F. Only for today.",
                                  "sure, just this once", "yes, send it to Robin", "ok K7F\nonly the small part",
                                  "Yes, please, only the small one", "go ahead, the cheaper one"])
def test_words_after_the_first_clause_that_the_tables_cannot_place_void_a_yes(text):
    """A comma, a full stop or a new line does not hide a condition: every clause after the answer is read too,
    and words the tables cannot place void the yes there as in the first clause. Extra words pass only as a
    separate request (a clause a conjunction introduces) or a later question, which reach the agent as usual."""
    for strict in (False, True):
        reading = read(text, codes=("K7F",), strict=strict)
        assert reading.verdict == UNCLEAR and not reading.positive, (text, strict)
        assert "cannot place" in reading.reason, (text, strict)


@pytest.mark.parametrize("text", ['yes K7F "only for today"', 'yes K7F, "only for today"', "yes K7F\n> only the small one",
                                  "ok K7F. Sam says only for today.", "Sure K7F, Robin said just the first draft"])
def test_a_condition_he_quotes_or_passes_on_after_the_yes_voids_it(text):
    """Words after their answer are read in everything they sent: a condition in quotation marks, a blockquote or a
    line they pass on from someone else is still a condition on their yes."""
    reading = read(text, codes=("K7F",))
    assert reading.verdict == UNCLEAR and not reading.positive, text


@pytest.mark.parametrize("text", ["yes K7F\n> Can I send the recap to Sam?", "Yes K7F. Also, what's the weather like?",
                                  'Yes K7F, and "the recap" is in the shared folder'])
def test_a_quoted_question_or_a_separate_request_after_the_yes_leaves_it(text):
    assert verdict(text, codes=("K7F",)) == YES


@pytest.mark.parametrize("text", ["yes, and remind me about the invoice", "Go ahead, and send me the receipt",
                                  "yes K7F, Sam says only for today", 'yes K7F\n> only the small part'])
def test_a_floor_or_widening_ask_takes_a_clean_yes_and_nothing_beside_it(text):
    """Read strict, nothing rides with the yes: no separate request, and no one else's words either."""
    reading = read(text, codes=("K7F",), strict=True)
    assert reading.verdict == UNCLEAR and not reading.positive, text


def test_a_no_with_words_beside_it_is_still_a_no():
    """A no only narrows: words beside it the tables cannot place change nothing about it."""
    assert verdict("no K7F, I changed my mind", codes=("K7F",)) == NO
    assert verdict("No. Sam will handle it himself.", strict=True) == NO


def test_none_is_returned_for_nothing_at_all():
    for text in (None, "", "   ", "\n\n", "..."):
        assert verdict(text) == NONE


# -- codes, gateway prefixes and quotes ------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["yes K7F", "Yes, K7F.", "YES k7f!", "K7F yes", "yes K7F thanks", "K7F: yes",
                                  "ok K7F", "sure, K7F sounds good", "  Yes, k7f! go ahead", "K7F 👍"])
def test_a_typed_code_reads_as_a_plain_yes(text):
    assert verdict(text, codes=("K7F",)) == YES
    assert verdict(text) == YES                 # a code with a digit is never a word, whether or not it is known


@pytest.mark.parametrize("text", ["no K7F", "K7F: no", "nope K7F", "No, K7F."])
def test_a_typed_code_reads_as_a_plain_no(text):
    assert verdict(text, codes=("K7F",)) == NO


def test_an_all_letter_code_is_set_aside_only_when_it_is_known():
    assert verdict("yes THE", codes=("THE",)) == YES
    assert verdict("yes THE") == UNCLEAR
    assert verdict("yes K7FX", codes=("K7F",)) == UNCLEAR
    assert verdict("yes AB2 AB3", codes=("AB2",)) == YES          # two codes: neither is a reference


@pytest.mark.parametrize("text, codes, expected", [
    ("Yes K7F, but...", ("K7F", "BUT"), UNCLEAR), ("yes K7F but wait", ("K7F", "BUT", "WAIT"), UNCLEAR),
    ("yes K7F", ("K7F", "YES"), YES), ("Yes K7F", ("K7F", "YES"), YES), ("sure K7F, go ahead", ("K7F", "SURE"), YES),
    ("nah K7F", ("K7F", "NAH"), NO), ("yes K7F, and the file", ("K7F", "THE"), YES),
])
def test_an_open_code_spelled_like_a_word_never_erases_the_word(text, codes, expected):
    """A code spelled like a word ("BUT", "YES") is set aside only where it stands as a code: typed in capitals as
    the ask shows it. In ordinary case it is their word, and a hedge or an answer word is never erased by it."""
    assert verdict(text, codes=codes) == expected, text


def test_a_code_spelled_like_a_word_answers_its_own_ask_in_capitals():
    assert verdict("yes BUT", codes=("BUT",)) == YES
    assert verdict("yes BUT", code="BUT") == YES
    assert verdict("yes but", code="BUT") == UNCLEAR                  # the hedge stays a hedge
    assert verdict("yes abc", code="ABC") == YES                      # their ask's own code, no word the tables read
    assert verdict("yes abc", codes=("ABC",)) == UNCLEAR              # another ask's code in lower case is a word
    assert typed("yes K7F but wait", "BUT", "WAIT", "K7F") == ("K7F",)
    assert typed("yes K7F, and the file", "THE", "K7F") == ("K7F",)
    assert typed("YES K7F", "YES", "K7F") == ("YES", "K7F")


@pytest.mark.parametrize("text, named", [
    ("yes K7F", ("K7F",)), ("yes k7f", ("K7F",)), ("K7F: yes", ("K7F",)), ('"Yes, K7F."', ("K7F",)),
    ("yes Q2Z\n> K7F", ("Q2Z",)), ('yes Q2Z, not "K7F"', ("Q2Z",)), ('[Replying to: "K7F"]\n\nyes Q2Z', ("Q2Z",)),
    ("Sam said yes K7F", ()), ("Robin: yes K7F", ()), ("yes Q2Z and K7F", ("Q2Z", "K7F")), ("yes AB2", ()),
    ("yes K7FX", ()),
])
def test_a_code_counts_only_in_their_own_words(text, named):
    """An open code the owner typed shows its ask (``named``; it never binds or approves): a quoted, replied-to or
    passed-on code names nothing, and a code no open ask holds is no reference."""
    assert typed(text, "K7F", "Q2Z") == named


def test_a_private_use_character_he_typed_never_stands_for_a_code():
    assert typed('yes \ue100 \uf8ff\n> K7F', "K7F") == ()
    assert verdict("yes \uf8ff K7F", codes=("K7F",)) == YES


@pytest.mark.parametrize("text, expected", [
    ('[Replying to: "Can I text Sam the recap?"]\n\nyes', YES),
    ('[Replying to your previous message: "Delete old-builds?"]\n\ngo ahead', YES),
    ('[Replying to: "yes K7F"]\n\nhm', NONE),
    ('[Replying to: "Set Friend to auto? Reply yes K7F or no K7F."]\n\nno', NO),
    ("[Voice note] yes please", YES),
    ('"Yes, go ahead."', YES),
    ('"No."', NO),
])
def test_the_gateways_prefixes_and_a_voice_notes_quotes_are_set_aside(text, expected):
    assert verdict(text) == expected


@pytest.mark.parametrize("text, expected", [("Robin, yes please", YES), ("@robin go ahead", YES),
                                            ("Wait, yes", UNCLEAR), ("Never, ever", NONE),
                                            ("Tomorrow, yes", LATER), ("Actually, no", NO)])
def test_a_leading_name_or_mention_is_set_aside_never_a_table_word(text, expected):
    assert verdict(text) == expected


def test_normalise_lowercases_and_strips_what_the_tables_do_not_read():
    assert normalise('[Replying to: "x"]\n\n  Yes, K7F!!  ', codes=("K7F",)) == "yes"
    assert normalise("👍🏽 Thanks") == "👍 thanks"
    assert normalise("Robin, sure") == "sure"
    assert normalise("Why?") == "why?"
    assert verdict("Robin: sure") == NONE          # a chat line naming its speaker is someone else's words


# -- what the reading carries --------------------------------------------------------------------------------

def test_the_reading_says_why_and_what_a_later_check_needs():
    clean = read("yes please")
    assert clean.verdict == YES and clean.positive and clean.answer_word and not clean.hard_hedge and clean.reason
    hedged = read("yes but not the payment")
    assert hedged.verdict == UNCLEAR and hedged.hedge == "but" and hedged.hard_hedge and hedged.answer_word
    soft = read("maybe yes")
    assert soft.verdict == UNCLEAR and soft.answer_word and not soft.hard_hedge
    asked = read("yes?")
    assert asked.verdict == UNCLEAR and asked.hard_hedge
    placed = read("yes hmm")
    assert placed.verdict == UNCLEAR and "hmm" in placed.reason
    assert read("no").said == "no" and read("sure").said == "yes" and read("later").said == ""
    assert not read("Sam said yes").answer_word


def test_the_tables_are_module_constants_and_every_yes_form_is_in_them():
    for form in YES_FORMS:
        assert answers.canonical(form) in answers.YES_PHRASES or answers.canonical(form) in answers.WEAK_YES, form
    for form in NO_FORMS:
        assert answers.canonical(form) in answers.NO_PHRASES, form


# -- the 1.11 people-change cases, carried over with their 1.12 reading ------------------------------------------

@pytest.mark.parametrize("reply", ["yes AB2", "Yes, AB2.", "YES ab2!", "ab2 yes", "y AB2", "ok AB2", "approve AB2",
                                   '"Yes, AB2."'])
def test_the_owners_yes_with_the_code_is_still_a_yes(reply):
    assert verdict(reply, codes=("AB2",), strict=True) == YES


@pytest.mark.parametrize("reply", ["no AB2", "No, AB2.", "ab2 no", "nope AB2", "deny AB2", '"No AB2"'])
def test_the_owners_no_with_the_code_is_still_a_no(reply):
    assert verdict(reply, codes=("AB2",), strict=True) == NO


@pytest.mark.parametrize("reply, expected", [
    ("If I say yes AB2, approve it.", UNCLEAR), ("Please explain yes AB2.", UNCLEAR),
    ("What does AB2 do? Do not approve it", NONE), ("yes AB2?", UNCLEAR), ("yes AB2 but not yet", LATER),
    ("AB2", NONE), ("yes and no AB2", UNCLEAR), ("She wrote: yes AB2", NONE),
    ('[Replying to: "x"]\n\nhmm', NONE), ("", NONE), (None, NONE), ("maybe AB2", NONE), ("yes AB2 tomorrow", LATER),
])
def test_what_1_11_refused_is_still_no_yes(reply, expected):
    reading = read(reply, codes=("AB2",), strict=True)
    assert reading.verdict == expected and not reading.positive, reply


@pytest.mark.parametrize("reply", ["yes AB2 thanks", "yes AB2 AB3", "yes", "yes AB3"])
def test_what_1_11_refused_only_for_its_shape_is_now_a_yes(reply):
    """1.11 read only the whole reply ``yes <code>``; a thanks after it, a second code or no code at all is still their
    plain yes. Which ask it answers is never the reader's to say: a code is never a reference."""
    assert verdict(reply, codes=("AB2", "AB3")) == YES


@pytest.mark.parametrize("message, yes", [("what is K7F?", True), ("what is K7F? do not approve it", False),
                                          ("yes K7F", False), ("If I say yes K7F, go ahead", True)])
def test_the_cadence_questions_rejections_carry_over(message, yes):
    reading = read(message, codes=("K7F",), strict=True)
    assert reading.verdict != (YES if yes else NO)


# -- states_change: a permission or cadence the owner states in their own words --------------------------------------

SAM = ("Sam Rivera", "Sam", "+15550100")


@pytest.mark.parametrize("text", ["You can message Sam any time", "you can text Sam whenever you like",
                                  "Message Sam without asking", "sam is fine to contact anytime",
                                  "You can call Sam Rivera any time.", "you may message +1 555 0100 whenever"])
def test_an_owner_stated_permission_names_the_person_and_the_change(text):
    assert states_change([text], {"operation": "set_permission", "value": "auto"}, SAM)


@pytest.mark.parametrize("text", ["sure", "yes", "you can message him any time", "never message Sam without asking",
                                  "don't message Sam any time you like", "stop texting Sam whenever",
                                  "You can message Samantha any time", "Robin said you can message Sam any time",
                                  "\"you can message Sam any time\" is what the note says"])
def test_words_that_do_not_state_the_widening_do_not(text):
    assert not states_change([text], {"operation": "set_permission", "value": "auto"}, SAM)


def test_a_name_another_contact_shares_does_not_count():
    change = {"operation": "set_permission", "value": "auto"}
    assert not states_change(["you can message Sam any time"], change, SAM, shared=("Sam",))
    assert states_change(["you can message Sam Rivera any time"], change, SAM, shared=("Sam",))


def test_any_of_his_last_three_messages_counts():
    change = {"operation": "set_permission", "value": "auto"}
    texts = ["you can message Sam any time", "and the recap too", "yes"]
    assert states_change(texts, change, SAM)
    assert not states_change(["ok", "yes"], change, SAM)


@pytest.mark.parametrize("text, minutes, stated", [
    ("check in with Sam every week", 10080, True), ("Check in with Sam every 2 days", 2880, True),
    ("check in with Sam every day", 1440, True), ("ping Sam every 90 minutes", 90, True),
    ("check in with Sam every two weeks", 20160, True), ("check in with Sam every 2 weeks", 10080, False),
    ("don't check in with Sam every day", 1440, False), ("check in with Sam every week", None, False),
    ("check in with Robin every week", 10080, False),
])
def test_an_owner_stated_cadence_is_read_exactly(text, minutes, stated):
    assert bool(states_change([text], {"operation": "set_cadence", "minutes": minutes}, SAM)) is stated


def test_a_merge_always_confirms_and_a_narrowing_is_never_stated_as_a_widening():
    assert not states_change(["merge Sam into Sam Rivera"], {"operation": "merge", "drop": "c-2"}, SAM)
    assert not states_change(["you can message Sam any time"], {"operation": "set_permission", "value": "never"}, SAM)
    statement = states_change(["never message Sam without asking"], {"operation": "set_permission", "value": "auto"},
                              SAM)
    assert not statement and "negat" in statement.reason
