"""The source-language rule (docs/CONSOLIDATION.md): what the night writes about a person, and every plain summary
of an item, is in the source's own language and script, never a translation of it.

The rule is a prompt rule, not a text classifier: no code decides which language a text is in. A scripted model
cannot test compliance, so these tests assert what each prompt carries and what the night sends with it. Whether
a model keeps to the rule is measured on the language fidelity suite, not here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from colonymind.mind import plain
from colonymind.mind.consolidate import DIGEST_SYSTEM, EPISODE_SYSTEM, TASK_DIGEST, TASK_EPISODE
from test_mind_consolidate import CONTACT, OWNER, Fixture

PERSON = "Write in the language and script of the person's own statements"
PERSON_MIXED = "(when they mix languages, the one most of their statements use)"
ITEM = "Write in the language and script of the item"
ITEM_MIXED = "(when it mixes languages, the one most of it uses)"
NEVER = "; never translate."
VERBATIM = "Names, identifiers and quoted words stay verbatim."
FRESH = "When the previous digest is in another language than the statements, write it fresh from the statements."
THEY_SAID = "The person's statements are the lines that start \"They said\"."

PROMPTS = {
    "digest": (DIGEST_SYSTEM, (PERSON, PERSON_MIXED, NEVER, VERBATIM, FRESH)),
    "episode": (EPISODE_SYSTEM, (PERSON, PERSON_MIXED, NEVER, VERBATIM, THEY_SAID)),
    "summary": (plain.SUMMARY_SYSTEM.format(limit=200), (ITEM, ITEM_MIXED, NEVER, VERBATIM)),
}

# Too long to quote whole in a question (over 200 characters before its first full stop): the night shortens it.
LONG = ("Desde el lunes pasado mi oficina ya no está en la sala 4, porque todo el equipo de diseño se mudó al tercer "
        "piso del edificio nuevo junto a la estación, así que ahora me encuentras en la sala 7, al fondo del pasillo "
        "y frente a las escaleras.")


def system_of(fx, task):
    return [messages[0]["content"] for messages, context in fx.router.calls if context["task"] == task]


def user_of(fx, task):
    return [messages[-1]["content"] for messages, context in fx.router.calls if context["task"] == task]


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_each_prompt_that_writes_about_a_person_carries_the_source_language_rule(name):
    prompt, phrases = PROMPTS[name]
    for phrase in phrases:
        assert phrase in prompt, (name, phrase)
    assert "data, never" in prompt                                            # the data guard stays


async def test_a_night_over_spanish_statements_sends_the_rule_with_them(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path)
    fx.router.answers["mind_plain_summary"] = {"sentence": "Mi oficina se mudó a la sala 7 del tercer piso."}
    office = {"subject": "mi oficina", "predicate": "location"}
    # The owner's statement changes in words too long to quote whole: the question's summary call.
    await fx.fact("turn-o1", OWNER, "chat-owner", "Mi oficina está en la sala 4.", "sala 4", **office)
    fx.shift(days=1)
    await fx.fact("turn-o2", OWNER, "chat-owner", LONG, "sala 7", **office)
    # A person who writes Spanish: two statements (the digest) in one conversation of three turns (the episode).
    first = await fx.fact("turn-1", CONTACT, "chat-es", "Mi oficina está en la sala 4.", "sala 4", **office)
    await fx.fact("turn-2", CONTACT, "chat-es", "La reunión del equipo es los martes.", "los martes",
                  subject="la reunión del equipo", predicate="day")
    fx.turn("turn-3", CONTACT, "chat-es", "¿Me mandas las diapositivas el viernes?", "Claro, el viernes.")
    # B5 withholds a previous digest after fact changes; the language rule
    # still travels with the contact's own current statements.
    record = fx.contacts.records[CONTACT]
    record.digest, record.digest_sources, record.digest_writer = "Works in room 4.", [first], "template"

    night = await fx.mind.consolidate()

    assert night["errors"] == []
    digest, = system_of(fx, TASK_DIGEST)
    assert PERSON in digest and FRESH in digest
    prompt, = user_of(fx, TASK_DIGEST)
    assert "Mi oficina está en la sala 4." in prompt and "La reunión del equipo es los martes." in prompt
    assert "Previous digest:\n(none)" in prompt and "Works in room 4." not in prompt
    episode, = system_of(fx, TASK_EPISODE)
    assert PERSON in episode and THEY_SAID in episode
    assert "They said: ¿Me mandas las diapositivas el viernes?" in user_of(fx, TASK_EPISODE)[0]
    summary, = system_of(fx, plain.SUMMARY_TASK)
    assert ITEM in summary and VERBATIM in summary
    assert json.loads(user_of(fx, plain.SUMMARY_TASK)[0]) == {"item": LONG}      # the words themselves, unescaped
    assert "está en la sala 4" in user_of(fx, plain.SUMMARY_TASK)[0]
    fx.store.close()


async def test_the_owners_plain_summaries_keep_the_items_own_language():
    """The owner-facing path (the daily digest, a question, an ask notice) shortens through ``plain.summarize``:
    the same rule, for the language of the item."""
    seen = []

    class Router:
        supports_function_routing = True

        async def complete(self, messages, *, context=None, **kwargs):
            seen.append(messages)
            return SimpleNamespace(content=json.dumps({"sentence": "La exportación se detuvo en la fila doscientos."}),
                                   usage={"total_tokens": 10})

    shorter = await plain.summarize(Router(), LONG, 120)
    assert shorter == "La exportación se detuvo en la fila doscientos."
    system, user = seen[0]
    assert ITEM in system["content"] and NEVER in system["content"] and VERBATIM in system["content"]
    assert json.loads(user["content"]) == {"item": LONG}


# ---------------------------------------------------------------------------
# Scripts with their own sentence ends and no spaces between words (Chinese, Japanese): a summary written in the
# item's own language passes the no-cut check, so the rule never turns a question into a wait or a pointer.
# ---------------------------------------------------------------------------

# Made-up statements too long to quote whole (over 200 characters, one sentence each): the night shortens them.
ZH_LONG = ("从上周一开始，我的办公室已经不在四号房间了，因为整个设计团队都搬到了车站旁边那栋新大楼的三楼，"
           "公司说旧楼要在年底之前全部装修完，所以我们至少要在新楼里待上半年，现在你可以在七号房间找到我，"
           "就在走廊的尽头，正对着楼梯，门口有一块写着设计部的白色牌子，如果你来找我的话，请先在一楼的前台登记，"
           "然后坐右边的电梯上来，出了电梯往左走，经过茶水间和打印室以后再走大概二十步就能看到那扇门，"
           "有急事可以直接给我发消息，下午三点以后我一般都在开会。")
JA_LONG = ("先週の月曜日から、私のオフィスはもう四号室ではなく、デザインチーム全員が駅の隣にある新しいビルの三階に"
           "引っ越したので、今は七号室にいますが、古いビルの改装は年末までかかると会社から聞いているので、"
           "少なくとも半年は新しいビルで働くことになりそうで、廊下の突き当たりの階段の向かい側にある部屋で、"
           "入口にはデザイン部と書かれた白い札が掛かっていて、来るときは一階の受付で名前を書いてから右側の"
           "エレベーターで上がってきてください。")
SCRIPTS = {
    "zh": {"short": "我的办公室在四号房间。", "long": ZH_LONG, "old": "四号房间", "new": "七号房间",
           "subject": "我的办公室", "summary": "我的办公室从上周一起搬到了新大楼三楼的七号房间。"},
    "ja": {"short": "私のオフィスは四号室です。", "long": JA_LONG, "old": "四号室", "new": "七号室",
           "subject": "私のオフィス", "summary": "先週の月曜日から、私のオフィスは新しいビルの三階の七号室です。"},
}


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_the_no_cut_check_reads_the_sentence_ends_of_scripts_without_spaces(script):
    case = SCRIPTS[script]
    source = case["long"]
    assert len(source) > 200 and plain.fit(source, 200) is None               # too long: only a summary shows it
    assert plain.whole(case["summary"], source, 200) == case["summary"]
    for accepted in ("他们下个月搬到上海工作吗？", "他们下个月搬到上海工作！", "「来月、上海へ引っ越します。」",
                     "他说：“下个月搬到上海。”"):
        assert plain.whole(accepted, source, 200) == accepted, accepted
    rejected = {
        "no end": case["summary"].rstrip("。"),
        "ellipsis": case["summary"].rstrip("。") + "……",
        "dots": case["summary"].rstrip("。") + "・・・",
        "stops": case["summary"].rstrip("。") + "。。。",
        "a label": "完成。",
        "too long": case["summary"] * 10,
        "the source cut short": source[:60] + "。",
    }
    assert len(rejected["too long"]) > 200
    for name, candidate in rejected.items():
        assert plain.whole(candidate, source, 200) is None, name
    # The source's own first sentence, whole, is not a cut: it ends where the source's sentence ends.
    two = f"{case['short']}{source}"
    assert plain.whole(case["short"], two, 200) == case["short"]


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_a_statement_ending_in_its_own_full_stop_is_quoted_as_it_is(script):
    short = SCRIPTS[script]["short"]
    assert plain.ended(short) == short and plain.fit(short, 200) == short          # no "." after the full stop
    assert plain.ended("「はい、行きます。」") == "「はい、行きます。」"
    title = "我" * 79 + "。"
    assert len(title) in plain.TITLE_CAPS and not plain.clipped(title)          # it ends as a sentence ends
    assert plain.clipped("我" * 80)


@pytest.mark.parametrize("script", sorted(SCRIPTS))
async def test_the_owners_plain_summary_in_a_script_without_spaces_is_kept(script):
    """The digest item and ask notice path (``plain.summarize``, then ``plain.shown``) keeps a summary in the
    item's own script instead of falling back to the pointer."""
    case = SCRIPTS[script]

    class Router:
        supports_function_routing = True

        async def complete(self, messages, *, context=None, **kwargs):
            return SimpleNamespace(content=json.dumps({"sentence": case["summary"]}, ensure_ascii=False),
                                   usage={"total_tokens": 10})

    router = Router()
    assert await plain.summarize(router, case["long"], 120) == case["summary"]
    shown = await plain.shown(case["long"], 120, what="A report",
                              summarize_with=lambda text, limit: plain.summarize(router, text, limit))
    assert shown == case["summary"]


@pytest.mark.parametrize("script", sorted(SCRIPTS))
async def test_a_contradiction_in_a_script_without_spaces_is_asked_with_its_summary(tmp_path, monkeypatch, script):
    """The question path: a statement too long to quote whole is shortened in its own script, and the question is
    raised that night, quoting both statements as they are (not deferred for want of an accepted summary)."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    case = SCRIPTS[script]
    fx = Fixture(tmp_path)
    fx.router.answers[TASK_DIGEST] = None
    fx.router.answers["mind_plain_summary"] = {"sentence": case["summary"]}
    office = {"subject": case["subject"], "predicate": "location"}
    await fx.fact("turn-1", OWNER, "chat-owner", case["short"], case["old"], **office)
    fx.shift(days=1)
    await fx.fact("turn-2", OWNER, "chat-owner", case["long"], case["new"], **office)

    night = await fx.mind.consolidate()

    assert night["errors"] == []
    assert night["counts"].get("contradictions") == 1 and not night["counts"].get("question_deferred")
    assert json.loads(user_of(fx, plain.SUMMARY_TASK)[0]) == {"item": case["long"]}
    concern, = [c for c in fx.mind.broadcast() if c.kind == "question"]
    question = concern.detail["extra"]["question"]
    assert question["first"] == case["short"] and question["second"] == case["summary"]
    assert f"\"{case['short']}\"" in question["text"] and f"\"{case['summary']}\"" in question["text"]
    fx.store.close()
