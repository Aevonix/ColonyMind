"""Plain words read in the text's own script (``mind/plain.py``): where a sentence ends, which full stop ends it,
and when a summary ends as a sentence ends.

A statement too long for its room is shown by its leading whole sentences (``fit``); only when not even the first one
fits does a summary call run. The sentence ends are the text's own: a Chinese or Japanese 。！？, the Devanagari danda,
the Arabic question mark, and the Latin . ! ? before a sentence that starts with a capital, a digit or a letter of a
script without capitals. A sentence that ends with no mark gets its own script's full stop (``ended``), and a script
written with no sentence mark at all (Thai, Lao) ends where its words end, so a summary in it passes the no-cut
check (``whole``) like any other. Every statement here is made up.
"""

from __future__ import annotations

import pytest

from colonymind.mind import plain

# A first sentence short enough to show, then one too long for any room here (made-up statements).
ZH_FIRST = "我的办公室现在在七号房间。"
ZH_REST = ("从上周一开始，整个设计团队都搬到了车站旁边那栋新大楼的三楼，公司说旧楼要在年底之前全部装修完，"
           "所以我们至少要在新楼里待上半年，如果你来找我的话，请先在一楼的前台登记，然后坐右边的电梯上来。")
JA_FIRST = "私のオフィスは七号室になりました！"
JA_REST = ("先週の月曜日からデザインチーム全員が駅の隣にある新しいビルの三階に引っ越したので、古いビルの改装が"
           "終わるまで少なくとも半年は新しいビルで働くことになりそうで、来るときは一階の受付で名前を書いてください。")
HI_FIRST = "मेरा दफ्तर अब सात नंबर कमरे में है।"
HI_REST = ("पिछले सोमवार से पूरी डिज़ाइन टीम स्टेशन के पास वाली नई इमारत की तीसरी मंज़िल पर चली गई है और हम कम से "
           "कम छह महीने वहीं काम करेंगे क्योंकि पुरानी इमारत की मरम्मत साल के अंत तक चलेगी।")
AR_FIRST = "هل ستأتي غدا؟"
AR_REST = ("الاجتماع في الغرفة السابعة بعد الظهر وسنناقش خطة المشروع الجديدة مع الفريق كله قبل أن نرسلها إلى "
           "الإدارة في نهاية الأسبوع القادم إن شاء الله.")
AR_DOT_FIRST = "ذهبت إلى المكتب."
AR_DOT_REST = ("ثم عدت إلى البيت بعد الاجتماع الطويل مع فريق التصميم في الطابق الثالث من المبنى الجديد قرب المحطة "
               "وتحدثنا عن خطة المشروع كلها.")
DE_FIRST = "Das Büro ist jetzt in Raum sieben."
DE_REST = ("Über den Winter bleibt das ganze Team im neuen Gebäude am Bahnhof, weil das alte Haus bis zum Jahresende "
           "renoviert wird und niemand dort arbeiten darf, auch nicht an den Wochenenden oder nachts.")
ES_FIRST = "Mi oficina está en la sala siete."
ES_REST = ("¿Vienes mañana a la reunión del equipo de diseño en el edificio nuevo junto a la estación, en la tercera "
           "planta, justo al final del pasillo frente a la escalera?")

LONG_CASES = {
    "zh": (ZH_FIRST, ZH_REST, ""),
    "ja": (JA_FIRST, JA_REST, ""),
    "hi": (HI_FIRST, HI_REST, " "),
    "ar": (AR_FIRST, AR_REST, " "),
    "ar-latin-stop": (AR_DOT_FIRST, AR_DOT_REST, " "),
    "de": (DE_FIRST, DE_REST, " "),
    "es": (ES_FIRST, ES_REST, " "),
}


@pytest.mark.parametrize("case", sorted(LONG_CASES))
def test_a_long_statement_shows_its_first_sentence_in_its_own_script(case):
    first, rest, space = LONG_CASES[case]
    text = f"{first}{space}{rest}"
    assert len(first) < 80 < len(text)
    assert plain.sentences(text) == [first, rest]
    assert plain.fit(text, 80) == first


async def test_a_long_chinese_statement_needs_no_summary_call_when_its_first_sentence_fits():
    calls = []

    async def summarize_with(text, limit):
        calls.append(text)
        return None

    shown = await plain.shown(f"{ZH_FIRST}{ZH_REST}", 80, what="A statement", summarize_with=summarize_with)
    assert shown == ZH_FIRST and calls == []


def test_the_sentences_a_text_keeps_together():
    # A lower-case word after a Latin full stop continues the sentence; a full-width full stop before a digit is a
    # decimal point; an ellipsis in the middle ends nothing a lower-case word follows.
    for text in ("It ran at 9.30 p.m. and failed.", "Die Datei ist 3.5 MB groß und liegt im Ordner.",
                 "我们的预算是３．５万元。", "Wait... then it ran again."):
        assert plain.sentences(text) == [text], text
    # Several of one script's marks, and a closing quote after a mark, stay with the sentence they end. A quoted
    # clause that ends with a mark ends the sentence only where a space and a new sentence follow its closing quote:
    # right after it the sentence may go on ("之前", "という"), and the text alone cannot tell.
    assert plain.sentences("真的吗？！「好的。」然后呢？") == ["真的吗？！", "「好的。」然后呢？"]
    assert plain.sentences('He said "Go." Then he left.') == ['He said "Go."', "Then he left."]


# A condition or a negation that follows a quoted clause (made-up instructions, longer than the room).
ZH_QUOTED = ("在收到“同意发布。”的书面确认之前，无论团队里其他人怎么说，也无论截止日期有多紧、客户催得多急、上级在电话里怎么"
             "暗示，都不要把这份季度报告发布到公司网站、新闻稿、社交媒体账号或者任何其他对外渠道上，否则公司会违反与客户签订的"
             "保密合同并承担赔偿责任。")
JA_QUOTED = ("「公開を承認します。」という返信を受け取るまでは、チームの誰が何と言っても、締め切りがどれほど迫っていても、"
             "顧客からどれだけ催促されても、この四半期報告書を会社のウェブサイトやプレスリリース、ソーシャルメディア、"
             "その他の外部の経路には絶対に公開しないでください。")
EN_QUOTED = ('Do not publish the quarterly report until the owner writes "Approved. Go ahead." in the thread, whatever '
             'the deadline says and whoever asks for it before then.')


@pytest.mark.parametrize("text", [ZH_QUOTED, JA_QUOTED, EN_QUOTED], ids=["zh", "ja", "en"])
def test_a_mark_inside_a_quoted_clause_never_ends_the_sentence_around_it(text):
    """Shortened to its opening quote, the instruction would lose the condition or the "do not" that follows it: the
    whole sentence is one, so a room too small for it takes a summary or a pointer, never the quote alone."""
    assert len(text) > 120
    assert plain.sentences(text) == [text]
    assert plain.fit(text, 120) is None


def test_a_mark_inside_a_quote_that_goes_on_ends_nothing():
    assert plain.sentences("在收到“同意。可以发布。”之前不要发布。") == ["在收到“同意。可以发布。”之前不要发布。"]
    assert plain.sentences("「はい。どうぞ。」と言うまで待ってください。次に進みます。") == [
        "「はい。どうぞ。」と言うまで待ってください。", "次に進みます。"]
    assert plain.sentences("Il a dit « Non. Pas encore. » puis il est parti. Nous attendons.") == [
        "Il a dit « Non. Pas encore. » puis il est parti.", "Nous attendons."]


def test_a_quote_or_bracket_that_never_closes_holds_nothing():
    # An emoticon or a lone quote opens nothing: the sentences after it still end where they end.
    assert plain.sentences("The export failed :( The rerun is queued. Check it tonight.") == [
        "The export failed :( The rerun is queued.", "Check it tonight."]
    assert plain.sentences("导出失败了（原因未知。重新运行已排队。今晚再看。") == [
        "导出失败了（原因未知。", "重新运行已排队。", "今晚再看。"]


@pytest.mark.parametrize("text, ended", [
    ("我的办公室在七号房间", "我的办公室在七号房间。"),
    ("我的办公室在七号房间，", "我的办公室在七号房间。"),
    ("私のオフィスは七号室です", "私のオフィスは七号室です。"),
    ("मेरा दफ्तर सात नंबर कमरे में है", "मेरा दफ्तर सात नंबर कमरे में है।"),
    ("ذهبت إلى المكتب", "ذهبت إلى المكتب."),
    ("ذهبت إلى المكتب،", "ذهبت إلى المكتب."),
    ("میں دفتر گیا۔ پھر گھر واپس آیا", "میں دفتر گیا۔ پھر گھر واپس آیا۔"),
    ("Das Büro ist in Raum sieben", "Das Büro ist in Raum sieben."),
    ("Done:", "Done."),
    ("我明天去 Tokyo 开会", "我明天去 Tokyo 开会。"),
    ("Meeting with 田中 about the plan", "Meeting with 田中 about the plan."),
], ids=["zh", "zh-comma", "ja", "hi", "ar", "ar-comma", "ur-own-stop", "de", "colon", "mostly-zh", "mostly-en"])
def test_a_sentence_with_no_mark_ends_with_its_own_scripts_full_stop(text, ended):
    assert plain.ended(text) == ended


@pytest.mark.parametrize("text", ["สำนักงานของฉันย้ายไปที่ห้องเจ็ดแล้ว", "ຫ້ອງການຂອງຂ້ອຍຍ້າຍໄປຫ້ອງເຈັດແລ້ວ"],
                         ids=["th", "lo"])
def test_a_script_written_without_sentence_marks_gets_none(text):
    assert plain.ended(text) == text and plain.ended(f"{text},") == text


TH_SOURCE = ("ตั้งแต่วันจันทร์ที่แล้วทีมออกแบบทั้งหมดย้ายไปอยู่ที่ชั้นสามของอาคารใหม่ข้างสถานี "
             "เพราะอาคารเก่าจะปรับปรุงจนถึงสิ้นปี ตอนนี้คุณหาฉันได้ที่ห้องเจ็ดตรงสุดทางเดิน "
             "ถ้ามาหาฉันกรุณาลงชื่อที่เคาน์เตอร์ชั้นหนึ่งก่อนแล้วขึ้นลิฟต์ด้านขวา")
TH_SUMMARY = "ทีมออกแบบย้ายไปชั้นสามของอาคารใหม่และตอนนี้ฉันอยู่ห้องเจ็ด"
LO_SOURCE = ("ຕັ້ງແຕ່ວັນຈັນທີ່ຜ່ານມາທີມອອກແບບທັງໝົດໄດ້ຍ້າຍໄປຢູ່ຊັ້ນສາມຂອງຕຶກໃໝ່ໃກ້ສະຖານີ "
             "ເພາະວ່າຕຶກເກົ່າຈະປັບປຸງຈົນຮອດທ້າຍປີ ດຽວນີ້ເຈົ້າຊອກຫາຂ້ອຍໄດ້ທີ່ຫ້ອງເຈັດ")
LO_SUMMARY = "ທີມອອກແບບຍ້າຍໄປຊັ້ນສາມຂອງຕຶກໃໝ່ແລະຂ້ອຍຢູ່ຫ້ອງເຈັດ"


@pytest.mark.parametrize("source, summary", [(TH_SOURCE, TH_SUMMARY), (LO_SOURCE, LO_SUMMARY)], ids=["th", "lo"])
def test_a_thai_or_lao_summary_with_no_final_mark_passes_the_no_cut_check(source, summary):
    assert plain.whole(summary, source, 120) == summary
    # The rest of the check still holds: the source's own opening cut short, a trailing ellipsis, a trailing comma
    # and a summary over the room are refused.
    for rejected in (source[:40], summary + "...", summary + "…", summary + ",", summary * 4):
        assert plain.whole(rejected, source, 120) is None, rejected


def test_a_latin_summary_with_no_final_mark_still_fails():
    source = "The nightly export stopped at row two hundred because the disk was full. " * 3
    assert plain.whole("The export stopped at row two hundred", source, 120) is None
    ended = "The export stopped at row two hundred."
    assert plain.whole(ended, source, 120) == ended
