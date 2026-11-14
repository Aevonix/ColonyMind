"""The owner's daily digest and the messages that share its style (architecture 6.3).

One message a day, only when something waits or happened, in three labelled parts written in plain words:

- **Waiting for your OK**: one line per open ask, what it is and when it was asked, with no code and no reply
  instruction. The ask's own message carries its reply line and is what the owner answers (``outbox``): a reaction,
  a quoted reply or their next message binds to it, never a code.
- **Question (just answer in words)**: what the mind asked the owner that the owner answers in their own words (which of
  two statements to keep, for one) (``questions.py``).
- **Done or found since yesterday**: what the mind did, found, held back or learned since the last digest. What
  already reached the owner (a question, a notice, an answer the owner asked for) is not listed again.

Every item is one self-contained plain sentence or a few: long source text is shortened to whole sentences, or
summarised by a checked typed call, never cut (``plain.py``). A question the mind sends on its own uses the same
label.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence

TITLE = "Daily digest"
WAITING = "Waiting for your OK"
QUESTION = "Question (just answer in words)"
DONE = "Done or found since yesterday"
# The label 1.11 notices and digests listed open asks under, each line starting with its code: read only to find
# which asks such a message showed (``Mind._asks_shown_by``), never written.
NEEDS_OK_1_11 = "Needs your OK (reply yes or no with the code)"
# The room one item takes, in characters: what does not fit is shortened whole (``plain.shown``).
ITEM_CHARS = 280
QUESTION_CHARS = 600


def _block(label: str, lines: Iterable[str]) -> str:
    return "\n".join([f"{label}:", *(f"- {line}" for line in lines)])


def asked_ago(asked_at: Optional[datetime], now: datetime) -> str:
    """When an ask was asked, in words: "Asked 5 hours ago." (hours, then days)."""
    if asked_at is None:
        return ""
    hours = max(0.0, (now - asked_at) / timedelta(hours=1))
    if hours < 1:
        return "Asked less than an hour ago."
    if hours < 24:
        count = int(hours)
        return f"Asked {count} hour{'s' if count != 1 else ''} ago."
    count = int(hours // 24)
    return f"Asked {count} day{'s' if count != 1 else ''} ago."


def ask_list(sentences: Sequence[str], reply: str) -> str:
    """The one message that shows the asks waiting beyond the day's own messages, numbered: the numbers mean this list
    (``answers.bind``). ``reply`` is the reply line of a single ask, used when only one waits."""
    if len(sentences) == 1:
        return f"One more thing waits for your OK: {sentences[0]} {reply}"
    numbered = " ".join(f"{number}. {sentence}" for number, sentence in enumerate(sentences, 1))
    answer = ("Reply 1, 2, both or neither." if len(sentences) == 2
              else "Reply with the numbers you approve, all or none.")
    return f"{len(sentences)} more things wait for your OK: {numbered} {answer}"


def render(*, needs: Sequence[str] = (), questions: Sequence[str] = (), done: Sequence[str] = (),
           context: Sequence[str] = ()) -> Optional[str]:
    """The digest's text, or None when nothing waits and nothing happened. ``needs`` are the waiting asks' lines,
    ``questions`` and ``done`` plain sentences; ``context`` (what the mind keeps working on) joins the last part
    but never makes a digest on its own."""
    if not (needs or questions or done):
        return None
    blocks = [TITLE]
    if needs:
        blocks.append(_block(WAITING, needs))
    if questions:
        blocks.append(_block(QUESTION, questions))
    if done or context:
        blocks.append(_block(DONE, [*done, *context]))
    return "\n\n".join(blocks)


def question_message(text: str) -> str:
    """A question the mind sends on its own: the digest's question label, then the question."""
    return f"{QUESTION}: {text}"


def lines_of(text: str, label: str) -> List[str]:
    """The items listed under ``label`` in a rendered digest (for readers and tests)."""
    items: List[str] = []
    inside = False
    for line in str(text or "").splitlines():
        if line == f"{label}:":
            inside = True
            continue
        if inside and line.startswith("- "):
            items.append(line[2:])
        elif inside and line.strip() == "":
            break
    return items


__all__ = ["DONE", "ITEM_CHARS", "NEEDS_OK_1_11", "QUESTION", "QUESTION_CHARS", "TITLE", "WAITING", "ask_list",
           "asked_ago", "lines_of", "question_message", "render"]
