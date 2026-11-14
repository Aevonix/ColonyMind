"""The four arms: what the parent reads about a finished child, answered through one parent prompt.

- ``summary``: the child's own handoff summary, which the stand-in model writes from the child's exact context under
  the handoff instruction a stock host gives its subagents (today's contract);
- ``excerpts``: verbatim passages the child selects, with their handles, at most ``EXCERPT_BUDGET`` tokens per child
  (my assumption); the child names handles and the text is copied from its material, so an excerpt is verbatim by
  construction;
- ``consult``: the question appended to the child's exact context (its prompt and work, byte for byte), no tools
  offered, the answer returned with handles; the parent reads that answer;
- ``full``: the parent reads the child's task, full material and work, cold.

The parent is given the handbacks of the children a question involves (one, or two for a question that spans
children), in child order (my assumption: the parent knows which workers a question concerns).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from .sets import render_unit, task_text
except ImportError:  # run as a script
    from sets import render_unit, task_text  # type: ignore[no-redef]

ARMS = ("summary", "excerpts", "consult", "full")
#: Tokens of verbatim excerpts per child (my assumption).
EXCERPT_BUDGET = 2000
#: The handles a child may name, at most (my assumption).
EXCERPT_HANDLES = 40
#: A stock host's completion instruction to its subagents (Hermes v0.21.3, its workspace-path rule left out): the
#: child's final answer is returned to the parent as its summary.
HANDOFF_INSTRUCTION = (
    "Your task is finished. Provide a clear, concise summary of:\n"
    "- What you did\n- What you found or accomplished\n- Any files you created or modified\n"
    "- Any issues encountered\n\n"
    "Keep your final summary tight: lead with outcomes, prefer bullet points over paragraphs, and don't replay your "
    "whole process. Your response is returned to the parent agent as a summary, and overlong summaries crowd out the "
    "parent's context window.")
EXCERPT_INSTRUCTION = (
    "Your task is finished. The parent agent keeps only verbatim passages of your material, each with its handle, "
    "and will later ask detailed questions about the material: values, corrections, who reports to whom, what "
    "depends on what, and who was assigned what. Select the passages it will need. Reply with their handles only, "
    f"most important first, one per line, at most {EXCERPT_HANDLES}.")
CONSULT_INSTRUCTION = (
    "The parent agent has a question about the material you worked on. Answer from the material above only; no "
    "tools are available. Give the answer, then the handles of the passages it comes from in square brackets.")
PARENT_SYSTEM = (
    "You are the parent agent. Workers finished tasks for you; what each one handed back is below. Answer the "
    "question from what the workers handed back. Reply with the answer value only, on one line, with no other words. "
    "If the handbacks do not contain the answer, reply: unknown.")
HANDLE = re.compile(r"\b(d\d+\.p\d+|t\d+|m\d+)\b")


def summary_request(child: Dict[str, Any]) -> List[Dict[str, str]]:
    return list(child["context"]) + [{"role": "user", "content": HANDOFF_INSTRUCTION}]


def excerpt_request(child: Dict[str, Any]) -> List[Dict[str, str]]:
    return list(child["context"]) + [{"role": "user", "content": EXCERPT_INSTRUCTION}]


def consult_request(child: Dict[str, Any], question: Dict[str, Any]) -> List[Dict[str, str]]:
    return list(child["context"]) + [{"role": "user",
                                      "content": f"{CONSULT_INSTRUCTION}\n\nQuestion: {question['question']}"}]


def parse_handles(text: str, child: Dict[str, Any]) -> List[str]:
    """The handles a child named, in its order, once each, only those in its material, at most ``EXCERPT_HANDLES``."""
    known = {unit["handle"] for unit in child["units"]}
    out: List[str] = []
    for handle in HANDLE.findall(text or ""):
        if handle in known and handle not in out:
            out.append(handle)
    return out[:EXCERPT_HANDLES]


def excerpt_block(child: Dict[str, Any], handles: List[str], count: Callable[[str], int],
                  budget: int = EXCERPT_BUDGET) -> Tuple[str, List[str]]:
    """The named passages, verbatim with their handles, in the child's order of naming, stopping before the first
    one that would pass ``budget`` tokens; and the handles used."""
    units = {unit["handle"]: unit for unit in child["units"]}
    block, used = "", []
    for handle in handles:
        candidate = (block + "\n\n" if block else "") + render_unit(units[handle])
        if count(candidate) > budget:
            break
        block, used = candidate, used + [handle]
    return block, used


def full_text(child: Dict[str, Any]) -> str:
    return task_text(child) + "\n\nThe worker's own work:\n" + child["work"]


def _inputs(arm: str, the_set: Dict[str, Any], question: Dict[str, Any], summaries: Optional[Dict[str, str]],
            excerpts: Optional[Dict[str, str]], consults: Optional[Dict[str, str]]) -> Dict[str, str]:
    by_id = {child["id"]: child for child in the_set["children"]}
    out = {}
    for cid in question["children"]:
        if arm == "summary":
            out[cid] = (summaries or {})[cid]
        elif arm == "excerpts":
            out[cid] = (excerpts or {})[cid]
        elif arm == "consult":
            out[cid] = (consults or {})[cid]
        elif arm == "full":
            out[cid] = full_text(by_id[cid])
        else:
            raise ValueError(f"unknown arm {arm!r}")
    return out


def parent_messages(arm: str, the_set: Dict[str, Any], question: Dict[str, Any], *,
                    summaries: Optional[Dict[str, str]] = None, excerpts: Optional[Dict[str, str]] = None,
                    consults: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """The one parent prompt every arm answers through: the involved children's handbacks, then the question."""
    by_id = {child["id"]: child for child in the_set["children"]}
    handbacks = _inputs(arm, the_set, question, summaries, excerpts, consults)
    parts = [f"Worker {cid} ({by_id[cid]['project']}) handed back:\n{text}" for cid, text in handbacks.items()]
    return [{"role": "system", "content": PARENT_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts) + "\n\nQuestion: " + question["question"]}]
