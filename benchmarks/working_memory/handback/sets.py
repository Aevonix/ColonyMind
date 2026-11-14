"""The handback sets: synthetic finished worker tasks with keyed detail questions, gold by construction.

A set holds ``CHILDREN`` children. Each child is a finished worker task: the material the worker read (document
passages, tool results and messages, each a unit with a handle such as ``d2.p3``, ``t4`` or ``m7``), and the
worker's own work. Its exact context is three messages: the worker's system prompt, the task with the material, and
the work. Each child has ``QUESTIONS_PER_CHILD`` keyed detail questions in the shares of ``SHARES``:

- ``contradiction``: a value stated and later corrected, in a later unit; the gold is the later value and the stale
  value is the question's distractor;
- ``relation``: who is whose direct manager, which task another depends on directly; the next link of the chain
  (the manager's manager, the dependency's dependency) is the distractor, and the question asks for the direct link
  only, since every link of a dependency chain must finish before the task can start;
- ``cross_child``: the value of the entity another child's material assigns to a person; the child holds the values
  of three entities of that kind (the other two are distractors) and only the other child holds the assignment;
- ``single``: a value stated once.

Every gold answer and distractor is a code (two letters, a dash, four digits) unique in its set, so scoring is an
exact match (my assumption: codes keep scoring free of paraphrase). Every gold answer occurs in its child's
material and never in its question. Only generic words appear: Owner, Sam, Robin, and handles such as ``p-214``.

A set is a pure function of its seed and ``VERSION``; ``frozen.json`` holds each run seed's hash, and
``load_frozen_set`` refuses a set whose hash is not frozen. Version 2 asks for a task's direct prerequisite (version 1
asked which task must be finished before it can start, which the dependency's own dependency answers too); the
material is drawn from the same random stream (``STREAM``), so each seed's material is version 1's, byte for byte.

Sizes (my assumptions): ``CHILDREN`` 8, ``QUESTIONS_PER_CHILD`` 40 in equal shares, material of ``MATERIAL_TOKENS``
8,000 to 24,000 tokens per child, counted here at ``CHARS_PER_TOKEN`` 3.5 characters per token (what Qwen3.5-4B's
tokenizer measured on this text; the run reports the real tokenizer's count).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

VERSION = "wm-handback-2"
#: The random stream the material and questions are drawn from: version 1's, so only the wording changed.
STREAM = "wm-handback-1"
CHILDREN = 8
QUESTIONS_PER_CHILD = 40
SHARES = {"contradiction": 10, "relation": 10, "cross_child": 10, "single": 10}
MATERIAL_TOKENS = (8000, 24000)
CHARS_PER_TOKEN = 3.5
#: How many people and tasks the filler reuses (my assumption); a quarter as many values per kind.
BACKGROUND = 40
FROZEN = Path(__file__).resolve().parent / "frozen.json"

PROJECTS = ["Alder", "Birch", "Cedar", "Delta", "Ember", "Fjord", "Garnet", "Harbor", "Iris", "Juniper", "Kestrel",
            "Lumen", "Maple", "Nimbus", "Onyx", "Pine", "Quartz", "Ridge", "Sable", "Tundra"]
#: (entity kind, its code, the attribute asked about, the value's code).
KINDS = [("shipment", "SH", "delivery dock", "DK"), ("workshop", "WS", "room code", "RM"),
         ("contract", "CT", "reference number", "CN"), ("budget line", "BL", "cost code", "CC"),
         ("vendor", "VN", "account number", "VA"), ("badge", "BG", "access code", "AC"),
         ("server", "SV", "maintenance ticket", "MT"), ("printer", "PR", "asset tag", "AT"),
         ("order", "OR", "tracking number", "TR"), ("room booking", "RB", "confirmation code", "BK")]
SENDERS = ["Owner", "Sam", "Robin"]
TOOLS = ["inventory lookup", "calendar export", "ticket search", "directory query", "spreadsheet read"]
DOCUMENTS = ["status memo", "vendor review notes", "planning spec", "meeting minutes", "handover checklist",
             "risk register", "procurement summary", "facilities plan"]

SINGLE = ["The {attr} for {entity} is {value}.", "For {entity}, the {attr} on record is {value}.",
          "{Entity} has {attr} {value}."]
CORRECTION = ["Correction: the {attr} for {entity} is now {value}; disregard the earlier one.",
              "Update: {entity} was changed to {attr} {value}.",
              "Please note that the {attr} for {entity} changed to {value}."]
REPORTS = ["{a} reports to {b}.", "{b} is the direct manager of {a}."]
DEPENDS = ["Task {x} depends on task {y}.", "Task {x} cannot start until task {y} is finished."]
LINKS = ["{person} was assigned {entity}.", "{Entity} is handled by {person}."]
FILLER = [
    "The team reviewed the open items for {entity} and agreed to revisit them next week.",
    "{sender} asked for the latest figures on {entity} before the planning call.",
    "No blockers were reported for {entity} in the last cycle.",
    "Notes from the sync: {entity} stays on track and {person} follows up.",
    "The schedule for {entity} was discussed again without a new decision.",
    "{person} confirmed that the paperwork for {entity} is complete.",
    "A reminder went out to file the weekly report on {entity} by Friday.",
    "{sender} mentioned {entity} twice but raised no new questions.",
    "The review of {entity} is expected to close before the end of the month.",
    "{person} will share the updated checklist for {entity} with the group.",
    "Budget questions about {entity} were deferred to the next review.",
    "The vendor meeting touched on {entity} only briefly.",
    "Everyone agreed the handover for {entity} needs one more pass.",
    "{person} and {person2} are both copied on changes to {entity}.",
    "Quality checks on {entity} passed without comments.",
    "Travel for the {entity} visit is still being arranged by {person}.",
]
FILLER_FACTS = SINGLE + ["{person} reports to {person2}.", "Task {task} depends on task {task2}."]
QUESTIONS = {
    "single": "What is the {attr} for {entity}?",
    "contradiction": "What is the current {attr} for {entity}?",
    "reports": "Who is the direct manager of {a}?",
    "depends": "Which task does task {x} depend on directly, not through another task?",
    "cross_child": "What is the {attr} for the {kind} that was assigned to {person}?",
}
WORKER_SYSTEM = ("You are a worker agent. A parent agent gave you one task. You read the material you were given "
                 "and did the work. Answer requests about your task from that material.")


class _Names:
    """Unique codes within one set."""

    def __init__(self, rng: random.Random):
        self.rng, self.used = rng, set()

    def code(self, prefix: str, digits: int = 4) -> str:
        low, high = 10 ** (digits - 1), 10 ** digits - 1
        while True:
            value = f"{prefix}-{self.rng.randint(low, high)}"
            if value not in self.used:
                self.used.add(value)
                return value

    def entity(self, kind: Tuple[str, str, str, str]) -> str:
        return f"{kind[0]} {self.code(kind[1], 3)}"

    def person(self) -> str:
        return self.code("p", 3)

    def task(self) -> str:
        return self.code("T", 3)


class _Background:
    """Names the filler reuses: reserved in the set, so no question's gold or distractor is ever among them."""

    def __init__(self, names: _Names):
        self.people = [names.person() for _ in range(BACKGROUND)]
        self.tasks = [names.task() for _ in range(BACKGROUND)]
        self.values = {kind[3]: [names.code(kind[3]) for _ in range(BACKGROUND // 4)] for kind in KINDS}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def render_unit(unit: Dict[str, Any]) -> str:
    return f"[{unit['handle']}] {unit['label']}\n{unit['text']}"


def material_text(child: Dict[str, Any]) -> str:
    return "\n\n".join(render_unit(unit) for unit in child["units"])


def task_text(child: Dict[str, Any]) -> str:
    return (f"You are worker {child['id']}. Task from the parent agent: review the material collected for "
            f"{child['project']} and prepare the review notes. The material follows; every passage starts with its "
            f"handle in square brackets.\n\n" + material_text(child))


def _filler(rng: random.Random, pool: _Background, entities: List[str]) -> str:
    template = rng.choice(FILLER + FILLER_FACTS) if rng.random() < 0.25 else rng.choice(FILLER)
    kind = rng.choice(KINDS)
    entity = rng.choice(entities)
    person, person2 = rng.sample(pool.people, 2)
    task, task2 = rng.sample(pool.tasks, 2)
    value = rng.choice(pool.values[kind[3]])
    return template.format(entity=entity, Entity=_cap(entity), sender=rng.choice(SENDERS), person=person,
                           person2=person2, task=task, task2=task2, attr=kind[2], value=value)


def _skeleton(rng: random.Random, pool: _Background, entities: List[str], target_chars: int) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    documents = tools = messages = 0
    size, day, minute = 0, 1, 8 * 60
    while size < target_chars:
        roll = rng.random()
        if roll < 0.45:
            documents += 1
            title = rng.choice(DOCUMENTS)
            batch = [{"handle": f"d{documents}.p{p}", "kind": "document",
                      "label": f"Document {documents} ({title}), paragraph {p}"} for p in range(1, rng.randint(3, 7))]
        elif roll < 0.65:
            tools += 1
            batch = [{"handle": f"t{tools}", "kind": "tool_result", "label": f"Tool result ({rng.choice(TOOLS)})"}]
        else:
            messages += 1
            minute += rng.randint(20, 400)
            day, minute = day + minute // (24 * 60), minute % (24 * 60)
            sender = rng.choice(SENDERS)
            batch = [{"handle": f"m{messages}", "kind": "message",
                      "label": f"Message from {sender}, day {day} {minute // 60:02d}:{minute % 60:02d}"}]
        for unit in batch:
            unit["sentences"] = [_filler(rng, pool, entities) for _ in range(rng.randint(3, 8))]
            size += sum(len(s) + 1 for s in unit["sentences"]) + len(unit["label"]) + 12
            units.append(unit)
    return units


def _place(rng: random.Random, unit: Dict[str, Any], sentence: str) -> None:
    unit["sentences"].insert(rng.randint(0, len(unit["sentences"])), sentence)


def build_set(seed: int) -> Dict[str, Any]:
    rng = random.Random(f"{STREAM}:{seed}")
    names = _Names(rng)
    pool = _Background(names)
    projects = rng.sample(PROJECTS, CHILDREN)
    children: List[Dict[str, Any]] = []
    plans: List[Dict[str, Any]] = []
    for index in range(CHILDREN):
        cid = f"w{index + 1}"
        entities = [names.entity(rng.choice(KINDS)) for _ in range(12)]
        target = int(rng.randint(*MATERIAL_TOKENS) * CHARS_PER_TOKEN)
        children.append({"id": cid, "project": f"Project {projects[index]}", "entities": entities,
                         "target_chars": target})
        plans.append({"facts": [], "questions": []})

    def fact(child: int, sentence: str, after: Any = None) -> Dict[str, Any]:
        item = {"sentence": sentence, "after": after, "unit": None}
        plans[child]["facts"].append(item)
        return item

    for index in range(CHILDREN):
        cid = children[index]["id"]
        for _ in range(SHARES["single"]):
            kind = rng.choice(KINDS)
            entity, value = names.entity(kind), names.code(kind[3])
            item = fact(index, rng.choice(SINGLE).format(attr=kind[2], entity=entity, Entity=_cap(entity), value=value))
            plans[index]["questions"].append({"type": "single", "question": QUESTIONS["single"].format(
                attr=kind[2], entity=entity), "gold": value, "distractors": [], "gold_fact": item})
        for _ in range(SHARES["contradiction"]):
            kind = rng.choice(KINDS)
            entity, stale, value = names.entity(kind), names.code(kind[3]), names.code(kind[3])
            old = fact(index, rng.choice(SINGLE).format(attr=kind[2], entity=entity, Entity=_cap(entity), value=stale))
            new = fact(index, rng.choice(CORRECTION).format(attr=kind[2], entity=entity, value=value), after=old)
            plans[index]["questions"].append({"type": "contradiction", "question": QUESTIONS["contradiction"].format(
                attr=kind[2], entity=entity), "gold": value, "distractors": [stale], "gold_fact": new,
                "stale_fact": old})
        for n in range(SHARES["relation"]):
            if n % 2 == 0:
                a, b, c = names.person(), names.person(), names.person()
                first = fact(index, rng.choice(REPORTS).format(a=a, b=b))
                fact(index, rng.choice(REPORTS).format(a=b, b=c))
                question, gold, distractor = QUESTIONS["reports"].format(a=a), b, c
            else:
                x, y, z = names.task(), names.task(), names.task()
                first = fact(index, rng.choice(DEPENDS).format(x=x, y=y))
                fact(index, rng.choice(DEPENDS).format(x=y, y=z))
                question, gold, distractor = QUESTIONS["depends"].format(x=x), y, z
            plans[index]["questions"].append({"type": "relation", "question": question, "gold": gold,
                                              "distractors": [distractor], "gold_fact": first})
        for _ in range(SHARES["cross_child"]):
            other = rng.choice([i for i in range(CHILDREN) if i != index])
            kind = rng.choice(KINDS)
            trio = [(names.entity(kind), names.code(kind[3]), names.person()) for _ in range(3)]
            items = [fact(index, rng.choice(SINGLE).format(attr=kind[2], entity=e, Entity=_cap(e), value=v))
                     for e, v, _ in trio]
            links = [rng.choice(LINKS).format(person=p, entity=e, Entity=_cap(e)) for e, _, p in trio]
            link_items = [fact(other, sentence) for sentence in links]
            entity, value, person = trio[0]
            plans[index]["questions"].append({
                "type": "cross_child", "question": QUESTIONS["cross_child"].format(attr=kind[2], kind=kind[0],
                                                                                   person=person),
                "gold": value, "distractors": [v for _, v, _ in trio[1:]], "gold_fact": items[0],
                "link": links[0], "link_fact": link_items[0], "children": sorted({cid, children[other]["id"]},
                                                                                 key=lambda c: int(c[1:]))})
    out_children, questions = [], []
    for index, child in enumerate(children):
        facts = plans[index]["facts"]
        fact_chars = sum(len(item["sentence"]) + 1 for item in facts)
        units = _skeleton(rng, pool, child["entities"], max(child["target_chars"] - fact_chars, 2000))
        for item in facts:
            if item["after"] is not None:
                earliest = item["after"]["unit"] + 1
                item["unit"] = rng.randint(earliest, len(units) - 1)
            else:
                item["unit"] = rng.randint(0, max(0, (len(units) * 2) // 3))
            _place(rng, units[item["unit"]], item["sentence"])
        for unit in units:
            unit["text"] = " ".join(unit.pop("sentences"))
        counts = {kind: sum(1 for u in units if u["kind"] == kind) for kind in ("document", "tool_result", "message")}
        reviewed = ", ".join(child["entities"][:4])
        work = (f"Work notes for {child['project']}: I read {counts['document']} document passages, "
                f"{counts['tool_result']} tool results and {counts['message']} messages. I reviewed the records for "
                f"{reviewed} and the rest of the material. The review notes are ready for the parent; I changed "
                f"nothing in the source records.")
        record = {"id": child["id"], "project": child["project"], "units": units, "work": work}
        record["context"] = [{"role": "system", "content": WORKER_SYSTEM},
                             {"role": "user", "content": task_text(record)},
                             {"role": "assistant", "content": work}]
        out_children.append(record)
        for number, plan in enumerate(plans[index]["questions"], start=1):
            question = {"id": f"s{seed}-{child['id']}-q{number:02d}", "child": child["id"],
                        "children": plan.get("children", [child["id"]]), "type": plan["type"],
                        "question": plan["question"], "gold": plan["gold"], "distractors": plan["distractors"],
                        "gold_handle": units[plan["gold_fact"]["unit"]]["handle"]}
            if "stale_fact" in plan:
                question["stale_handle"] = units[plan["stale_fact"]["unit"]]["handle"]
            if "link" in plan:
                question["link"] = plan["link"]
                question["link_handle"] = {"child": plan["children"][0] if plan["children"][0] != child["id"]
                                           else plan["children"][1]}
            questions.append(question)
    # A link's unit is known only once its host child's units exist.
    by_child = {c["id"]: c for c in out_children}
    for index, child in enumerate(children):
        for plan, question in zip(plans[index]["questions"],
                                  [q for q in questions if q["child"] == child["id"]]):
            if "link" in plan:
                host = question["link_handle"]["child"]
                question["link_handle"] = {"child": host,
                                           "handle": by_child[host]["units"][plan["link_fact"]["unit"]]["handle"]}
    return {"version": VERSION, "seed": seed, "children": out_children, "questions": questions}


def set_hash(built: Dict[str, Any]) -> str:
    canonical = json.dumps(built, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def frozen() -> Dict[str, Any]:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def load_frozen_set(seed: int) -> Dict[str, Any]:
    """The set for ``seed``, only when its seed is frozen and it still has the frozen hash."""
    table = frozen()
    entry = next((item for item in table["sets"] if item["seed"] == seed), None)
    if entry is None or table["version"] != VERSION:
        raise ValueError(f"seed {seed} is not frozen for {VERSION}")
    built = build_set(seed)
    if set_hash(built) != entry["sha256"]:
        raise ValueError(f"the set for seed {seed} no longer has its frozen hash")
    return built


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Print the hash of each seed's set (to freeze them).")
    parser.add_argument("seeds", type=int, nargs="+")
    args = parser.parse_args()
    print(json.dumps({"version": VERSION, "sets": [{"seed": s, "sha256": set_hash(build_set(s))} for s in args.seeds]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
