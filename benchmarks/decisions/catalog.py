"""The bake-off's sets together: built under a seed, frozen by hash, read back only when every hash matches.

``build(seed)`` returns every set (``SETS``): the five measured sets of ``sets.py`` with their novel and adversarial
supplements, the mind's new sets (``sets_mind.py``) and the Hermes-side sets (``sets_hermes.py``). ``freeze`` writes
their hashes, counts and seed to ``frozen.json``; a run reads ``frozen()``, which rebuilds the sets under the frozen
seed and refuses any whose hash is not the manifest's, so every arm of a campaign answers exactly the frozen items.
A set changes when a phrase it is built from changes (a test's labelled phrases, a template): ``drift()`` names
such sets, and re-freezing is a deliberate step taken before a campaign, never during one. The items are not
committed: they are rebuilt from the repository, and ``dump`` writes them out for review.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sets  # noqa: E402
import sets_hermes  # noqa: E402
import sets_mind  # noqa: E402
from sets_mind import SEED  # noqa: E402

FIVE = ("outreach_reply", "opt_out", "no_reminders", "interest_settled", "owner_verdict")
MIND = ("ask_answer", "owner_send", "owner_instruction", "question_answer", "report_substance", "claim_relation")
HERMES = ("cron_wake", "inbound_triage", "tool_screen")
SETS = (*FIVE, *MIND, *HERMES)
#: Sets read from pinned public files (``public_sets.py``), fixed by their pins rather than this manifest.
PUBLIC = ("tool_screen_public",)
SUBSETS = ("repo", "novel", "adversarial")
#: The least share of a set's items in forms neither the tables nor the templates use (an assumption, not measured).
NOVEL_SHARE = 0.25
FROZEN = HERE / "frozen.json"
#: The field that names one message whatever else varies (the ask, the tool, the chat or sender, the gate's short
#: and full states by day and night): items that share it are variants of one message and share a group.
VARIANT_FIELD = {"owner_send": "text", "ask_answer": "words", "inbound_triage": "text", "tool_screen": "text"}


class FrozenError(ValueError):
    """The frozen items are missing, or not the ones the manifest names."""


def key(item: Dict[str, Any]) -> str:
    """An item's key: the hash of its point and fields (``measure.key``)."""
    return hashlib.sha256(json.dumps([item["point"], item["fields"]], sort_keys=True).encode()).hexdigest()[:16]


def group(item: Dict[str, Any]) -> str:
    """The item's group for folding: the one it was built with (the passage a screen result was made from, clean
    or planted), else its message (``VARIANT_FIELD``), else the item alone. A group is never split across folds, so
    no held-out fold is scored on text a training fold was fitted on."""
    if item.get("group"):
        return str(item["group"])
    field = VARIANT_FIELD.get(item["point"])
    value = item["fields"].get(field) if field else None
    if isinstance(value, str):
        return f"{item['point']}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"
    return key(item)


def _five(name: str) -> List[Dict[str, Any]]:
    measured = [{**item, "subset": "novel" if item["source"] == "novel" else "repo"}
                for item in sets.build([name])[name]]
    return sets_mind.unique(measured + sets_mind.supplement(name))


def build_one(name: str, seed: int = SEED) -> List[Dict[str, Any]]:
    if name in FIVE:
        return _five(name)
    if name in MIND:
        return sets_mind.BUILDERS[name](seed)
    if name in HERMES:
        return sets_hermes.BUILDERS[name](seed)
    raise KeyError(name)


def build(seed: int = SEED, names=None) -> Dict[str, List[Dict[str, Any]]]:
    return {name: build_one(name, seed) for name in (names or SETS)}


def canonical(items: List[Dict[str, Any]]) -> bytes:
    return json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(items: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(canonical(items)).hexdigest()


def describe(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"sha256": digest(items), "n": len(items),
            "subsets": dict(sorted(Counter(item["subset"] for item in items).items())),
            "labels": dict(sorted(Counter(item["gold"] for item in items).items())),
            "points": dict(sorted(Counter(item["point"] for item in items).items()))}


def manifest(built: Dict[str, List[Dict[str, Any]]], seed: int) -> Dict[str, Any]:
    return {"seed": seed, "sets": {name: describe(items) for name, items in built.items()}}


def freeze(seed: int = SEED, path: Path = FROZEN) -> Dict[str, Any]:
    """Build every set under ``seed`` and write its manifest (hashes, counts, the seed)."""
    made = manifest(build(seed), seed)
    Path(path).write_text(json.dumps(made, indent=1, sort_keys=True) + "\n")
    return made


def pinned(path: Path = FROZEN) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
        value["sets"], value["seed"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FrozenError(f"no frozen manifest at {path} ({type(error).__name__}); run measure.py freeze") from None
    return value


def frozen(path: Path = FROZEN, names=None) -> Dict[str, List[Dict[str, Any]]]:
    """The frozen sets, rebuilt under the frozen seed; ``FrozenError`` for a set whose hash is not the manifest's."""
    made = pinned(path)
    out = {}
    for name in (names or SETS):
        if name in PUBLIC:
            import public_sets
            out[name] = public_sets.screen()            # a missing or changed pinned file is refused
            continue
        if name not in made["sets"]:
            raise FrozenError(f"set {name} is not frozen")
        items = build_one(name, made["seed"])
        if digest(items) != made["sets"][name]["sha256"]:
            raise FrozenError(f"set {name} is not the frozen one: a phrase or template it is built from changed "
                              "since the freeze (measure.py freeze --check names them; re-freeze before a campaign)")
        out[name] = items
    return out


def drift(path: Path = FROZEN) -> Dict[str, str]:
    """Per set, "changed" where a fresh build under the frozen seed differs from the manifest, "new" where the
    manifest does not name it."""
    made = pinned(path)
    out = {}
    for name, items in build(made["seed"]).items():
        if name not in made["sets"]:
            out[name] = "new"
        elif digest(items) != made["sets"][name]["sha256"]:
            out[name] = "changed"
    return out


def dump(path: Path, seed: int = SEED) -> None:
    """Write every set's items (for review; never committed)."""
    Path(path).write_text(json.dumps(build(seed), indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def all_items(built: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [item for items in built.values() for item in items]


def identity(built: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
    """What a campaign's answers were asked about: each set's hash, and each of its points' question fingerprint
    (``questions.fingerprint``). Every arm records it; a summary reads only answers whose identity is the current
    one (``report.check_identity``)."""
    import questions
    points = sorted({item["point"] for items in built.values() for item in items})
    return {"sets": {name: digest(items) for name, items in built.items()},
            "questions": {point: questions.fingerprint(point) for point in points}}


__all__ = ["FIVE", "FROZEN", "FrozenError", "HERMES", "MIND", "NOVEL_SHARE", "PUBLIC", "SEED", "SETS", "SUBSETS",
           "VARIANT_FIELD", "all_items", "build", "build_one", "canonical", "describe", "digest", "drift", "dump",
           "freeze", "frozen", "group", "identity", "key", "manifest", "pinned"]
