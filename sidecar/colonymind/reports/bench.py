"""The paired benchmark's report files, read as plain words (``colonymind report bench DIR``).

Each ``paired_report`` (``colonymind models paired report --json``, and the ``report-NNN.json`` a run writes) holds
its contrasts under the plan's pre-registered rule. A contrast is read as the one faculty its two arms differ in
(``arms.subject``), oriented so that the difference is the faculty's: with it minus without it. The rule is read
in the faculty's direction too: it **helped** when the sign test and the interval clear it in its favour exactly
as the plan's rule demands of a treatment; it **looked harmful** when the same rule clears it the other way, or
when it did worse by more than the plan's non-inferiority margin (a point estimate, so "points to harm", not
proven); otherwise it is **unproven**. A faculty with no usable contrast is **not measured**. By the rule the
reports carry, a faculty stays on unless it has shown harm.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import arms
from .registry import BY_KEY, FACULTIES
from .text import Document, listing, plural, points, sentence

READINGS = ("helped", "unproven", "harmful", "not measured")
SPLITS = {"dev": "development set", "heldout": "held-out set", "held_out": "held-out set",
          "frozen_public_evaluation": "public evaluation set", "public": "public set"}
# A family's name by the longest prefix of its dataset version. Names are for reading only: evidence is kept per
# exact dataset version and split (``dataset_key``), so two families with one name never stand in for each other.
FAMILIES = {"mind-initiative": "Initiative", "mind-outreach": "Owner outreach", "mind-affect": "Feelings",
            "mind-memory": "Memory", "mind-memory-2": "Harder memory", "mind-people": "People",
            "mind-opinions": "Opinions", "mind-self": "Identity", "mind-improve": "Self-improvement",
            "mind-drives": "Drives", "paired-agent": "General task"}
DEFAULT_MARGIN = -10.0


@dataclass
class Loaded:
    path: Path
    mtime: float
    report: Dict[str, Any]


@dataclass
class Evidence:
    """One contrast read for the faculty (or drive, or ColonyMind as a whole) it measures."""

    subject: arms.Subject
    loaded: Loaded
    contrast: Dict[str, Any]
    family: str                    # the family and set in words, for reading
    dataset: Tuple[str, str] = ("", "")    # the exact dataset version and split: what a reading is kept per
    unit: str = "scenario"
    units: int = 0
    repetitions: int = 1           # runs of each unit per arm; a unit's pass value is the mean of its runs
    reading: str = "unavailable"
    strength: Optional[str] = None
    with_rate: float = 0.0
    without_rate: float = 0.0
    difference: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    p: float = 1.0
    mde: Optional[int] = None
    margin: float = DEFAULT_MARGIN
    demonstrated: bool = False
    harm_demonstrated: bool = False
    with_arm: Dict[str, Any] = field(default_factory=dict)
    without_arm: Dict[str, Any] = field(default_factory=dict)
    shipped_off: Optional[frozenset] = None   # what the report records as shipped off (``arms.recorded_off``)

    @property
    def key(self) -> Optional[str]:
        return self.subject.key if self.subject.kind == "faculty" else None

    @property
    def when(self) -> float:
        return self.loaded.mtime

    @property
    def runs(self) -> int:
        return self.units * self.repetitions

    @property
    def with_passes(self) -> Optional[int]:
        """Passing runs (passing scenarios without repetitions); None when the rate is not a whole count."""
        return _whole(self.with_rate * self.runs)

    @property
    def without_passes(self) -> Optional[int]:
        return _whole(self.without_rate * self.runs)


def _whole(value: float) -> Optional[int]:
    return round(value) if abs(value - round(value)) < 1e-6 else None


def _number(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else 0


def _candidates(directory: Path) -> Iterable[Path]:
    """The report files under ``directory``: ``report-NNN.json`` in a plan directory and ``*.report.json``
    beside one. A plan's ``runs`` tree (containers' records, no reports) is not walked."""
    for root, directories, files in os.walk(directory):
        directories[:] = sorted(name for name in directories if name != "runs" and not name.startswith("."))
        for name in sorted(files):
            if (name.startswith("report-") or name.endswith(".report.json")) and name.endswith(".json"):
                yield Path(root) / name


def load(directory: Any) -> List[Loaded]:
    """Every paired report under ``directory``, each plan once: the latest view of it (the newest file, then the
    highest report number), oldest plan first."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such directory: {directory}")
    chosen: Dict[str, Tuple[Tuple[float, int], Loaded]] = {}
    for path in _candidates(directory):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (not isinstance(value, dict) or value.get("kind") != "paired_report"
                or not isinstance(value.get("statistics"), dict) or not isinstance(value.get("profiles"), dict)):
            continue
        mtime = path.stat().st_mtime
        key = str(value.get("manifest_sha256") or path)
        rank = (mtime, _number(path))
        if key not in chosen or rank > chosen[key][0]:
            chosen[key] = (rank, Loaded(path, mtime, value))
    return sorted((item for _, item in chosen.values()), key=lambda item: (item.mtime, str(item.path)))


def family_words(report: Dict[str, Any]) -> Tuple[str, str]:
    """The scenario family and its set in plain words: ``("Memory", "development set")``."""
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    version = str(dataset.get("version") or "")
    name = next((FAMILIES[prefix] for prefix in sorted(FAMILIES, key=len, reverse=True)
                 if version.startswith(prefix)), None)
    if name is None:
        stem = version.rsplit("-", 1)[0] if version.rsplit("-", 1)[-1].isdigit() else version
        name = sentence(stem.replace("-", " ").replace("_", " ")).rstrip(".") or "Unnamed"
    split = str(dataset.get("split") or "")
    return name, SPLITS.get(split, (split.replace("_", " ") or "unlabelled") + " set")


def family(report: Dict[str, Any]) -> str:
    """"Memory scenarios (development set)": one family and set in words."""
    name, split = family_words(report)
    return f"{name} scenarios ({split})"


def dataset_key(report: Dict[str, Any]) -> Tuple[str, str]:
    """The exact dataset version and split a report measured, the unit a faculty's latest reading is kept per: two
    families that read alike (``mind-memory-1`` and ``mind-memory-2``) stay apart. A report that names no version
    is kept per its family in words."""
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    version = str(dataset.get("version") or "")
    return (version, str(dataset.get("split") or "")) if version else ("", family(report))


def _oriented(contrast: Dict[str, Any], on: str) -> Tuple[float, float, float, float, float, int, int]:
    sign = 1 if on == "treatment" else -1
    interval = contrast.get("ci_pp") or {}
    lower, upper = float(interval.get("lower", 0.0)), float(interval.get("upper", 0.0))
    with_rate = float(contrast["treatment_pass_rate"] if sign > 0 else contrast["comparator_pass_rate"])
    without_rate = float(contrast["comparator_pass_rate"] if sign > 0 else contrast["treatment_pass_rate"])
    wins, losses = int(contrast.get("wins", 0)), int(contrast.get("losses", 0))
    if sign < 0:
        lower, upper, wins, losses = -upper, -lower, losses, wins
    return sign * float(contrast["delta_pp"]), lower, upper, with_rate, without_rate, wins, losses


def read(loaded: Loaded, contrast: Dict[str, Any]) -> Evidence:
    """One contrast of one report, read for what it measures."""
    from colonymind.qualification.paired_statistics import ALPHA, MIN_WINS, significant
    report = loaded.report
    profiles = report["profiles"]
    off = arms.recorded_off(report)
    treatment, comparator = profiles.get(contrast.get("treatment")), profiles.get(contrast.get("comparator"))
    subject = (arms.subject(treatment, comparator, off) if isinstance(treatment, dict) and isinstance(comparator, dict)
               else arms.Subject("mixed", keys=["an arm this report does not describe"]))
    rule = (report.get("statistics") or {}).get("rule") or {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    item = Evidence(subject=subject, loaded=loaded, contrast=contrast, family=family(report),
                    dataset=dataset_key(report),
                    unit=str(contrast.get("unit") or "scenario"),
                    units=int(contrast.get("units") or contrast.get("declared_units") or 0),
                    repetitions=max(1, int(dataset.get("repetitions") or 1)),
                    margin=float(rule.get("non_inferior_pp", DEFAULT_MARGIN)), shipped_off=off)
    on = subject.faculty_on or "treatment"
    item.with_arm, item.without_arm = (treatment, comparator) if on == "treatment" else (comparator, treatment)
    if contrast.get("verdict") not in {"demonstrated", "not_demonstrated"}:
        return item
    (item.difference, item.lower, item.upper, item.with_rate, item.without_rate,
     wins, losses) = _oriented(contrast, on)
    item.p = float((contrast.get("sign_test") or {}).get("p_two_sided", 1.0))
    item.mde = contrast.get("mde_pp")
    alpha, min_wins = float(rule.get("alpha", ALPHA)), int(rule.get("min_wins", MIN_WINS))
    item.demonstrated = significant(wins, losses, alpha=alpha, min_wins=min_wins) and item.lower > 0
    item.harm_demonstrated = significant(losses, wins, alpha=alpha, min_wins=min_wins) and item.upper < 0
    if subject.kind in {"mixed", "same", "unrecorded"}:
        item.reading = "unattributed"
    elif item.demonstrated:
        item.reading, item.strength = "helped", "demonstrated"
    elif item.harm_demonstrated:
        item.reading, item.strength = "harmful", "demonstrated"
    elif item.difference < item.margin:
        item.reading, item.strength = "harmful", "indicated"
    else:
        item.reading = "unproven"
    return item


def evidence(loaded: Sequence[Loaded]) -> List[Evidence]:
    """Every contrast of every report, newest report first."""
    items = []
    for report in sorted(loaded, key=lambda item: -item.mtime):
        for contrast in (report.report.get("statistics") or {}).get("contrasts") or []:
            if isinstance(contrast, dict):
                items.append(read(report, contrast))
    return items


def latest(items: Iterable[Evidence], key: str) -> List[Evidence]:
    """The newest usable reading of ``key`` on each dataset version and split (``dataset_key``), newest first."""
    chosen: Dict[Tuple[str, str], Evidence] = {}
    for item in sorted(items, key=lambda item: -item.when):
        if item.key == key and item.reading in READINGS and item.dataset not in chosen:
            chosen[item.dataset] = item
    return list(chosen.values())


def verdict(items: Iterable[Evidence], key: str) -> str:
    """A faculty's reading across its families: harmful anywhere wins, then helped, then unproven."""
    readings = {item.reading for item in latest(items, key)}
    for reading in ("harmful", "helped", "unproven"):
        if reading in readings:
            return reading
    return "not measured"


def verdicts(items: Sequence[Evidence]) -> Dict[str, str]:
    return {faculty.key: verdict(items, faculty.key) for faculty in FACULTIES}


# -- words ------------------------------------------------------------------------------------------------------


def _noun(item: Evidence) -> str:
    if item.subject.kind == "system":
        return "ColonyMind"
    if item.subject.kind == "drive":
        return f"the {item.subject.drive} drive"
    faculty = BY_KEY.get(item.subject.key or "")
    return faculty.noun if faculty is not None else "it"


def title(item: Evidence) -> str:
    kind = item.subject.kind
    if kind == "system":
        return "ColonyMind as a whole"
    if kind == "drive":
        return f"Drives: the {item.subject.drive} drive alone"
    if kind == "mixed":
        return "More than one change at once"
    if kind == "same":
        return "The same arm twice"
    if kind == "unrecorded":
        return "A difference the report cannot place"
    faculty = BY_KEY.get(item.subject.key or "")
    return faculty.title if faculty is not None else "Unnamed"


def compared(item: Evidence) -> str:
    """What was compared, the arm with the faculty first."""
    with_words = arms.describe(item.with_arm, item.shipped_off or frozenset())
    without_words = arms.describe(item.without_arm, item.shipped_off or frozenset())
    noun = _noun(item)
    if item.subject.kind == "unrecorded":
        switches = _names(item.subject.keys)
        pronoun = "them" if len(item.subject.keys) > 1 or BY_KEY[item.subject.keys[0]].plural else "it"
        return (f"Compared: {with_words}, against {without_words}; one of them sets {switches} itself and the "
                f"other leaves {pronoun} as its image shipped.")
    if item.subject.kind in {"faculty", "drive"} and item.subject.key != "initiative":
        if without_words == f"{with_words}, without {noun}":
            return f"Compared: {with_words}, against the same mind without {noun}."
        if item.subject.kind == "drive" and without_words == f"{with_words}, with {noun} at zero":
            return f"Compared: {with_words}, against the same mind with {noun} at zero."
        if with_words == f"{without_words}, with {noun} added":
            return f"Compared: {without_words}, with {noun} added, against the same mind without {noun}."
    return f"Compared: {with_words}, against {without_words}."


def _passed(item: Evidence, words: str, rate: float, passes: Optional[int], *, first: bool) -> str:
    units = plural(item.units, "probe" if item.unit == "probe" else "scenario")
    if item.repetitions > 1:
        # A unit's pass value is the mean of its runs: count runs, never units.
        over = f" over {units}, {item.repetitions} runs each" if first else ""
        if passes is not None:
            runs = plural(item.runs, "run") if first else str(item.runs)
            return f"{words} {passes} of {runs} ({round(100 * rate)}%){over}"
        return f"{words} {round(100 * rate)}% of its runs{over}"
    if passes is not None:
        return f"{words} {passes} of {units if first else item.units} ({round(100 * rate)}%)"
    return f"{words} {round(100 * rate)}% of the time" + (f" over {units}" if first else "")


def results(item: Evidence) -> str:
    kind, noun = item.subject.kind, _noun(item)
    if kind == "system":
        first, second = "With ColonyMind it passed", "stock Hermes"
    elif item.subject.key == "initiative":
        first, second = "With the mind on it passed", "with it off"
    else:
        first, second = f"With {noun} it passed", "without"
    lead = _passed(item, first, item.with_rate, item.with_passes, first=True)
    rest = _passed(item, f"{second},", item.without_rate, item.without_passes, first=False)
    return f"{lead}; {rest}."


def difference(item: Evidence) -> str:
    noun = _noun(item)
    interval = f"95% interval {points(item.lower)} to {points(item.upper)} points"
    if abs(item.difference) < 0.05:
        return f"That is no difference ({interval})."
    side = "in favour of" if item.difference > 0 else "against"
    return f"That is a difference of {points(item.difference)} points {side} {noun} ({interval})."


def _p(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}" if value < 0.01 else f"p = {value:.2f}"


def certainty(item: Evidence) -> str:
    if item.demonstrated or item.harm_demonstrated:
        return (f"This is unlikely to be chance: the pre-registered test gives {_p(item.p)} and the whole "
                f"interval is on one side of zero.")
    units = plural(item.units, "probe" if item.unit == "probe" else "scenario")
    reach = (f"only a difference of about {item.mde} points or more would reliably show"
             if isinstance(item.mde, (int, float)) else "only a very large difference would reliably show")
    return f"This could be chance ({_p(item.p)}). With {units}, {reach}."


def meaning(item: Evidence) -> str:
    noun = _noun(item)
    kind = item.subject.kind
    if item.reading == "unavailable":
        return ("Some scenarios had no usable result in one of the arms, so no comparison is made here; the run needs "
                "to be repeated or completed.")
    if kind == "mixed":
        return (f"These arms differ in more than one thing ({listing(item.subject.keys)}), so the result is not "
                f"read as any one faculty's.")
    if kind == "same":
        return "The same arm ran twice: this difference is the noise floor, not an effect."
    if kind == "unrecorded":
        return (f"Whether these arms differ in {_names(item.subject.keys)} depends on what the image that ran them "
                "shipped, which the report does not record, so the result is not read as any one faculty's.")
    if kind == "drive":
        return "A diagnostic of one drive; the drives faculty is judged by its own contrast."
    margin = points(abs(item.margin), signed=False)
    if item.reading == "helped":
        return sentence(f"{noun} measurably helped here").rstrip(".") + (
            ": keep it on." if kind == "faculty" else ".")
    if item.reading == "harmful" and item.strength == "demonstrated":
        return (sentence(f"{noun} measurably hurt here").rstrip(".")
                + ": by the rule, it goes off until a re-measure shows no harm.")
    if item.reading == "harmful":
        return (f"This points to harm: worse by more than the {margin}-point margin, though not proven. By the rule "
                f"that a faculty stays on unless it has shown harm, it is a candidate to switch off, and a re-measure "
                f"decides.")
    faculty = BY_KEY.get(item.subject.key or "")
    stays = faculty.verb("stays", "stay") if faculty is not None else "stays"
    return sentence(f"Not proven either way: by the rule, {noun} {stays} on")


def sentences(item: Evidence) -> List[str]:
    """Everything one contrast says, in order."""
    if item.reading == "unavailable":
        return [compared(item), meaning(item)]
    return [compared(item), results(item), difference(item), certainty(item), meaning(item)]


def _names(keys: Sequence[str]) -> str:
    return listing([BY_KEY[key].noun for key in keys])


def summary(items: Sequence[Evidence]) -> List[str]:
    """The three lines: what helped, what is unproven, what looked harmful."""
    readings = verdicts(items)
    helped = [key for key, value in readings.items() if value == "helped"]
    unproven = [key for key, value in readings.items() if value == "unproven"]
    harmful = [key for key, value in readings.items() if value == "harmful"]
    unmeasured = [key for key, value in readings.items() if value == "not measured"]

    def where(key: str, reading: str) -> str:
        found = [item for item in latest(items, key) if item.reading == reading]
        if not found:
            return ""
        item = found[0]
        place = item.family.split(" (")[0].lower()
        if reading == "harmful":
            how = "proven" if item.strength == "demonstrated" else "not proven"
            return f" ({points(abs(item.difference), signed=False)} points worse on {place}, {how})"
        return f" (on {place})"

    lines = []
    lines.append("What helped: " + (listing([BY_KEY[key].noun + where(key, "helped") for key in helped]) + "."
                                    if helped else "nothing has been demonstrated yet."))
    text = (f"{_names(unproven)}, measured with no clear difference" if unproven else "")
    if unmeasured:
        text += ("; " if text else "") + f"{plural(len(unmeasured), 'faculty', 'faculties')} not measured yet"
    lines.append("Unproven: " + (text or "nothing") + "; by the rule, an unproven faculty stays on.")
    lines.append("What looked harmful: " + (listing([BY_KEY[key].noun + where(key, "harmful") for key in harmful])
                                            + "." if harmful else "nothing."))
    return lines


def _date(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d")


def document(loaded: Sequence[Loaded]) -> Document:
    """The whole benchmark report: the summary, then each plan (newest first) and each of its contrasts."""
    doc = Document("Paired benchmark results")
    items = evidence(loaded)
    if not loaded:
        doc.paragraph("No paired benchmark reports were found in that directory.")
        return doc
    doc.paragraph(f"What {plural(len(loaded), 'benchmark plan')} say about the mind's faculties. A faculty stays on "
                  "unless it has shown harm.")
    doc.bullets(summary(items))
    for report in sorted(loaded, key=lambda item: -item.mtime):
        scenarios = [item for item in items if item.loaded is report]
        units = max((item.units for item in scenarios), default=0) or int(report.report.get("declared_episodes") or 0)
        name, split = family_words(report.report)
        doc.heading(f"{name} scenarios ({split}, {plural(units, 'scenario')})")
        reference = report.report.get("profiles", {}).get(report.report.get("reference_arm"))
        lead = f"Measured on {_date(report.mtime)}."
        if isinstance(reference, dict):
            off = arms.recorded_off(report.report) or frozenset()
            lead += f" Every other arm is compared with {arms.describe(reference, off)}."
        doc.paragraph(lead)
        for item in scenarios:
            doc.heading(title(item), level=3)
            doc.paragraph(" ".join(sentences(item)))
    return doc


__all__ = ["Evidence", "Loaded", "READINGS", "compared", "dataset_key", "document", "evidence", "family",
           "family_words", "latest", "load",
           "meaning", "read", "sentences", "summary", "title", "verdict", "verdicts"]
