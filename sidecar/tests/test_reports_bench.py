"""``colonymind report bench``: the paired benchmark's report files read as plain words, one contrast at a time."""
import json
import re
import subprocess
import sys

import pytest

from colonymind.qualification import paired
from colonymind.reports import arms, bench
from report_fixtures import write_bench_dir

# What a reader must never see: arm and switch names, the harness's field names, hashes, em dashes.
JARGON = re.compile(r"all-on|full-|minus_|plus_|_pp|treatment|comparator|delta|manifest|verdict:|"
                    r"not_demonstrated|mind\.faculties|[0-9a-f]{16}|\u2014")


def profile(name, **extra):
    return {"name": name, **paired.PROFILES.get(name, {}), **extra}


@pytest.mark.parametrize("treatment, comparator, kind, key, faculty_on", [
    ("all-on-skills", "all-on", "faculty", "skills", "comparator"),
    ("all-on", "all-on-status", "faculty", "status", "treatment"),
    ("full-plus-owner_gate", "full", "unrecorded", None, None),
    ("full-affect-plus-rules", "full-affect", "faculty", "affect_rules", "treatment"),
    ("full", "full-affect", "faculty", "affect", "treatment"),
    ("colonymind", "full", "faculty", "initiative", "comparator"),
    ("full", "base_hermes", "system", None, "treatment"),
    ("full-duty", "full", "drive", "drives", "comparator"),
    ("full-affect-plus-rules", "full", "mixed", None, None),
    ("full", "full", "same", None, None),
])
def test_a_contrast_is_read_as_the_one_faculty_its_arms_differ_in(treatment, comparator, kind, key, faculty_on):
    """Read as a report that does not record the defaults its image shipped: the owner gate's arm over full sets a
    switch full leaves to them, and the gate has shipped both ways, so that contrast is no one faculty's."""
    subject = arms.subject(profile(treatment), profile(comparator))
    assert subject.kind == kind and subject.key == key and subject.faculty_on == faculty_on
    if kind == "mixed":
        assert set(subject.keys) == {"affect", "affect_rules"}
    if kind == "unrecorded":
        assert subject.keys == ["owner_gate"]


def test_a_declared_profile_is_read_by_what_it_switches():
    """The status block's first pilot declared its own profile (an overlay on full): it is the status block."""
    declared = {"name": "full-plus-status", "plugin": True, "overlay": {"COLONYMIND_CONTEXT_STATUS": "on"},
                "full": True}
    unknown = arms.subject(declared, profile("full"))
    assert (unknown.kind, unknown.keys) == ("unrecorded", ["status"])   # it shipped off then, on today
    off = arms.recorded_off({"shipped_defaults": {name: False for name in arms.EARLIER_OFF}})
    subject = arms.subject(declared, profile("full"), off)
    assert (subject.kind, subject.key, subject.faculty_on) == ("faculty", "status", "treatment")
    assert arms.describe(declared, off) == "the mind at its shipped settings, with the status block added"
    assert arms.describe(profile("all-on-skills")) == "the mind with every faculty on, without skills"
    assert arms.describe(profile("base_hermes")) == "stock Hermes"
    assert arms.describe(profile("colonymind")) == "ColonyMind with the mind off"


def test_a_report_is_read_with_the_defaults_it_records():
    """A profile names only what an arm switches; the rest is the shipped defaults of the image that ran it. A
    report that records them is read with them; one that records none (or not all of the switches that have shipped
    both ways) is read both ways, and an arm's name never stands in for the record."""
    pilot = {name: profile(name) for name in ("full", "full-plus-owner_gate")}
    earlier = {"shipped_defaults": {name: False for name in arms.EARLIER_OFF}}
    assert arms.recorded_off(earlier) == arms.EARLIER_OFF
    assert arms.recorded_off({"shipped_defaults": {name: True for name in arms.EARLIER_OFF}}) == frozenset()
    assert arms.recorded_off({"profiles": pilot}) is None
    assert arms.recorded_off({"shipped_defaults": {"owner_gate": False}}) is None
    assert arms.recorded_off({"shipped_defaults": {**earlier["shipped_defaults"], "status": "off"}}) is None
    subject = arms.subject(pilot["full-plus-owner_gate"], pilot["full"], arms.recorded_off(earlier))
    assert (subject.kind, subject.key, subject.faculty_on) == ("faculty", "owner_gate", "treatment")
    assert arms.describe(pilot["full-plus-owner_gate"], {"owner_gate"}) == (
        "the mind at its shipped settings, with the owner gate added")
    assert arms.describe(pilot["full"], {"owner_gate"}) == "the mind at its shipped settings"
    # Arms that set the switch on both sides, or leave it to the defaults on both, read the same either way.
    for treatment, comparator in (("all-on", "all-on-owner_gate"), ("full", "full-consolidation"),
                                  ("all-on", "all-on-initiative"), ("colonymind", "full")):
        assert arms.subject(profile(treatment), profile(comparator)).kind != "unrecorded", treatment
    # full against all-on is skills alone today, and more than one change when five of its switches shipped off.
    assert arms.subject(profile("full"), profile("all-on"), frozenset()).key == "skills"
    assert arms.subject(profile("full"), profile("all-on")).keys == sorted(arms.EARLIER_OFF)


def test_identical_arms_of_a_current_run_are_never_read_as_a_faculty():
    """full-plus-owner_gate turns on a faculty that ships on today, so a run made today runs what full runs. Where
    the "with the gate" arm won every scenario, a report that does not record its image's defaults must not say the
    owner gate helped: it cannot tell today's run from the pilot's, made while the gate shipped off."""
    from pathlib import Path
    from colonymind.qualification.paired_statistics import contrast
    today = {name: profile(name) for name in ("full", "full-plus-owner_gate")}
    assert arms.state(today["full"]) == arms.state(today["full-plus-owner_gate"])
    units = {f"s{index:02d}": (1.0, 0.0) for index in range(20)}
    entry = {"treatment": "full-plus-owner_gate", "comparator": "full", "unit": "scenario", **contrast(units, seed=1)}
    report = {"kind": "paired_report", "statistics": {"rule": {}, "contrasts": [entry]}, "profiles": today,
              "dataset": {"version": "mind-outreach-1", "split": "dev"}}
    item = bench.read(bench.Loaded(Path("report-001.json"), 0.0, report), entry)
    assert item.key is None and item.reading == "unattributed"
    assert bench.verdicts([item])["owner_gate"] == "not measured"
    assert bench.title(item) == "A difference the report cannot place"
    assert bench.compared(item) == ("Compared: the mind at its shipped settings, against the mind at its shipped "
                                    "settings; one of them sets the owner gate itself and the other leaves it as its "
                                    "image shipped.")
    assert "does not record" in bench.meaning(item) and "the owner gate" in bench.meaning(item)
    assert not JARGON.search(" ".join(bench.sentences(item)))
    # A report made today that records its image's defaults reads the two as the same arm: the noise floor.
    shipped = {name: True for name in arms.EARLIER_OFF}
    item = bench.read(bench.Loaded(Path("report-001.json"), 0.0, {**report, "shipped_defaults": shipped}), entry)
    assert item.subject.kind == "same" and item.reading == "unattributed"
    # The owner gate's pilot, which recorded that the gate shipped off, still reads as the owner gate.
    earlier = {name: False for name in arms.EARLIER_OFF}
    item = bench.read(bench.Loaded(Path("report-001.json"), 0.0, {**report, "shipped_defaults": earlier}), entry)
    assert (item.key, item.reading) == ("owner_gate", "helped")


def test_an_overlay_is_a_difference_of_its_own_unless_it_sets_a_faculty():
    """Only the overlays that set a faculty's switch read as that faculty, with the mind on or off (the claim-side
    learning rules run outside the mind); any other overlay one arm carries makes the contrast more than one
    change, so a faculty is never blamed for what the overlay did."""
    declared = {**profile("all-on-lessons"), "name": "all-on-lessons-keywords",
                "overlay": {"COLONYMIND_EMBED_PROVIDER": "skip"}}
    subject = arms.subject(declared, profile("all-on"))
    assert subject.kind == "mixed" and "lessons" in subject.keys
    assert any("COLONYMIND_EMBED_PROVIDER" in key for key in subject.keys)
    both = arms.subject(declared, {**profile("all-on"), "overlay": {"COLONYMIND_EMBED_PROVIDER": "skip"}})
    assert (both.kind, both.key, both.faculty_on) == ("faculty", "lessons", "comparator")
    assert "COLONYMIND_EMBED_PROVIDER" in arms.describe(declared)
    # The learning rules run with the mind off (the served host reads them from its environment, as an install
    # exports them). They ship on today and shipped off before, so an overlay that turns them off is that faculty
    # alone in a report that records its image's defaults, and no one faculty's in one that does not.
    learning = {"name": "colonymind-no-learning", "plugin": True, "overlay": {"COLONYMIND_LEARNING": "0"}}
    subject = arms.subject(learning, profile("colonymind"), frozenset())
    assert (subject.kind, subject.key, subject.faculty_on) == ("faculty", "learning", "comparator")
    subject = arms.subject(learning, profile("colonymind"))
    assert (subject.kind, subject.keys) == ("unrecorded", ["learning"])


def test_repeated_runs_are_counted_as_runs_not_as_scenarios():
    """With repetitions a scenario's pass value is the mean of its runs: four scenarios that each passed one of
    two runs are four passing runs of eight, never "2 of 4 scenarios"."""
    from pathlib import Path
    from colonymind.qualification.paired_statistics import contrast
    units = {f"s{index}": (0.5, 0.0) for index in range(4)}
    entry = {"treatment": "all-on", "comparator": "all-on-skills", "unit": "scenario", **contrast(units, seed=1)}
    report = {"kind": "paired_report", "statistics": {"rule": {}, "contrasts": [entry]},
              "profiles": {name: profile(name) for name in ("all-on", "all-on-skills")},
              "dataset": {"version": "mind-memory-1", "split": "dev", "repetitions": 2}}
    item = bench.read(bench.Loaded(Path("report-001.json"), 0.0, report), entry)
    assert bench.results(item) == ("With skills it passed 4 of 8 runs (50%) over 4 scenarios, 2 runs each; "
                                   "without, 0 of 8 (0%).")
    single = {**report, "dataset": {"version": "mind-memory-1", "split": "dev"}}
    units = {f"s{index}": (1.0 if index < 2 else 0.0, 0.0) for index in range(4)}
    entry = {"treatment": "all-on", "comparator": "all-on-skills", "unit": "scenario", **contrast(units, seed=1)}
    item = bench.read(bench.Loaded(Path("report-001.json"), 0.0, single), entry)
    assert bench.results(item) == "With skills it passed 2 of 4 scenarios (50%); without, 0 of 4 (0%)."


def test_reports_are_found_and_each_plan_is_read_once_in_its_latest_view(tmp_path):
    directory = write_bench_dir(tmp_path / "results")
    reports = bench.load(directory)
    assert sorted(item.report["label"] for item in reports) == ["affect-1", "gate-outreach-1", "initiative-1",
                                                                 "suite-memory-1"]
    suite = next(item for item in reports if item.report["label"] == "suite-memory-1")
    assert suite.path.name == "report-001.json"      # the newer of the two copies of that plan


def test_each_faculty_gets_the_verdict_its_evidence_supports(tmp_path):
    evidence = bench.evidence(bench.load(write_bench_dir(tmp_path / "results")))
    verdicts = bench.verdicts(evidence)
    assert verdicts["skills"] == "helped"                  # 12 scenarios won with it, 2 lost
    assert verdicts["self_narrative"] == "harmful"         # 12.5 points worse, beyond the 10-point margin
    assert verdicts["status"] == "unproven"
    assert verdicts["owner_gate"] == "unproven"
    assert verdicts["affect"] == "unproven" and verdicts["affect_rules"] == "unproven"
    assert verdicts["wishes"] == "not measured"            # its only run had unattributed scenarios
    assert verdicts["goals"] == "not measured"
    skills = next(item for item in evidence if item.key == "skills")
    assert skills.with_passes == 29 and skills.without_passes == 19 and skills.units == 48
    assert skills.difference == pytest.approx(100 * 10 / 48)
    assert skills.lower > 0 and skills.demonstrated and skills.reading == "helped"
    narrative = next(item for item in evidence if item.key == "self_narrative")
    assert narrative.difference == pytest.approx(-12.5) and not narrative.demonstrated
    assert narrative.reading == "harmful" and narrative.strength == "indicated"


def test_the_bench_report_says_what_was_compared_and_what_it_means(tmp_path):
    document = bench.document(bench.load(write_bench_dir(tmp_path / "results")))
    text = document.text()
    assert "Memory scenarios (development set, 48 scenarios)" in text
    assert "Compared: the mind with every faculty on, against the same mind without skills." in text
    assert "With skills it passed 29 of 48 scenarios (60%); without, 19 of 48 (40%)." in text
    assert "a difference of +20.8 points in favour of skills" in text
    assert "95% interval" in text and "unlikely to be chance" in text
    assert "Skills measurably helped here: keep it on." in text
    assert "points to harm" in text and "switch it off: set mind.faculties.self_narrative to false" not in text
    assert "ColonyMind as a whole" in text and "stock Hermes" in text
    assert "no usable result" in text
    summary = document.blocks[1].items
    assert summary[0].startswith("What helped: skills")
    assert summary[1].startswith("Unproven:") and "the status block" in summary[1]
    assert summary[2].startswith("What looked harmful: the self-narrative")
    assert not JARGON.search(text), JARGON.search(text)


def test_html_is_one_page_with_the_same_words(tmp_path):
    from colonymind.reports.text import render
    document = bench.document(bench.load(write_bench_dir(tmp_path / "results")))
    page = render(document, "html")
    assert page.startswith("<!doctype html>") and "<h1>" in page and "prefers-color-scheme: dark" in page
    assert "the same mind without skills" in page


def test_the_cli_renders_a_directory_and_refuses_a_missing_one(tmp_path):
    directory = write_bench_dir(tmp_path / "results")
    command = [sys.executable, "-m", "colonymind", "report", "bench", str(directory)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("# Paired benchmark results")
    out = tmp_path / "bench.html"
    done = subprocess.run([*command, "--format", "html", "--out", str(out)], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0 and out.read_text().startswith("<!doctype html>")
    missing = subprocess.run([sys.executable, "-m", "colonymind", "report", "bench", str(tmp_path / "none")],
                             capture_output=True, text=True, timeout=120)
    assert missing.returncode == 2 and "no such directory" in missing.stderr
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "other.json").write_text(json.dumps({"kind": "other"}))
    done = subprocess.run([sys.executable, "-m", "colonymind", "report", "bench", str(empty)], capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0 and "No paired benchmark reports were found" in done.stdout


def test_the_harder_memory_family_is_evidence_of_its_own(tmp_path):
    """mind-memory-1 and mind-memory-2 are two families: a newer reading on one never hides an older one on the
    other, so harm shown on either wins, while a newer reading on the same family and set replaces the older."""
    from colonymind.qualification.paired_statistics import contrast

    def report(version, split, won, lost, when):
        units = {f"s{index}": ((1.0, 0.0) if index < won else (0.0, 1.0) if index < won + lost else (1.0, 1.0))
                 for index in range(24)}
        entry = {"treatment": "all-on", "comparator": "all-on-skills", "unit": "scenario", **contrast(units, seed=1)}
        value = {"kind": "paired_report", "statistics": {"rule": {}, "contrasts": [entry]},
                 "profiles": {name: profile(name) for name in ("all-on", "all-on-skills")},
                 "dataset": {"version": version, "split": split}, "label": f"{version}-{split}-{when}"}
        return bench.Loaded(tmp_path / f"{version}-{split}-{when}.json", float(when), value)

    harmful = report("mind-memory-1", "dev", 0, 12, when=1)
    helped = report("mind-memory-2", "dev", 12, 0, when=2)
    evidence = bench.evidence([harmful, helped])
    assert [item.reading for item in evidence] == ["helped", "harmful"]
    assert [item.family for item in evidence] == ["Harder memory scenarios (development set)",
                                                  "Memory scenarios (development set)"]
    assert len(bench.latest(evidence, "skills")) == 2
    assert bench.verdict(evidence, "skills") == "harmful"
    # A newer reading on the same family and set replaces the older one; another set is evidence of its own.
    again = report("mind-memory-1", "dev", 12, 0, when=3)
    assert bench.verdict(bench.evidence([harmful, helped, again]), "skills") == "helped"
    heldout = report("mind-memory-1", "heldout", 12, 0, when=4)
    assert bench.verdict(bench.evidence([harmful, heldout]), "skills") == "harmful"
