"""The cut-over to Instinct: one mechanism in place of the three, behaviour-identical.

Explicit synthetic records reproduce the five fast points' 1.11 readings (``owner_verdict`` live, the other four
in shadow); the four typed calls and the owner gate's second opinion are points whose one model tier is the
deliberate call moved unchanged; an outage adds no model call; the ``decisions`` section is carried into
``instinct`` once by ``colonymind upgrade`` and refused afterwards; nothing that grants imports Instinct."""

from __future__ import annotations

import ast
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from colonymind import config
from colonymind.config import ConfigError, load_config
from colonymind.instinct import REGISTRY, calibration
from colonymind.instinct.calibration import calibrate, calibrate_yes
from colonymind.instinct.log import Store
from colonymind.instinct.providers import Answer
from colonymind.instinct.tiers import Instinct

SIDECAR = Path(__file__).resolve().parents[1]
ROOT = SIDECAR.parent
MEASURED = "typed-decisions@1c5edc17a7acd8701df6fc341c0d179f1c62c982"


# -- 2. the import guard -------------------------------------------------------------------------------------

GUARDED = [
    SIDECAR / "colonymind" / "mind" / "authority.py",
    ROOT / "plugins" / "hermes-plugin" / "guard.py",          # the floor patterns live here
    ROOT / "plugins" / "hermes-plugin" / "capture.py",        # capture.SessionMap
    ROOT / "plugins" / "hermes-plugin" / "secrets.py",
    SIDECAR / "colonymind" / "api" / "routers" / "secrets.py",
    SIDECAR / "colonymind" / "api" / "credentials.py",
    SIDECAR / "colonymind" / "contacts" / "identity_links.py",
    SIDECAR / "colonymind" / "contacts" / "changes.py",
    *sorted((SIDECAR / "colonymind" / "secrets").glob("*.py")),
]


def _imports_instinct(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.startswith("colonymind.instinct")]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            if module.startswith("colonymind.instinct") or (module == "colonymind" and "instinct" in names) \
                    or (node.level and (module.split(".")[-1] == "instinct" or "instinct" in names)):
                found.append(module or ".")
    return found


def test_nothing_that_grants_imports_instinct():
    assert all(path.is_file() for path in GUARDED[:8]), [str(path) for path in GUARDED if not path.is_file()]
    assert len(GUARDED) > 8
    for path in GUARDED:
        assert _imports_instinct(path) == [], path


def test_the_guard_sees_every_form_of_the_import(tmp_path):
    for line in ("import colonymind.instinct", "from colonymind import instinct", "from colonymind.instinct import x",
                 "from ..instinct import tiers", "from .. import instinct"):
        probe = tmp_path / "probe.py"
        probe.write_text(line + "\n")
        assert _imports_instinct(probe), line


def test_the_three_old_mechanisms_are_gone():
    assert not (SIDECAR / "colonymind" / "decisions.py").exists()
    assert not (SIDECAR / "colonymind" / "mind" / "decisions.py").exists()
    from colonymind.mind import gate
    for name in ("DecisionClient", "decisions_client"):
        assert not hasattr(gate, name), name
    assert not hasattr(gate.OwnerGate, "_fast")
    source = "\n".join(path.read_text(encoding="utf-8") for path in (SIDECAR / "colonymind").rglob("*.py"))
    assert "COLONYMIND_DECISIONS_POINTS\"]" not in source and "decisions.shared()" not in source


# -- 4. behaviour identity: the moved fast points -----------------------------------------------------------------

# The 1.11 defaults (temperature, abstain band) and whether the point was on: the oracle.
DEFAULTS_1_11 = {"outreach_reply": (0.33, (0.0, 0.55), False), "opt_out": (0.33, (0.15, 0.85), False),
                 "no_reminders": (0.25, (0.1, 0.9), False), "interest_settled": (0.25, (0.1, 0.9), False),
                 "owner_verdict": (0.25, (0.03, 0.97), True)}
FIELDS = {"outreach_reply": {"text": "Who ran it?", "topic": "tidal energy"}, "opt_out": {"text": "Lose my number."},
          "no_reminders": {"item": "Send the form", "text": "No pings about it."},
          "interest_settled": {"topic": "tide tables", "text": "I'm sorted there."},
          "owner_verdict": {"reply": "The order code is C4411.", "text": "That was wrong."}}
ANSWERS = {
    "outreach_reply": {"sure": {"engaged": 0.02, "dig_deeper": 0.9, "not_interested": 0.03, "not_now": 0.03,
                                "stop": 0.02},
                       "unsure": {"engaged": 0.3, "dig_deeper": 0.3, "not_interested": 0.2, "not_now": 0.1,
                                  "stop": 0.1}},
    **{name: {"sure": {"yes": 0.99, "no": 0.01}, "no": {"yes": 0.01, "no": 0.99},
              "unsure": {"yes": 0.5, "no": 0.5}} for name in ("opt_out", "no_reminders", "interest_settled",
                                                               "owner_verdict")},
}


def oracle(point, raw, *, enabled):
    """What 1.11's ``Decider`` applied: nothing when off or without an answer; else the band's reading."""
    temperature, (low, high), _ = DEFAULTS_1_11[point]
    if not enabled or raw is None:
        return None
    if point == "outreach_reply":
        probabilities = calibrate(raw, temperature)
        label = max(probabilities, key=probabilities.get)
        return label if probabilities[label] >= high else None
    p_yes = calibrate_yes(raw["yes"], temperature)
    return "yes" if p_yes >= high else "no" if p_yes <= low else None


class Provider:
    kind, model, private, max_state_chars, batch, timeout_ms = "local-decision.v1", "", True, 1200, False, 250
    types = frozenset({"yes_no", "choice"})

    def __init__(self, raw, *, revision=MEASURED, fail=False):
        self.id, self.raw, self.revision, self.fail, self.calls = "local", raw, revision, fail, 0

    async def answer(self, state, questions):
        self.calls += 1
        if self.fail:
            raise httpx.ConnectError("refused")
        return Answer({name: dict(self.raw) for name in questions}, revision=self.revision)


def engine_with(provider, *, live=(), store=None):
    # The historical cutover comparison uses fixtures, not today's shipped letter calibration.
    records = [calibration.Record(
        point=name, version=REGISTRY[name].version, tier="fast", provider_kind="local-decision.v1",
        revision=revision, temperature=temperature,
        bars=calibration.band_bars(REGISTRY[name].labels, REGISTRY[name].type, band), source="synthetic cutover",
    ) for name, (temperature, band, _) in DEFAULTS_1_11.items() for revision in calibration.MEASURED_REVISIONS]
    made = Instinct(providers={"local": provider} if provider is not None else {},
                    fast="local" if provider is not None else "", store=store, records=records)
    for point in live:
        made.set_state(point, "fast", "live", source="test")
    return made


@pytest.mark.parametrize("point", sorted(DEFAULTS_1_11))
@pytest.mark.parametrize("behaviour", ["sure", "no", "unsure", "failing", "absent"])
@pytest.mark.parametrize("enabled", [False, True], ids=["shipped", "enabled_in_1_11"])
async def test_each_moved_point_applies_what_1_11_applied(point, behaviour, enabled):
    """The original states and explicit synthetic calibration reproduce the 1.11 decider's behavior, including
    a point the upgrade made live. Today's shipped letter calibration is checked separately."""
    if behaviour == "no" and point == "outreach_reply":
        return
    raw = None if behaviour in {"failing", "absent"} else ANSWERS[point][behaviour]
    provider = None if behaviour == "absent" else Provider(raw or {}, fail=behaviour == "failing")
    on = enabled or DEFAULTS_1_11[point][2]
    made = engine_with(provider, live=[point] if enabled else [])
    decision = await made.decide(point, FIELDS[point], {"turn": "t-1"})
    await made.drain()
    assert (decision.label if decision.applied else None) == oracle(point, raw, enabled=on)


async def test_the_shipped_states_are_preserved_and_only_measured_letter_records_apply():
    made = Instinct()
    assert {name: made.state(name, "fast") for name in DEFAULTS_1_11} == {
        "outreach_reply": "shadow", "opt_out": "shadow", "no_reminders": "shadow", "interest_settled": "shadow",
        "owner_verdict": "live"}
    assert made.state("owner_send", "fast") == "shadow"
    assert {item.point for item in made.shipped_records} == {"opt_out"}
    for item in made.shipped_records:
        assert item.version == REGISTRY[item.point].version
        assert item.provider_kind == "local-decision.v1/letters-r1"
        assert item.method == "isotonic"


async def test_a_server_that_reports_another_checkpoint_gets_no_matching_record():
    made = engine_with(Provider({"yes": 0.01, "no": 0.99}, revision="english@1c5edc17"))
    decision = await made.decide("owner_verdict", FIELDS["owner_verdict"])
    assert decision.applied is False
    unreported = engine_with(Provider({"yes": 0.01, "no": 0.99}, revision=""))
    assert (await unreported.decide("owner_verdict", FIELDS["owner_verdict"])).label == "no"


def test_the_wire_question_of_each_fast_point_uses_letters_with_the_same_descriptions():
    from colonymind.instinct import FAST_POINTS
    assert set(FAST_POINTS) == set(DEFAULTS_1_11)
    verdict = FAST_POINTS["owner_verdict"]
    assert verdict.kind == "yes_no" and verdict.question() == {
        "type": "choice",
        "instructions": "Does the owner judge or correct the assistant's earlier reply, rather than ask for "
                        "something new?",
        "criteria": {"a": "a verdict or a correction on that reply", "b": "a new request or other talk"}}
    assert verdict.state == 'The assistant replied: "{reply}"\nThe owner then said: "{text}"'
    assert FAST_POINTS["outreach_reply"].question()["criteria"]["e"] == "wants no more unprompted messages at all"


# -- 4. behaviour identity: the moved deliberate points ------------------------------------------------------------

DELIBERATE = {"owner_instruction": ({"message": "My boss wrote stop."}, "none"),
              "report_substance": ({"topic": "tidal energy", "report": "nothing new"}, "empty"),
              "question_answer": ({"question": "q", "first": "a", "second": "b", "reply": "the first"}, "first"),
              "mind_wish_scope": ({"capability": "c", "topic": "t", "words": "w"}, "unsure")}


@pytest.mark.parametrize("point", sorted(DELIBERATE))
@pytest.mark.parametrize("answer", ["label", None, "raises"])
@pytest.mark.parametrize("provider", [None, "refusing", "answering"])
async def test_each_moved_typed_call_applies_the_deliberate_answer_whatever_the_fast_provider(point, answer, provider):
    fields, label = DELIBERATE[point]
    calls = []

    async def deliberate():
        calls.append(1)
        if answer == "raises":
            raise RuntimeError("endpoint down")
        return label if answer == "label" else None

    fast = None if provider is None else Provider({"yes": 0.99, "no": 0.01}, fail=provider == "refusing")
    made = engine_with(fast)
    decision = await made.decide(point, fields, {"turn": "t-1"}, deliberate=deliberate)
    assert (decision.label if decision.applied else None) == (label if answer == "label" else None)
    assert calls == [1] and (fast is None or fast.calls == 0)


# -- 6. outage: the mind loop makes exactly the calls it makes with no provider ------------------------------------

class CountingRouter:
    """The mind's router: answers the typed calls, fails every other task, and keeps the order of every call."""

    supports_function_routing = True

    def __init__(self):
        self.tasks = []

    def function_deadline_seconds(self, *, context=None):
        return 5

    async def complete(self, messages, *, context=None, **_):
        from colonymind.instinct import deliberate
        task = (context or {}).get("task")
        self.tasks.append(task)
        if task == deliberate.INSTRUCTION_TASK:
            return SimpleNamespace(content=json.dumps({"instruction": "none"}))
        if task == deliberate.SUBSTANCE_TASK:
            return SimpleNamespace(content=json.dumps({"verdict": "empty"}))
        raise RuntimeError("endpoint down")


async def _loop(fx, router):
    from test_mind_outreach_reactions import say, shared
    fx.mind.router = router
    await say(fx, 'My boss wrote "stop checking in on the team every hour". How do I answer that?', "t-1")
    fx.shift(timedelta(minutes=30))
    await shared(fx)
    fx.shift(timedelta(minutes=5))
    await say(fx, "Ah, the tidal energy item you sent reminds me of a trip I took.", "t-2", "owner-2")
    fx.owner_spoke()
    fx.mind.add_interest("moss lawns", by="turn:t-3")
    done = fx.found("moss lawns", "finding: Moss lawns: nothing groundbreaking, sadly")
    fx.shift(timedelta(minutes=30))
    await fx.tick()
    return done


async def test_with_the_fast_provider_refusing_the_mind_loop_makes_exactly_the_calls_it_makes_without_one(tmp_path,
                                                                                                          monkeypatch):
    from test_mind_outreach_loop import OWNER, Fx
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    runs = []
    for name, provider in (("none", None), ("refusing", Provider({}, fail=True))):
        directory = tmp_path / name
        directory.mkdir()
        fx = Fx(directory, config={"faculties": {"owner_gate": True}})
        made = engine_with(provider, live=list(DEFAULTS_1_11), store=Store(directory / "instinct.db"))
        fx.mind.instinct = fx.mind.owner_gate.instinct = fx.mind.lessons.instinct = made
        fx.mind.questions.instinct = made
        router = CountingRouter()
        await _loop(fx, router)
        await made.drain()
        runs.append((router.tasks, fx.outreach_rows(), provider))
        fx.close()
    (plain, rows_plain, _), (refused, rows_refused, provider) = runs
    assert plain == refused and "mind_owner_instruction" in plain and "mind_report_substance" in plain
    assert [row.status for row in rows_plain] == [row.status for row in rows_refused]
    assert 1 <= provider.calls <= 3                               # the breaker opened after three failures


# -- 10. configuration: the upgrade carries decisions into instinct once ----------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    for name in ("COLONYMIND_HOME", "COLONYMIND_STATE_DIR", "COLONYMIND_INSTINCT", "COLONYMIND_DECISIONS_URL",
                 "COLONYMIND_DECISIONS_TIMEOUT_MS", "COLONYMIND_DECISIONS_POINTS"):
        monkeypatch.delenv(name, raising=False)
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


def _write(home, data):
    (home / config.CONFIG_FILE).write_text(yaml.safe_dump(data))


def test_the_upgrade_carries_a_1_11_decisions_section_exactly(home):
    from colonymind.init import decisions_pending, upgrade_decisions
    _write(home, {"sidecar": {"port": 7790}, "decisions": {
        "url": "http://192.0.2.1:8080", "timeout_ms": 300,
        "points": {"opt_out": {"enabled": True, "temperature": 2.0, "abstain": [0.1, 0.95]},
                   "owner_verdict": {"enabled": False}, "no_reminders": {"abstain": [0.2, 0.8]}}}})
    assert decisions_pending(home)
    notes = upgrade_decisions(home)
    raw = yaml.safe_load((home / config.CONFIG_FILE).read_text())
    assert "decisions" not in raw and raw["sidecar"] == {"port": 7790}
    assert raw["instinct"] == {"providers": {"local": {"kind": "local-decision.v1", "url": "http://192.0.2.1:8080",
                                                       "timeout_ms": 300}},
                               "fast": "local", "reader": "", "points": {}, "external": {}}
    store = Store(home / "instinct.db")
    assert store.states() == {("opt_out", "fast"): "live", ("owner_verdict", "fast"): "shadow"}
    carried = store.records()
    # One record per revision the measured defaults applied to: the measured checkpoint and no revision reported.
    assert sorted((item.point, item.revision) for item in carried) == [
        ("no_reminders", ""), ("no_reminders", MEASURED), ("opt_out", ""), ("opt_out", MEASURED)]
    records = {item.point: item for item in carried}
    assert records["opt_out"].temperature == 2.0 and records["opt_out"].bars == {"yes": 0.95,
                                                                                 "no": pytest.approx(0.9)}
    assert records["no_reminders"].temperature == 1.0    # no named measured default remains
    assert records["no_reminders"].bars == {"yes": 0.8, "no": pytest.approx(0.8)}
    assert all(item.version == REGISTRY[item.point].version for item in carried)
    assert any("instinct" in note for note in notes)
    assert not decisions_pending(home) and upgrade_decisions(home) == []
    assert load_config(home).get("instinct.providers.local.url") == "http://192.0.2.1:8080"


@pytest.mark.parametrize("revision, applied", [(MEASURED, True), ("", True), ("another-checkpoint@0f0f", False)])
async def test_a_carried_record_applies_only_to_the_measured_checkpoint(home, revision, applied):
    """A 1.11 temperature and band tuned the measured defaults: carried, they apply where those defaults apply, and a
    server that reports another checkpoint gets no record, so owner_verdict's live fast tier abstains (no veto of an
    owner correction on an unmeasured checkpoint) while its raw answer is still logged."""
    from colonymind.init import upgrade_decisions
    _write(home, {"decisions": {"url": "http://127.0.0.1:9", "points": {
        "owner_verdict": {"enabled": True, "temperature": 0.5, "abstain": [0.2, 0.8]}}}})
    upgrade_decisions(home)
    store = Store(home / "instinct.db")
    provider = Provider({"yes": 0.05, "no": 0.95}, revision=revision)
    made = Instinct(providers={"local": provider}, fast="local", store=store)
    decision = await made.decide("owner_verdict", {"reply": "Done.", "text": "that was wrong"})
    assert (decision.label, decision.applied) == (("no", True) if applied else (None, False))
    fast = store.rows(limit=1)[0]["tiers"]["fast"]
    assert fast["outcome"] == ("settled" if applied else "no_record") and fast["raw"]["no"] == pytest.approx(0.95)
    if applied:                                   # the carried record, not the shipped one: T 0.5, no at 0.8
        assert fast["record"]["source"] == "carried from the 1.11 decisions section"


@pytest.mark.parametrize("value, state", [("false", "shadow"), ("off", "shadow"), (0, "shadow"), ("no", "shadow"),
                                          (" False ", "shadow"), ("true", "live"), ("on", "live"), (1, "live"),
                                          ("yes", "live"), (False, "shadow"), (True, "live")])
def test_the_upgrade_reads_enabled_as_the_1_11_loader_did(home, value, state):
    """1.11's loader read enabled as a boolean or a word or number for one ("false", "off", 0): the carry reads it the
    same way, so a veto the owner switched off stays off and a point they switched on goes live."""
    from colonymind.init import upgrade_decisions
    _write(home, {"decisions": {"url": "http://127.0.0.1:9", "points": {"owner_verdict": {"enabled": value},
                                                                        "opt_out": {"enabled": value}}}})
    upgrade_decisions(home)
    assert Store(home / "instinct.db").states() == {("owner_verdict", "fast"): state, ("opt_out", "fast"): state}


def test_the_upgrade_refuses_an_enabled_the_1_11_loader_refused_and_changes_nothing(home):
    from colonymind.init import upgrade_decisions
    data = {"decisions": {"points": {"owner_verdict": {"enabled": "maybe"}}}}
    _write(home, data)
    with pytest.raises(ConfigError, match="decisions.points.owner_verdict.enabled"):
        upgrade_decisions(home)
    assert yaml.safe_load((home / config.CONFIG_FILE).read_text()) == data
    assert not (home / "instinct.db").exists()


def test_the_old_variables_are_not_read_and_the_instinct_section_is_exported(home, monkeypatch):
    _write(home, {"instinct": {"providers": {"local": {"kind": "local-decision.v1", "url": "http://127.0.0.1:9/",
                                                       "timeout_ms": 300}}}})
    loaded = load_config(home, environ={})
    environ = {"COLONYMIND_DECISIONS_URL": "http://192.0.2.1:1"}
    config.apply_environment(loaded, environ=environ)
    exported = json.loads(environ["COLONYMIND_INSTINCT"])
    assert exported["providers"]["local"]["url"] == "http://127.0.0.1:9/" and exported["fast"] == "local"
    made = Instinct.from_environment(environ, state_dir=home)
    assert made.provider_for("fast").url == "http://127.0.0.1:9" and made.provider_for("fast").timeout_ms == 300
    only_old = Instinct.from_environment({"COLONYMIND_DECISIONS_URL": "http://192.0.2.1:1"}, state_dir=home)
    assert only_old.provider_for("fast") is None


def test_doctor_names_a_retired_decisions_variable_still_set(home, monkeypatch):
    from colonymind import doctor
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    _write(home, {})
    monkeypatch.setenv("COLONYMIND_DECISIONS_URL", "http://192.0.2.1:8080")
    result = doctor.check_instinct()
    assert result.status == doctor.WARN and "COLONYMIND_DECISIONS_URL" in result.detail
    assert "192.0.2.1" not in result.detail and "nothing reads" in result.detail
    monkeypatch.delenv("COLONYMIND_DECISIONS_URL")
    result = doctor.check_instinct()
    assert result.status == doctor.PASS and "no fast provider" in result.detail


def test_colonymind_upgrade_carries_a_decisions_section_the_loader_refuses(tmp_path, monkeypatch, capsys):
    """An instance a 1.11 init wrote (a decisions section, as every save wrote one): the loader refuses it, and
    ``colonymind upgrade`` takes its backup, carries the section and completes; a second upgrade has nothing to do."""
    upgrade = pytest.importorskip("test_upgrade")           # stock Hermes, as the other upgrade tests
    from colonymind import init
    from test_init import STOCK_CONFIG, _args
    home, hermes_home = tmp_path / "colonymind", tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(STOCK_CONFIG, sort_keys=False))
    for name, value in (("COLONYMIND_HOME", home), ("COLONYMIND_STATE_DIR", home), ("HERMES_HOME", hermes_home)):
        monkeypatch.setenv(name, str(value))
    for name in ("COLONYMIND_API_KEY", "COLONYMIND_INSTINCT"):
        monkeypatch.delenv(name, raising=False)
    assert init.run_init(_args(home, hermes_home)) == 0
    assert init.run_upgrade(upgrade._upgrade_args(home)) == 0
    capsys.readouterr()
    raw = yaml.safe_load((home / config.CONFIG_FILE).read_text())
    raw.pop("instinct")
    raw["decisions"] = {"url": "http://127.0.0.1:9", "timeout_ms": 300, "points": {"opt_out": {"enabled": True}}}
    (home / config.CONFIG_FILE).write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ConfigError, match="colonymind upgrade"):
        load_config(home)
    assert init.run_upgrade(upgrade._upgrade_args(home)) == 0
    output = capsys.readouterr().out
    assert "backup taken" in output and "the decisions section is now instinct" in output
    assert load_config(home).get("instinct.providers.local") == {"kind": "local-decision.v1",
                                                                 "url": "http://127.0.0.1:9", "timeout_ms": 300}
    assert Store(home / "instinct.db").states() == {("opt_out", "fast"): "live"}
    [backup] = [path for path in (home / "backups").iterdir() if (path / config.CONFIG_FILE).is_file()
                and "decisions" in yaml.safe_load((path / config.CONFIG_FILE).read_text())]
    assert init.run_upgrade(upgrade._upgrade_args(home)) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_colonymind_upgrade_stops_the_writers_before_it_backs_up_and_carries_the_decisions_section(tmp_path,
                                                                                               monkeypatch, capsys):
    """The service runs while ``colonymind upgrade`` starts: the carried section is prepared in memory, the service is
    stopped, and only then are the backup taken and the migration written, so the backup holds every store as the
    stopped writers left it."""
    upgrade = pytest.importorskip("test_upgrade")
    from colonymind import init
    from test_init import STOCK_CONFIG, _args
    home, hermes_home = tmp_path / "colonymind", tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(STOCK_CONFIG, sort_keys=False))
    for name, value in (("COLONYMIND_HOME", home), ("COLONYMIND_STATE_DIR", home), ("HERMES_HOME", hermes_home)):
        monkeypatch.setenv(name, str(value))
    for name in ("COLONYMIND_API_KEY", "COLONYMIND_INSTINCT"):
        monkeypatch.delenv(name, raising=False)
    assert init.run_init(_args(home, hermes_home)) == 0
    assert init.run_upgrade(upgrade._upgrade_args(home)) == 0
    capsys.readouterr()
    raw = yaml.safe_load((home / config.CONFIG_FILE).read_text())
    raw.pop("instinct")
    raw["decisions"] = {"url": "http://127.0.0.1:9", "points": {"owner_verdict": {"enabled": False}}}
    (home / config.CONFIG_FILE).write_text(yaml.safe_dump(raw, sort_keys=False))
    backups_before = set((home / "backups").iterdir()) if (home / "backups").exists() else set()
    events = []
    running = [True]

    def stop(cfg):
        events.append(("stop", "decisions" in yaml.safe_load((home / config.CONFIG_FILE).read_text()),
                       set((home / "backups").iterdir()) - backups_before if (home / "backups").exists() else set(),
                       Store(home / "instinct.db").states()))
        running[0] = False
        return True

    real_backup = init.backup_instance

    def backup(directory):
        events.append(("backup", running[0]))
        return real_backup(directory)

    monkeypatch.setattr(init, "service_running_for_upgrade", lambda cfg: running[0])
    monkeypatch.setattr(init, "stop_service_for_upgrade", stop)
    monkeypatch.setattr(init, "backup_instance", backup)
    monkeypatch.setattr(init, "start_service_after_upgrade", lambda cfg: "sidecar service started again")
    assert init.run_upgrade(upgrade._upgrade_args(home)) == 0
    assert [event[0] for event in events] == ["stop", "backup"]
    _, decisions_left, new_backups, states = events[0]
    assert decisions_left and new_backups == set() and states == {}        # nothing written before the stop
    assert events[1] == ("backup", False)
    [taken] = set((home / "backups").iterdir()) - backups_before
    assert "decisions" in yaml.safe_load((taken / config.CONFIG_FILE).read_text())
    assert Store(home / "instinct.db").states() == {("owner_verdict", "fast"): "shadow"}
    assert load_config(home).get("instinct.providers.local.url") == "http://127.0.0.1:9"
