"""M9 deletions: the dormant learning machinery lessons and Hermes skills replace is gone.

Nothing imports the removed packages, no route serves them, no client calls those routes, no
capability advertises them, no environment flag names them, and the state files and tables they
left behind move into the upgrade backup.
"""

from importlib.util import find_spec
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PACKAGE = ROOT / "colonymind"

RETIRED_MODULES = (
    "colonymind.toolsmith",
    "colonymind.skills",          # the whole package: registry, executor, sandbox runner, synthesis, packager
    "colonymind.self_model.experiments",
    "colonymind.self_model.params",
    "colonymind.intelligence.cognition",     # MetaLearner, CPI, StrategyAdjuster, gap detector
    "colonymind.skills_memory",
    "colonymind.mining",
    "colonymind.sandbox",                    # the exploration sandbox; the qualification coding sandbox stays
    "colonymind.api.routers.mining",
    "colonymind.self_model.trust",           # the trust ladder; the floor and breaker are the mind's (authority.py)
    "colonymind.self_model.supervised",
)
RETIRED_ROUTE_PREFIXES = ("/v1/host/self/tools", "/v1/host/skills/", "/v1/host/self/experiments",
                          "/v1/host/self/params", "/v1/host/skills-memory", "/v1/host/sandbox", "/v1/host/mining")
RETIRED_ENV = ("COLONYMIND_TOOLSMITH", "COLONYMIND_EXPERIMENTS_", "COLONYMIND_EXPERIMENT_PREGRANTS_JSON",
               "COLONYMIND_SKILLS_DISTILL", "COLONYMIND_ESCALATION_MINING", "COLONYMIND_CORPUS_EXPORT_ENABLED",
               "COLONYMIND_SANDBOX_", "COLONYMIND_TRUST_", "COLONYMIND_SUPERVISED_LIVE_DOMAINS")
RETIRED_CAPABILITIES: tuple = ("skills", "skill_sandbox", "security_scanner", "cognition")


def absent(module: str) -> bool:
    try:
        return find_spec(module) is None
    except ModuleNotFoundError:
        return True


@pytest.mark.parametrize("module", RETIRED_MODULES)
def test_the_m9_retired_modules_do_not_exist(module):
    assert absent(module), f"{module} still exists"


def test_no_retired_route_is_served():
    from colonymind.server import create_app
    paths = set(create_app().openapi().get("paths", {}))
    assert "/v1/host/context/assemble" in paths
    assert sorted(path for path in paths if path.startswith(RETIRED_ROUTE_PREFIXES)) == []


def test_no_client_calls_a_retired_route():
    """Plugins, scripts, benchmarks, the MCP server and the e2e suites call only served routes."""
    route = re.compile("|".join(re.escape(prefix) + r"\b" for prefix in RETIRED_ROUTE_PREFIXES))
    roots = [REPO / "plugins", REPO / "scripts", REPO / "benchmarks", REPO / "tests", ROOT / "scripts",
             ROOT / "tests" / "e2e", ROOT / "tests" / "integration", PACKAGE / "mcp"]
    hits = []
    for base in roots:
        for path in sorted(base.rglob("*.py")) if base.is_dir() else ():
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if route.search(line):
                    hits.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert hits == []


def test_no_source_line_imports_a_retired_module():
    pattern = re.compile("|".join(re.escape(module) + r"\b" for module in RETIRED_MODULES))
    hits = []
    for base in (PACKAGE, ROOT / "tests", REPO / "plugins", REPO / "scripts", REPO / "tests"):
        for path in sorted(base.rglob("*.py")):
            if path.name == Path(__file__).name:
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert hits == []


def test_no_environment_flag_names_a_retired_subsystem():
    text = (PACKAGE / "server.py").read_text()
    assert [name for name in RETIRED_ENV if name in text] == []


def test_health_capabilities_never_advertise_the_m9_subsystems():
    from colonymind.api.routers.host import supported_capabilities
    assert not set(RETIRED_CAPABILITIES) & set(supported_capabilities())


M9_RETIRED_STATE = ("colonymind-toolsmith.db", "toolsmith_library", "colonymind-experiments.db", "colonymind-params.db",
                    "colonymind-skills.db", "colonymind-mining.db")
M9_RETIRED_TABLES = {"colonymind-self-model.db": ("trust_stage", "trust_notices")}


def test_upgrade_retires_the_m9_stores_and_the_trust_tables(tmp_path):
    """An upgrade backs up first, moves the removed subsystems' stores into the backup and drops the trust
    tables from the surviving self-model store (the competence tables stay); a second pass changes nothing."""
    import sqlite3
    from colonymind import init
    assert set(M9_RETIRED_STATE) <= set(init.RETIRED_STATE)
    assert set(M9_RETIRED_TABLES["colonymind-self-model.db"]) <= set(init.RETIRED_TABLES["colonymind-self-model.db"])
    home = tmp_path / "home"
    home.mkdir()
    for name in M9_RETIRED_STATE:
        if name == "toolsmith_library":
            (home / name / "t-1").mkdir(parents=True)
            (home / name / "t-1" / "tool.py").write_text("def run():\n    return 1\n")
            continue
        with sqlite3.connect(home / name) as db:
            db.execute("CREATE TABLE t (x)")
            db.execute("INSERT INTO t VALUES (1)")
    (home / "colonymind-mining.db-wal").write_bytes(b"")
    with sqlite3.connect(home / "colonymind-self-model.db") as db:
        db.execute("CREATE TABLE competence (domain TEXT PRIMARY KEY, success INTEGER)")
        db.execute("INSERT INTO competence VALUES ('worker:x', 3)")
        for table in ("trust_stage", "trust_notices"):
            db.execute(f"CREATE TABLE {table} (domain TEXT, value TEXT)")
            db.execute(f"INSERT INTO {table} VALUES ('worker:x', 'act_first')")
    (home / "exports").mkdir()
    (home / "exports" / "corpus.jsonl").write_text("{}\n")
    assert sorted(init.retired_state_present(home)) == sorted(M9_RETIRED_STATE)
    assert sorted(init.retired_tables_present(home)) == [
        "colonymind-self-model.db:trust_notices", "colonymind-self-model.db:trust_stage"]

    backup = init.backup_instance(home)                      # what run_upgrade does first
    notes = init.retire_state(home, backup) + init.retire_tables(home)

    assert init.retired_state_present(home) == [] and init.retired_tables_present(home) == []
    assert len(notes) == len(M9_RETIRED_STATE) + 2
    retired = backup / "retired"
    assert sorted(path.name for path in retired.iterdir()) == sorted(M9_RETIRED_STATE + ("colonymind-mining.db-wal",))
    assert (retired / "toolsmith_library" / "t-1" / "tool.py").exists()
    with sqlite3.connect(retired / "colonymind-skills.db") as db:
        assert db.execute("SELECT x FROM t").fetchall() == [(1,)]
    with sqlite3.connect(backup / "colonymind-self-model.db") as db:          # the backup keeps the rows
        assert db.execute("SELECT value FROM trust_stage").fetchall() == [("act_first",)]
    with sqlite3.connect(home / "colonymind-self-model.db") as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert names == {"competence"} and db.execute("SELECT success FROM competence").fetchone() == (3,)
    assert (home / "exports" / "corpus.jsonl").exists()                    # the owner's exports stay
    # A second pass finds nothing.
    assert init.retire_state(home, tmp_path / "again") == [] and init.retire_tables(home) == []
    assert not (tmp_path / "again").exists()
