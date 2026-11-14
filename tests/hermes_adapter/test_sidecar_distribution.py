"""Runtime wheels omit repository tests and retain importable product code."""
import os
from pathlib import Path
import zipfile

import pytest

from conftest import ROOT, run_python


# This test installs the sidecar wheel into a target directory, so it needs a sidecar-compatible interpreter.
@pytest.mark.requires_sidecar
def test_sidecar_wheel_contains_product_without_embedded_tests(tmp_path):
    prepared = os.environ.get("COLONYMIND_DISTRIBUTIONS_DIR")
    distributions = Path(prepared).resolve() if prepared else tmp_path
    if not prepared:
        run_python("-m", "build", "--wheel", "--no-isolation", "--outdir", tmp_path,
                   cwd=ROOT / "sidecar")
    wheel, = distributions.glob("colonymind-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not [name for name in names if any(
            part == "tests" or part.startswith("test_") or part == "conftest.py"
            for part in name.split("/"))]
        for name in ("mind/tick.py", "events/journal.py",
                     "intelligence/components/preference_learner.py",
                     "contacts/affect.py", "devtools/fake_model.py", "devtools/mind_loop_probe.py"):
            assert "colonymind/" + name in names
        for name in ("vision-arrangement.png", "vision-labels.png", "vision-covered-label.png"):
            member = "colonymind/qualification/fixtures/" + name
            assert archive.read(member) == (ROOT / "sidecar" / member).read_bytes()
    installed = tmp_path / "installed"
    run_python("-m", "pip", "install", "--no-deps", "--no-index", "--target", installed,
               wheel, cwd=tmp_path)
    env = {key: os.environ[key] for key in ("PATH", "LANG") if key in os.environ}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(installed), PYTHON_DOTENV_DISABLED="1")
    run_python("-c", "from colonymind.mind.tick import Mind; "
               "from colonymind.events.journal import append_event; "
               "from colonymind.intelligence.components.preference_learner import PreferenceLearner; "
               "from colonymind.contacts.affect import AffectStore; "
               "from colonymind.qualification.cases import select_cases; "
               "assert len(select_cases(['vision'])) == 3",
               cwd=tmp_path, env=env)
