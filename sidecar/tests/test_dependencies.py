"""The documented install (``pipx install colonymind``) must produce a sidecar with a vector store."""
import tomllib
from pathlib import Path

SIDECAR = Path(__file__).resolve().parents[1] / "pyproject.toml"
ADAPTER = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_adapter_declares_the_same_hermes_range_as_attachment():
    from colonymind.init import SUPPORTED_HERMES_RANGE
    declared = tomllib.loads(ADAPTER.read_text())["tool"]["colonymind"]["requires-hermes"]
    assert declared == SUPPORTED_HERMES_RANGE


def _requirement_names(items):
    names = set()
    for item in items:
        name = item.split(";")[0].strip()
        for separator in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(separator)[0]
        names.add(name.lower())
    return names


def test_vector_store_is_a_base_dependency_not_an_extra():
    project = tomllib.loads(SIDECAR.read_text())["project"]
    base = _requirement_names(project["dependencies"])
    assert {"lancedb", "pyarrow", "pandas"} <= base
    extras = project.get("optional-dependencies", {})
    assert "lancedb" not in extras, "a redundant extra would keep the old two-step install alive"
    # In-process models stay optional: a remote embeddings endpoint needs none of them.
    assert "sentence-transformers" not in base
    assert "sentence-transformers" in _requirement_names(extras["vectors"])


def test_sidecar_and_adapter_agree_on_the_interpreter_range():
    """The adapter runs inside Hermes' interpreter (3.14 for the 0.21.7 line), the sidecar in its own."""
    import yaml
    from packaging.specifiers import SpecifierSet

    sidecar = tomllib.loads(SIDECAR.read_text())["project"]["requires-python"]
    adapter = tomllib.loads(ADAPTER.read_text())["project"]["requires-python"]
    assert (sidecar, adapter) == (">=3.11,<3.14", ">=3.11,<3.15")
    jobs = yaml.safe_load((ADAPTER.parent / ".github/workflows/ci.yml").read_text())["jobs"]
    sidecar_ci = set(jobs["python"]["strategy"]["matrix"]["python-version"])
    adapter_ci = set(jobs["hermes-adapter"]["strategy"]["matrix"]["python-version"])
    adapter_ci.update(step["with"]["python-version"] for step in jobs["hermes-adapter-py314"]["steps"]
                      if step.get("uses", "").startswith("actions/setup-python@"))
    candidates = {f"3.{minor}" for minor in range(10, 16)}
    assert {version for version in candidates if version in SpecifierSet(sidecar)} == sidecar_ci
    assert {version for version in candidates if version in SpecifierSet(adapter)} == adapter_ci


def _floor(requirement):
    """The oldest release a requirement admits (its ``>=``), or None."""
    from packaging.requirements import Requirement
    floors = [spec.version for spec in Requirement(requirement).specifier if spec.operator == ">="]
    return floors[0] if floors else None


def test_the_vectors_extra_installs_what_every_pinned_local_model_needs():
    """``pip install 'colonymind[vectors]'`` gives the local embedder libraries it can load every pinned model with:
    the extra (and the dev list the tests run with) declares each library a pin requires at its release or later,
    so init and doctor never refuse an environment the documented install just made."""
    from packaging.version import Version

    from colonymind.vector.local_embedder import LIBRARY_FLOOR, PINNED
    extras = tomllib.loads(SIDECAR.read_text())["project"]["optional-dependencies"]
    needed = dict(LIBRARY_FLOOR)
    for pin in PINNED.values():
        for name, oldest in pin.requires:
            needed[name] = max(needed.get(name, oldest), oldest, key=Version)
    for extra in ("vectors", "dev"):
        declared = {}
        for item in extras[extra]:
            name = next(iter(_requirement_names([item])))
            declared[name] = _floor(item)
        for name, oldest in needed.items():
            assert declared.get(name) is not None, f"the {extra} list does not declare {name} >= {oldest}"
            assert Version(declared[name]) >= Version(oldest), (extra, name, declared[name], oldest)
