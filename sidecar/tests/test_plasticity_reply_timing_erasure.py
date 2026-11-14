"""Personal reply timing requires erasure even when its learner is unloaded.

This runtime prerequisite uses a numeric probe, not a reply-timing implementation.
The constant probability and source identifiers are fixture assumptions.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from colonymind.plasticity import _io
from colonymind.plasticity.nightly import Runtime, _synthetic
from colonymind.plasticity.numeric_comparator import NumericComparator, configure
from colonymind.plasticity.synthetic import SyntheticSource, make_rows
from test_plasticity_wiring_runtime import NOW, config

ROW = "reply-row-1"
TURN = "owner-turn-1"


class ReferencedSource(SyntheticSource):
    def refs_resting_on(self, ids):
        return [ROW] if TURN in ids else []


def numeric_probe(**kwargs):
    configure(kwargs["controller"], kwargs["declaration"], variant="logistic")
    return _synthetic(**kwargs)


@pytest.mark.parametrize("loaded", [True, False, "omitted", "rejected", "faculty_off"])
@pytest.mark.parametrize("by_source", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_reply_timing_numeric_erasure_does_not_require_a_loaded_driver(tmp_path, loaded, by_source, legacy):
    declaration = {
        "implementation": "numeric_probe", "kind": "numeric-online", "base": "fixture@1",
        "mode": "rebuild", "dependents": ["threshold"], "personal_data": True,
        "erasure_path": "rebuild_from_manifest",
    }
    cfg = config(tmp_path, {"probe": declaration})
    options = {
        "factories": {"numeric_probe": numeric_probe},
        "sources": {"probe": ReferencedSource()},
        "check_learner": lambda name, value: SimpleNamespace(name=name, kind=value["kind"]),
        "clock": lambda: NOW,
    }
    service = Runtime(cfg, **options)
    try:
        assert not service.rejected
        comparator = service.loaded["probe"].comparator
        assert isinstance(comparator, NumericComparator)
        service.serve("probe", ROW, {}, predict=lambda pin, inputs: 0.5)
        service.outcome("probe", ROW, observed=1.0, teacher="realized_value")
        paths = list(comparator.directory.glob("numeric-*.jsonl"))
        assert len(paths) == 1 and paths[0].read_text()
        if legacy:
            catalog = _io.read_json(service.root / "catalog.json")
            catalog["probe"].pop("kind", None)
            _io.write_json(service.root / "catalog.json", catalog)
        if loaded is not True:
            service.close()
            if loaded == "omitted":
                cfg.data["plasticity"]["learners"] = {}
            elif loaded == "rejected":
                cfg.data["plasticity"]["learners"]["probe"]["implementation"] = "unavailable"
            elif loaded == "faculty_off":
                cfg.data["mind"]["faculties"]["plasticity"] = False
            else:
                cfg.data["plasticity"]["learners"]["probe"]["enabled"] = False
            service = Runtime(cfg, **options)
            assert not service.loaded
        if by_source:
            service.erase_sources([TURN])
        else:
            service.erase([ROW])
        # Inspect physical bytes before reloading NumericComparator, whose records()
        # would otherwise apply the tombstone and hide the unloaded erasure failure.
        remaining = [path for path in paths if path.exists() and path.read_text().strip()]
        assert remaining == [], "numeric personal outcomes remain on disk after erasure"
    finally:
        service.close()


class ProvenanceSource(SyntheticSource):
    """Fixture dependency ids cover both the label and the last-turn feature."""

    def source_ids(self, ref):
        return ("label-" + ref, "feature-" + ref)

    def refs_resting_on(self, ids):
        return [row["ref"] for row in self._rows if set(ids).intersection(self.source_ids(row["ref"]))]


@pytest.mark.parametrize("state", ["loaded", "disabled", "omitted", "rejected"])
@pytest.mark.parametrize("dependency", ["label", "feature"])
def test_saved_dependencies_erase_weights_and_rebuild_like_clean_input(tmp_path, state, dependency):
    rows = make_rows(60)
    source = ProvenanceSource(rows)
    spec = {"implementation": "numeric_probe", "kind": "numeric-online", "base": "fixture@1",
            "mode": "rebuild", "dependents": ["threshold"]}
    cfg = config(tmp_path, {"probe": spec.copy()})
    calls = []

    def factory(**kwargs):
        calls.append(True)
        return numeric_probe(**kwargs)

    options = dict(factories={"numeric_probe": factory}, sources={"probe": source}, clock=lambda: NOW,
                   check_learner=lambda name, value: SimpleNamespace(name=name, kind=value["kind"]))
    service = Runtime(cfg, **options)
    try:
        assert service.run()[0]["result"] == "promoted"
        store = service.loaded["probe"].store
        old = store.revisions.serving()
        for ref in ("case-1", "case-2"):
            service.serve("probe", ref, {}, predict=lambda pin, inputs: .5)
            service.outcome("probe", ref, observed=1., teacher="realized_value")
        if state != "loaded":
            service.close()
            if state == "omitted":
                cfg.data["plasticity"]["learners"] = {}
            elif state == "rejected":
                cfg.data["plasticity"]["learners"]["probe"]["implementation"] = "unavailable"
            else:
                cfg.data["plasticity"]["learners"]["probe"]["enabled"] = False
            service = Runtime(cfg, **options)
            assert not service.loaded and len(calls) == 1
        # A committed source deletion precedes the runtime's callback.
        source.erase(["case-1"])
        service.erase_sources([dependency + "-case-1"])
        assert not store.revisions.exists(old.id)
        assert store.revisions.status(old.id).state == "withdrawn"
        assert not any(b'"case-1"' in path.read_bytes() for path in store.directory.rglob("*") if path.is_file())
        # Precisely mapped erasure preserves unrelated numeric outcomes.
        assert sum(len(path.read_text().splitlines()) for path in
                   (store.directory / "comparator").glob("numeric-*.jsonl")) == 1
        if state != "loaded":
            assert store.revisions.serving().mode == "base"
            service.close()
            cfg.data["plasticity"]["learners"] = {"probe": spec.copy()}
            service = Runtime(cfg, **options)
        binding = service.loaded["probe"]
        assert binding.store.revisions.serving().mode == "rebuild"
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        clean = Runtime(config(clean_dir, {"probe": spec.copy()}), **options)
        try:
            assert clean.run()[0]["result"] == "promoted"
            expected = clean.pin("probe", "owner").state
            actual = service.pin("probe", "owner").state
            assert expected.keys() == actual.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
        finally:
            clean.close()
    finally:
        service.close()


def test_unknown_provenance_marks_rebuild_and_unserves_before_factory(tmp_path):
    from colonymind.turns.erasure_hooks import notify_source_erasure

    rows = make_rows(60)
    source = ReferencedSource(rows)
    cfg = config(tmp_path, {"probe": {"implementation": "numeric_probe", "kind": "numeric-online",
        "base": "fixture@1", "mode": "rebuild", "dependents": ["threshold"]}})
    options = dict(factories={"numeric_probe": numeric_probe}, sources={"probe": source}, clock=lambda: NOW,
                   check_learner=lambda name, value: SimpleNamespace(name=name, kind=value["kind"]))
    service = Runtime(cfg, **options)
    try:
        assert service.run()[0]["result"] == "promoted"
        store = service.loaded["probe"].store
        old = store.revisions.serving().id
        # Simulate a legacy catalog without kind and provenance records.
        catalog = _io.read_json(service.root / "catalog.json")
        catalog["probe"].pop("kind", None)
        _io.write_json(service.root / "catalog.json", catalog)
        (store.directory / "source-refs.json").unlink(missing_ok=True)
        service.serve("probe", ROW, {}, predict=lambda pin, inputs: .5)
        service.outcome("probe", ROW, observed=1., teacher="realized_value")
        service.close()
        cfg.data["plasticity"]["learners"] = {}
        service = Runtime(cfg, **options)
        source.erase(["case-1"])
        notify_source_erasure([TURN])
        assert store.revisions.serving().mode == "base"
        assert not store.revisions.exists(old)
        assert not list((store.directory / "comparator").glob("numeric-*.jsonl"))
        assert (store.directory / "erasures" / "REBUILD_REQUIRED").exists()
        service.close()

        def checked_factory(**kwargs):
            assert kwargs["store"].revisions.serving().mode == "base"
            assert not kwargs["store"].revisions.exists(old)
            return numeric_probe(**kwargs)

        cfg.data["plasticity"]["learners"] = {"probe": {"implementation": "numeric_probe",
            "kind": "numeric-online", "base": "fixture@1", "mode": "rebuild", "dependents": ["threshold"]}}
        service = Runtime(cfg, **{**options, "factories": {"numeric_probe": checked_factory}})
        assert service.pin("probe", "owner").closure.mode == "rebuild"
        assert not (store.directory / "erasures" / "REBUILD_REQUIRED").exists()
        assert service.loaded["probe"].last_build.refs == frozenset(row["ref"] for row in rows if row["ref"] != "case-1")
    finally:
        service.close()


@pytest.mark.parametrize("reload", [False, True])
def test_source_erasure_retry_keeps_refs_after_manifest_rewrite(tmp_path, monkeypatch, reload):
    source = ProvenanceSource(make_rows(60))
    # Persisted forward provenance must suffice when no reverse resolver remains.
    source.refs_resting_on = None
    cfg = config(tmp_path, {"probe": {"implementation": "numeric_probe", "kind": "numeric-online",
        "base": "fixture@1", "mode": "rebuild", "dependents": ["threshold"]}})
    options = dict(factories={"numeric_probe": numeric_probe}, sources={"probe": source}, clock=lambda: NOW,
                   check_learner=lambda name, value: SimpleNamespace(name=name, kind=value["kind"]))
    service = Runtime(cfg, **options)
    try:
        assert service.run()[0]["result"] == "promoted"
        binding = service.loaded["probe"]
        old = binding.store.revisions.serving().id
        build = binding.driver.build
        source.erase(["case-1"])

        def interrupted():
            raise OSError("fixture interrupted rebuild")

        monkeypatch.setattr(binding.driver, "build", interrupted)
        with pytest.raises(OSError, match="interrupted rebuild"):
            service.erase_sources(["feature-case-1"])
        assert "case-1" not in binding.store.source_refs()
        assert service.pin("probe", "owner").closure.mode == "base"
        if reload:
            service.close()
            service = Runtime(cfg, **options)
        else:
            monkeypatch.setattr(binding.driver, "build", build)
            assert service.erase_sources(["feature-case-1"])["clean"]
        assert service.pin("probe", "owner").closure.mode == "rebuild"
        assert not binding.store.revisions.exists(old)
        assert not binding.store.erasure_pending()
        assert service._source_erasures() == {}
    finally:
        service.close()


def test_interrupted_unknown_cleanup_finishes_before_driver_reload(tmp_path, monkeypatch):
    source = ReferencedSource(make_rows(60))
    spec = {"implementation": "numeric_probe", "kind": "numeric-online", "base": "fixture@1",
            "mode": "rebuild", "dependents": ["threshold"]}
    cfg = config(tmp_path, {"probe": spec.copy()})
    options = dict(factories={"numeric_probe": numeric_probe}, sources={"probe": source}, clock=lambda: NOW,
                   check_learner=lambda name, value: SimpleNamespace(name=name, kind=value["kind"]))
    service = Runtime(cfg, **options)
    try:
        assert service.run()[0]["result"] == "promoted"
        store = service.loaded["probe"].store
        old = store.revisions.serving().id
        stale = store.directory / "stale-vocabulary.json"
        _io.write_json(stale, {"channel": "erased-channel"})
        service.close()
        cfg.data["plasticity"]["learners"] = {}
        service = Runtime(cfg, **options)
        # The source transaction committed before its erasure callback ran.
        source.erase(["case-1"])
        remove = _io.remove

        def interrupted(path):
            if path == stale:
                raise OSError("fixture interrupted deletion")
            return remove(path)

        with monkeypatch.context() as patch:
            patch.setattr(_io, "remove", interrupted)
            with pytest.raises(OSError, match="interrupted deletion"):
                service.erase_sources([TURN])
        assert store.rebuild_required.exists() and stale.exists()
        service.close()

        def checked_factory(**kwargs):
            assert not stale.exists()
            assert kwargs["store"].revisions.serving().mode == "base"
            assert not kwargs["store"].revisions.exists(old)
            return numeric_probe(**kwargs)

        cfg.data["plasticity"]["learners"] = {"probe": spec.copy()}
        service = Runtime(cfg, **{**options, "factories": {"numeric_probe": checked_factory}})
        assert service.pin("probe", "owner").closure.mode == "rebuild"
        assert not store.rebuild_required.exists()
    finally:
        service.close()


def test_pending_erasure_blocks_old_revision_before_reload_factory(tmp_path, monkeypatch):
    source = ProvenanceSource(make_rows(60))
    cfg = config(tmp_path, {"probe": {"implementation": "numeric_probe", "kind": "numeric-online",
        "base": "fixture@1", "mode": "rebuild", "dependents": ["threshold"]}})
    options = dict(factories={"numeric_probe": numeric_probe}, sources={"probe": source}, clock=lambda: NOW,
                   check_learner=lambda name, value: SimpleNamespace(name=name, kind=value["kind"]))
    service = Runtime(cfg, **options)
    try:
        assert service.run()[0]["result"] == "promoted"
        store = service.loaded["probe"].store
        old = store.revisions.serving().id
        source.erase(["case-1"])
        write = _io.write_json

        def interrupted(path, value):
            write(path, value)
            if path.name == "PENDING":
                raise OSError("fixture interruption before unserve")

        with monkeypatch.context() as patch:
            patch.setattr(_io, "write_json", interrupted)
            with pytest.raises(OSError, match="before unserve"):
                service.erase_sources(["feature-case-1"])
        assert store.revisions.serving().id == old
        assert service.pin("probe", "owner").closure.mode == "base"
        assert service.serve("probe", "request", {}, predict=lambda pin, inputs: pin is None)
        service.close()

        def checked_factory(**kwargs):
            assert kwargs["store"].revisions.serving().mode == "base"
            return numeric_probe(**kwargs)

        service = Runtime(cfg, **{**options, "factories": {"numeric_probe": checked_factory}})
        assert not store.revisions.exists(old)
        assert not store.erasure_pending()
        assert service.pin("probe", "owner").closure.mode == "rebuild"
    finally:
        service.close()
