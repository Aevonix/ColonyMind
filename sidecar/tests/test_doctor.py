"""``colonymind doctor``: the sidecar checks read the served health and fail loudly."""
from __future__ import annotations

import plistlib

import pytest
import yaml

from colonymind import doctor
from colonymind.doctor import FAIL, PASS, SKIP, WARN, check_semantic_recall, check_sidecar


def _serve(monkeypatch, routes):
    """Answer ``_http_get`` from a mapping of path suffix to (status, body)."""
    calls = []

    def fake_get(url, api_key="", timeout=10.0):
        calls.append(url)
        for suffix, answer in routes.items():
            if url.endswith(suffix):
                return answer
        return 404, {"detail": "not found"}

    monkeypatch.setattr(doctor, "_http_get", fake_get)
    return calls


def _instance(tmp_path, monkeypatch, router):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"router": router}))


def test_sidecar_check_repeats_the_problems_in_words(monkeypatch):
    _serve(monkeypatch, {
        "/v1/host/health": (200, {"status": "degraded",
                                  "problems": ["semantic recall is off: the embedder did not initialise: boom"]}),
        "/v1/mind/state": (200, {}),
    })
    results = check_sidecar("http://127.0.0.1:7777", "key", 5)
    assert results[0].status == WARN
    assert "status=degraded: semantic recall is off: the embedder did not initialise: boom" in results[0].detail
    assert [r.name for r in results] == ["sidecar", "open-files", "sidecar-auth"] and results[2].status == PASS


def test_semantic_recall_is_skipped_without_an_embedding_endpoint(tmp_path, monkeypatch):
    _instance(tmp_path, monkeypatch, {"embed_url": ""})
    calls = _serve(monkeypatch, {})
    result = check_semantic_recall("http://127.0.0.1:7777", "key", 5)
    assert result.status == SKIP and "router.embed_url" in result.detail
    assert calls == []


def test_recall_mode_warns_when_an_endpoint_is_set_but_the_switch_is_off(tmp_path, monkeypatch):
    """An install from before the switch was live may still carry ``semantic_recall: false``: recall is
    lexical although an endpoint is recorded, and doctor says why, once, without asking the sidecar."""
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({
        "router": {"embed_url": "http://127.0.0.1:9101", "embed_model": "m"},
        "mind": {"faculties": {"semantic_recall": False}}}))
    calls = _serve(monkeypatch, {})
    result = doctor.check_recall_mode()
    assert result.status == WARN and result.detail.startswith("keyword recall only")
    assert "mind.faculties.semantic_recall is false" in result.detail
    assert "semantic_recall: true" in result.remedy and calls == []
    assert check_semantic_recall("http://127.0.0.1:7777", "key", 5).status == SKIP and calls == []


def test_recall_mode_warns_keyword_recall_only_without_an_embedding_endpoint(tmp_path, monkeypatch):
    _instance(tmp_path, monkeypatch, {"embed_url": ""})
    result = doctor.check_recall_mode()
    assert result.status == WARN
    assert result.detail == ("keyword recall only: no embedding endpoint (router.embed_url is empty), so recall "
                             "finds words, not meaning")
    assert result.remedy == ("colonymind init --embed-url auto (asks the model endpoint Hermes uses for an embeddings "
                             "route; never a chat request), --embed-url <root> --embed-model <model>, or --embed-url "
                             "local (an embedder the sidecar runs itself; needs colonymind[vectors])")
    _instance(tmp_path, monkeypatch, {"embed_url": "http://127.0.0.1:9101", "embed_model": "m"})
    assert doctor.check_recall_mode().status == PASS


def test_recall_cutoff_warns_that_recall_never_abstains_until_calibrated(tmp_path, monkeypatch):
    _instance(tmp_path, monkeypatch, {"embed_url": "http://127.0.0.1:9101", "embed_model": "m"})
    result = doctor.check_recall_cutoff()
    assert result.status == WARN
    assert result.detail.startswith("no calibrated cutoff: recall never abstains")
    assert "COLONYMIND_RECALL_RERANK_MIN_SCORE" in result.remedy
    assert "COLONYMIND_RECALL_RERANK_CALIBRATION" in result.remedy and "docs/RECALL-HYBRID.md" in result.remedy
    _calibrated(tmp_path, "0.5", "a" * 64)
    _calibrated(tmp_path, "0.5", _runtime_fingerprint(tmp_path))
    result = doctor.check_recall_cutoff()
    assert result.status == PASS and "0.5" in result.detail and "matches" in result.detail


_RERANKED = {"embed_url": "http://127.0.0.1:9101", "embed_model": "m", "embed_dims": 3,
             "rerank_url": "http://127.0.0.1:9102", "rerank_model": "r"}


def _calibrated(tmp_path, score, fingerprint, router=None, **environment):
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({
        "router": router or _RERANKED,
        "environment": {"COLONYMIND_RECALL_RERANK_MIN_SCORE": score, "COLONYMIND_RECALL_RERANK_CALIBRATION": fingerprint,
                        **environment}}))


def _runtime_fingerprint(tmp_path):
    """The fingerprint the running sidecar computes for this configuration's reranker (``server`` builds the
    provider from the same environment; ``RecallSelector`` compares its stamp)."""
    import os
    from unittest.mock import patch
    from colonymind.config import apply_environment, load_config
    from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
    from colonymind.vector.reranker import OpenAIAPIRerankerProvider
    derived = apply_environment(load_config(), environ={})
    with patch.dict(os.environ, derived, clear=True):
        provider = OpenAIAPIRerankerProvider(os.environ["COLONYMIND_RERANKER_MODEL"])
        provider.configure(os.environ["COLONYMIND_RERANKER_BASE_URL"], "",
                           os.environ.get("COLONYMIND_RERANKER_PROMPT_STYLE", ""))
        return calibration_fingerprint(provider_calibration_metadata(provider))


def test_recall_cutoff_warns_when_the_sidecar_would_switch_the_cutoff_off(tmp_path, monkeypatch):
    """A cutoff the runtime rejects leaves recall never abstaining: doctor says so instead of passing."""
    _instance(tmp_path, monkeypatch, _RERANKED)
    _calibrated(tmp_path, "0.5", "a" * 64)                       # a fingerprint of some other configuration
    result = doctor.check_recall_cutoff()
    assert result.status == WARN and "does not match" in result.detail and "never abstains" in result.detail
    assert "recalibrate" in result.remedy and "docs/RECALL-HYBRID.md" in result.remedy
    # Calibrated for this reranker, then the configuration changed under it.
    fingerprint = _runtime_fingerprint(tmp_path)
    _calibrated(tmp_path, "0.5", fingerprint)
    assert doctor.check_recall_cutoff().status == PASS
    _calibrated(tmp_path, "0.5", fingerprint, COLONYMIND_RERANKER_PROMPT_STYLE="qwen3")
    assert doctor.check_recall_cutoff().status == WARN
    for score in ("high", "nan", "inf", "0.5 or so"):
        _calibrated(tmp_path, score, fingerprint)
        result = doctor.check_recall_cutoff()
        assert result.status == WARN and "not a finite number" in result.detail, score
        assert "COLONYMIND_RECALL_RERANK_MIN_SCORE" in result.remedy


def test_recall_cutoff_warns_when_the_fingerprint_cannot_be_checked(tmp_path, monkeypatch):
    """An in-process reranker is chosen by the sidecar's hardware: doctor cannot compute its stamp, so it
    does not claim the cutoff applies, and says where the sidecar reports it."""
    _instance(tmp_path, monkeypatch, _RERANKED)
    _calibrated(tmp_path, "0.5", "a" * 64, router={"embed_url": "http://127.0.0.1:9101", "embed_model": "m"},
                COLONYMIND_RECALL_RERANK="on")
    result = doctor.check_recall_cutoff()
    assert result.status == WARN and "cannot be checked" in result.detail
    assert "Rerank abstention calibration" in result.remedy


@pytest.mark.parametrize("held", ["", "\n", "ab12\n", "00" * 31 + "\n", "00" * 33 + "\n", "not a key\n"])
def test_a_damaged_instance_secret_fails_with_how_to_recover(tmp_path, monkeypatch, held):
    """``instinct.key`` that holds no whole key keys nothing: every decision hashes its state under a key of its
    process alone and model usage drops each row that names a role or an endpoint by its alias, with no other sign
    than a log line. Doctor fails it and says how to recover; it never rewrites the file."""
    _instance(tmp_path, monkeypatch, {})
    (tmp_path / "instinct.key").write_text(held)
    result = doctor.check_instance_secret()
    assert result.status == FAIL and "holds no 32-byte key" in result.detail
    assert "decisions" in result.detail and "model usage" in result.detail
    assert "restore it from a backup" in result.remedy and "remove it" in result.remedy
    assert (tmp_path / "instinct.key").read_text() == held
    assert ("instance-secret", doctor.check_instance_secret) in doctor.LOCAL_CHECKS


def test_the_instance_secret_passes_whole_or_not_made_yet_and_warns_when_others_can_read_it(tmp_path, monkeypatch):
    from colonymind.instance import instance_secret
    _instance(tmp_path, monkeypatch, {})
    result = doctor.check_instance_secret()
    assert result.status == PASS and "not made yet" in result.detail
    assert not (tmp_path / "instinct.key").exists()
    instance_secret(tmp_path)
    result = doctor.check_instance_secret()
    assert result.status == PASS and "mode 600" in result.detail
    (tmp_path / "instinct.key").chmod(0o644)
    result = doctor.check_instance_secret()
    assert result.status == WARN and result.remedy == f"chmod 600 {tmp_path / 'instinct.key'}"


def test_a_dangling_instance_secret_link_fails_and_is_never_made_in_its_place(tmp_path, monkeypatch):
    """A link named ``instinct.key`` whose target is gone is no key and not "not made yet": the sidecar cannot make
    one there (the link holds the name, so the key it writes is never linked into place), so each decision hashes
    its state under a key of its process alone and model usage drops each row named by an alias. Doctor fails it,
    says it is a link and where it points, and never touches it."""
    from colonymind.instance import instance_secret
    _instance(tmp_path, monkeypatch, {})
    gone = tmp_path / "moved-away" / "instinct.key"
    (tmp_path / "instinct.key").symlink_to(gone)
    with pytest.raises(OSError):
        instance_secret(tmp_path)
    result = doctor.check_instance_secret()
    assert result.status == FAIL, result
    assert "link" in result.detail and str(gone) in result.detail
    assert "decisions" in result.detail and "model usage" in result.detail
    assert "restore" in result.remedy and f"rm {tmp_path / 'instinct.key'}" in result.remedy
    assert (tmp_path / "instinct.key").is_symlink() and not gone.exists()


def test_the_recall_checks_are_local_checks(tmp_path, monkeypatch):
    names = [name for name, _ in doctor.LOCAL_CHECKS]
    assert "recall-mode" in names and "recall-cutoff" in names

def test_semantic_recall_fails_when_the_configured_embedder_is_not_serving(tmp_path, monkeypatch):
    _instance(tmp_path, monkeypatch, {"embed_url": "http://127.0.0.1:9101", "embed_model": "m"})
    _serve(monkeypatch, {"/v1/host/embed/health": (200, {
        "status": "error", "error": "the embedder (provider=openai_api, model=m) did not initialise: "
                                    "ValueError: Embedding response dimension 4096 differs from the configured 384"})})
    result = check_semantic_recall("http://127.0.0.1:7777", "key", 5)
    assert result.status == FAIL
    assert "dimension 4096 differs from the configured 384" in result.detail
    assert "router.embed_dims" in result.remedy and "colonymind service restart" in result.remedy


def test_semantic_recall_passes_when_the_embedder_answers(tmp_path, monkeypatch):
    _instance(tmp_path, monkeypatch, {"embed_url": "http://127.0.0.1:9101", "embed_model": "m", "embed_dims": 4096})
    _serve(monkeypatch, {"/v1/host/embed/health": (200, {"status": "ok", "model": "m", "dims": 4096,
                                                        "latency_ms": 194.0})})
    result = check_semantic_recall("http://127.0.0.1:7777", "key", 5)
    assert result.status == PASS and "dims=4096" in result.detail


def test_run_doctor_asks_about_recall_only_when_the_sidecar_answers(tmp_path, monkeypatch):
    _instance(tmp_path, monkeypatch, {"embed_url": "http://127.0.0.1:9101", "embed_model": "m"})
    monkeypatch.setattr(doctor, "run_local_checks", lambda: [])
    _serve(monkeypatch, {"/v1/host/health": (200, {"status": "ok", "problems": []}),
                         "/v1/mind/state": (200, {}),
                         "/v1/host/embed/health": (200, {"status": "error", "error": "embedder not initialized"})})
    names = {r.name: r.status for r in doctor.run_doctor("http://127.0.0.1:7777", "key", 5)}
    assert names == {"sidecar": PASS, "open-files": SKIP, "sidecar-auth": PASS, "semantic-recall": FAIL}

    def down(url, api_key="", timeout=10.0):
        raise OSError("connection refused")
    monkeypatch.setattr(doctor, "_http_get", down)
    names = {r.name: r.status for r in doctor.run_doctor("http://127.0.0.1:7777", "key", 5)}
    assert names == {"sidecar": FAIL}


def test_sidecar_check_warns_when_the_running_sidecar_has_few_open_files(monkeypatch):
    _serve(monkeypatch, {"/v1/host/health": (200, {"status": "ok", "notes": {"fd_limit": "256"}}),
                         "/v1/mind/state": (200, {})})
    results = {r.name: r for r in check_sidecar("http://127.0.0.1:7777", "key", 5)}
    assert results["open-files"].status == WARN
    assert "256 open files" in results["open-files"].detail and "service restart" in results["open-files"].remedy
    _serve(monkeypatch, {"/v1/host/health": (200, {"status": "ok", "notes": {"fd_limit": "16384"}}),
                         "/v1/mind/state": (200, {})})
    results = {r.name: r for r in check_sidecar("http://127.0.0.1:7777", "key", 5)}
    assert results["open-files"].status == PASS
    _serve(monkeypatch, {"/v1/host/health": (200, {"status": "ok"}), "/v1/mind/state": (200, {})})
    results = {r.name: r for r in check_sidecar("http://127.0.0.1:7777", "key", 5)}
    assert results["open-files"].status == SKIP


def test_vector_store_check_fails_when_the_library_is_missing(monkeypatch):
    import importlib.util
    from colonymind.doctor import check_vector_store
    assert check_vector_store().status == PASS
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "lancedb" else real(name, *a, **k))
    result = check_vector_store()
    assert result.status == FAIL and "pipx install --force colonymind" in result.remedy
    assert doctor.run_local_checks()[0].name == "vector-store"


def _service_instance(tmp_path, monkeypatch, platform):
    """The doctor's view of this instance's user service, managed by the fake user manager."""
    import socket
    from colonymind import init
    from colonymind.services.instance import InstanceService
    from test_instance_service import Manager
    (tmp_path / "instance").mkdir()
    _instance(tmp_path / "instance", monkeypatch, {})
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()))
    manager = Manager()
    service = InstanceService(tmp_path / "instance", tmp_path / "hermes", python=str(tmp_path / "venv/bin/python"),
                              home=tmp_path / "user", platform=platform, runner=manager)
    monkeypatch.setattr(init, "_service", lambda cfg: service)
    monkeypatch.setattr(service, "health", lambda: {"status": "ok", "problems": []})
    return service



@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_service_check_says_whether_the_instance_service_keeps_the_sidecar_up(tmp_path, monkeypatch, platform):
    """A sidecar started by hand ('colonymind start --detach') answers every other
    check, but nothing restarts it after a crash or starts it at login. The service check says whether
    this instance's user service is installed, running and written by this version."""
    service = _service_instance(tmp_path, monkeypatch, platform)
    result = doctor.check_service()
    assert result.name == "service" and result.status == WARN
    assert "no user service" in result.detail and "colonymind service install" in result.remedy

    service.install()
    result = doctor.check_service()
    assert result.status == FAIL and "not running" in result.detail and "colonymind service start" in result.remedy

    service.start()
    result = doctor.check_service()
    assert result.status == PASS and service.label in result.detail and "321" in result.detail

    # A unit an earlier release wrote: one log shared with the rotating logger, a restart every 5 s.
    if platform == "darwin":
        plist = plistlib.loads(service.definition.read_bytes())
        plist.update(ThrottleInterval=5, StandardOutPath=str(service.log), StandardErrorPath=str(service.log))
        service.definition.write_bytes(plistlib.dumps(plist, sort_keys=True))
    else:
        unit = service.definition.read_text().replace("RestartSec=30", "RestartSec=5")
        service.definition.write_text(unit.replace(str(service.manager_log), str(service.log)))
    result = doctor.check_service()
    assert result.status == WARN and "earlier release" in result.detail
    assert "colonymind service stop" in result.remedy and "colonymind service install" in result.remedy
    assert "service" in [r.name for r in doctor.run_local_checks()]


def test_service_check_skips_where_no_user_manager_exists(tmp_path, monkeypatch):
    from colonymind import init
    from colonymind.services.instance import ServiceError
    _instance(tmp_path, monkeypatch, {})

    def unsupported(cfg):
        raise ServiceError("Instance autostart supports Linux systemd user services and macOS launchd")

    monkeypatch.setattr(init, "_service", unsupported)
    result = doctor.check_service()
    assert result.status == SKIP and "autostart supports" in result.detail
