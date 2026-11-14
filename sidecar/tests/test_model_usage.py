"""Model usage by lane: one row of counts per model call, in the instance's ``usage.db``, and ``colonymind report usage``.

A row holds when the call ended, which client made it, the caller's task, the role, the instance's own name for the
endpoint, input and output tokens, latency and outcome. Never the words of a prompt or an answer, a URL, a model's
name, or an id of a person or a source.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from argparse import Namespace
from contextlib import closing
from datetime import timedelta

import httpx
import pytest

from model_endpoint import ModelEndpoint, embedding_answer
from colonymind import usage
from colonymind.router.router import LLMRouter

SECRET_WORDS = "Robin's locker code is 4417"


@pytest.fixture
def instance(tmp_path, monkeypatch):
    """A state directory of its own, named the way the sidecar names it."""
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    return tmp_path


def chat_answer(content="done", finish="stop", tokens=(31, 7)):
    def answer(path, body):
        return 200, {"id": "fixture", "object": "chat.completion", "created": 1, "model": body["model"],
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                  "finish_reason": finish}],
                     "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1],
                               "total_tokens": sum(tokens)}}
    return answer


def failing(path, body):
    return 500, {"error": {"message": "fixture unavailable", "type": "server_error"}}


def routing(*urls, timeout=2.0):
    names = ["quick", "steady"][:len(urls)]
    return {"provider": "vllm", "apiKey": "fixture-key", "models": {},
            "modelPool": {name: {"model": f"fixture-{name}", "baseUrl": url + "/v1", "contextTokens": 32000}
                          for name, url in zip(names, urls)},
            "functionRoles": {"extraction": {"candidates": names, "timeoutSeconds": timeout,
                                             "deadlineSeconds": timeout * 2}}}


def configured(*urls, **kwargs):
    router = LLMRouter(tiers={})
    router.configure(routing(*urls, **kwargs))
    return router


async def extract(router, task="source_claim_extraction"):
    return await router.complete([{"role": "user", "content": SECRET_WORDS}],
                                 context={"task": task, "function_role": "extraction"})


def recorded(directory):
    assert usage.flush()
    return usage.rows(directory)


# -- the router ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_router_call_records_one_row_of_counts_and_no_text(instance):
    with ModelEndpoint(chat_answer()) as endpoint:
        response = await extract(configured(endpoint.url))
        assert response.content == "done"
    rows = recorded(instance)
    assert len(rows) == 1
    row = rows[0]
    assert {key: row[key] for key in ("client", "task", "role", "endpoint", "input_tokens", "output_tokens",
                                      "outcome")} == {
        "client": "router", "task": "source_claim_extraction", "role": "extraction", "endpoint": "quick",
        "input_tokens": 31, "output_tokens": 7, "outcome": "ok"}
    assert isinstance(row["latency_ms"], int) and row["latency_ms"] >= 0
    # The schema holds counts and codes only: no column for text, a URL, a model or an id.
    with sqlite3.connect(instance / usage.DB_FILE) as conn:
        columns = [line[1] for line in conn.execute("PRAGMA table_info(model_calls)")]
        tables = {line[0] for line in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert columns == list(usage.COLUMNS) and tables == {"model_calls", "captures"}
    stored = (instance / usage.DB_FILE).read_bytes()
    for word in ("Robin", "4417", "fixture-quick", "127.0.0.1", str(endpoint.url.rsplit(":", 1)[1]), "done"):
        assert word.encode() not in stored


@pytest.mark.asyncio
async def test_a_failed_call_records_its_outcome_and_the_fallback_its_own(instance):
    with ModelEndpoint(failing) as down, ModelEndpoint(chat_answer()) as up:
        response = await extract(configured(down.url, up.url))
        assert response.binding == "steady"
    rows = sorted(recorded(instance), key=lambda row: row["endpoint"])
    assert [(row["endpoint"], row["outcome"]) for row in rows] == [("quick", "error"), ("steady", "ok")]
    assert rows[0]["input_tokens"] is None and rows[0]["output_tokens"] is None


@pytest.mark.asyncio
async def test_a_timed_out_and_an_unfinished_call_record_what_happened(instance):
    with ModelEndpoint(chat_answer(), delay=0.5) as slow:
        with pytest.raises(RuntimeError):
            await extract(configured(slow.url, timeout=0.1))
    with ModelEndpoint(chat_answer("half an ans", finish="length", tokens=(40, 64))) as cut:
        with pytest.raises(RuntimeError):
            await extract(configured(cut.url))
    rows = recorded(instance)
    assert [row["outcome"] for row in rows] == ["timeout", "incomplete"]
    assert (rows[1]["input_tokens"], rows[1]["output_tokens"]) == (40, 64)   # the unusable answer still cost them


@pytest.mark.asyncio
async def test_a_task_or_name_that_could_carry_an_identifier_is_not_recorded_as_given(instance):
    """A task is a code name (letters and underscores) or a name the process registered (a decision point): anything
    else, a digit or a hyphen included, is recorded as ``unnamed``, so no id or number reaches the table through a
    caller's task."""
    with ModelEndpoint(chat_answer()) as endpoint:
        router = configured(endpoint.url)
        for task in ("turn_4417", "coding-ab12cd", "Robin's claims", None):
            await extract(router, task=task)
    assert {row["task"] for row in recorded(instance)} == {"unnamed"}


def keyed_alias(directory, value):
    """The alias a name has in the instance at ``directory``: an HMAC-SHA256 under its own secret, of the name's
    SHA-256 prefix (the form an earlier build kept, so its rows are keyed in place)."""
    import hashlib
    import hmac
    secret = bytes.fromhex((directory / "instinct.key").read_text().strip())
    digest = ("alias-" + hashlib.sha256(value.encode()).hexdigest()[:12]).encode()
    return "alias-" + hmac.new(secret, digest, hashlib.sha256).hexdigest()[:12]


@pytest.mark.asyncio
async def test_two_valid_endpoint_names_are_never_merged_and_never_kept_as_text(instance):
    """A binding's name may hold a dot or start with a digit (``worker.v1``): routing accepts it, so its rows keep
    it apart from every other binding's. A plain name is kept as it is; any other is kept as ``alias-`` and the
    first 12 hex digits of its HMAC-SHA256 under the instance's own secret, so a host name or an address never
    reaches the table as text."""
    with ModelEndpoint(failing) as down, ModelEndpoint(chat_answer()) as up:
        router = LLMRouter(tiers={})
        router.configure({"provider": "vllm", "apiKey": "fixture-key", "models": {},
                          "modelPool": {name: {"model": "fixture", "baseUrl": url + "/v1", "contextTokens": 32000}
                                        for name, url in (("worker.v1", down.url), ("worker.v2", up.url))},
                          "functionRoles": {"extraction": {"candidates": ["worker.v1", "worker.v2"]}}})
        assert (await extract(router)).binding == "worker.v2"
    rows = recorded(instance)
    aliases = [keyed_alias(instance, name) for name in ("worker.v1", "worker.v2")]
    assert [(row["endpoint"], row["outcome"]) for row in rows] == [(aliases[0], "error"), (aliases[1], "ok")]
    assert b"worker" not in (instance / usage.DB_FILE).read_bytes()
    secret = bytes.fromhex((instance / "instinct.key").read_text().strip())
    assert usage.name("gpu-box_2", secret) == "gpu-box_2"
    names = ("box.lan", "192.0.2.1", "http://x/v1", "2box", "decider 1")
    assert len({usage.name(value, secret) for value in names}) == len(names)
    assert all(usage.name(value, secret).startswith("alias-") and value not in usage.name(value, secret)
               for value in names)
    assert {usage.name(value, secret) for value in ("", None, 7)} == {"other"}


def test_an_alias_cannot_be_recovered_from_the_ledger_by_trying_names(tmp_path, monkeypatch):
    """An address is a small space to search: were its alias a plain digest, anyone holding the ledger could hash
    every address until one matched. The alias is keyed with the instance's own secret (``instinct.key``, which a
    backup carries only encrypted), so the same name has another alias in every instance, none is its plain
    digest, and the ledger alone tests no guess. The secret is made once, owner-only, and kept."""
    import hashlib
    import stat
    found = {}
    for label in ("one", "two"):
        directory = tmp_path / label
        directory.mkdir()
        monkeypatch.setenv("COLONYMIND_STATE_DIR", str(directory))
        usage.record(client="router", task="source_claim_extraction", role="extraction", endpoint="192.0.2.1",
                     latency_ms=5, outcome="ok")
        [row] = recorded(directory)
        found[label] = row["endpoint"]
        assert found[label] == keyed_alias(directory, "192.0.2.1")
        assert stat.S_IMODE((directory / "instinct.key").stat().st_mode) == 0o600
    assert found["one"] != found["two"]
    assert "alias-" + hashlib.sha256(b"192.0.2.1").hexdigest()[:12] not in found.values()
    key = (tmp_path / "one" / "instinct.key").read_text()
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path / "one"))
    usage.record(client="router", task="source_claim_extraction", role="extraction", endpoint="192.0.2.1",
                 latency_ms=5, outcome="ok")
    assert [row["endpoint"] for row in recorded(tmp_path / "one")] == [found["one"]] * 2
    assert (tmp_path / "one" / "instinct.key").read_text() == key


def test_an_upgrade_keys_the_aliases_an_earlier_build_wrote_so_one_name_keeps_one_alias(instance):
    """An earlier build kept a name that is not plain as its plain SHA-256 prefix. The first write of this build
    keys each such alias in place (once: the ledger records that it did), so a name's earlier rows and its new ones
    are one group in the report, ``--alias`` finds both, and no alias a guess could test is left."""
    import hashlib
    from colonymind.util.temporal import now_utc
    earlier = {value: "alias-" + hashlib.sha256(value.encode()).hexdigest()[:12] for value in ("192.0.2.1", "worker.v1")}
    with closing(sqlite3.connect(instance / usage.DB_FILE)) as conn, conn:
        conn.executescript(usage.SCHEMA)
        conn.executemany(f"INSERT INTO model_calls ({', '.join(usage.COLUMNS)}) VALUES (?,?,?,?,?,?,?,?,?)", [
            (usage.stamp(now_utc()), "router", "source_claim_extraction", earlier["worker.v1"], earlier["192.0.2.1"],
             10, 2, 5, "ok"),
            (usage.stamp(now_utc()), "router", "source_claim_extraction", "extraction", "quick", 10, 2, 5, "ok")])
    for _ in range(2):              # the second write finds the ledger keyed already and keys nothing twice
        usage.record(client="router", task="source_claim_extraction", role="worker.v1", endpoint="192.0.2.1",
                     latency_ms=5, outcome="ok")
        found = recorded(instance)
        assert {row["endpoint"] for row in found} == {usage.alias_of(instance, "192.0.2.1"), "quick"}
        assert {row["role"] for row in found} == {usage.alias_of(instance, "worker.v1"), "extraction"}
    assert len(found) == 4
    raw = (instance / usage.DB_FILE).read_bytes()
    assert not [alias for alias in earlier.values() if alias.encode() in raw]
    assert usage.alias_of(instance, "192.0.2.1") == keyed_alias(instance, "192.0.2.1")


def test_a_row_whose_alias_cannot_be_keyed_is_dropped_never_kept_unkeyed(instance):
    """A secret that cannot be read (a damaged ``instinct.key``) drops the rows that need an alias, counted as
    dropped; rows whose names are plain are still written."""
    (instance / "instinct.key").write_text("not a key\n")
    before = usage.dropped()
    for endpoint in ("192.0.2.1", "quick"):
        usage.record(client="router", task="source_claim_extraction", role="extraction", endpoint=endpoint,
                     latency_ms=5, outcome="ok")
    assert [row["endpoint"] for row in recorded(instance)] == ["quick"]
    assert usage.dropped() == before + 1


@pytest.mark.parametrize("held", ["", "\n", "ab12\n", "00" * 31 + "\n", "00" * 33 + "\n"])
def test_a_secret_file_that_holds_no_whole_key_keys_nothing(instance, held):
    """Only 32 whole bytes key an alias: an empty file (one another process has made and not yet written, or one a
    crash left) or a cut or overlong one keys nothing, so no row is written under a key anyone can guess. The file
    is kept (what it keyed would match nothing), and the error says how to recover."""
    from colonymind.instance import instance_secret
    (instance / "instinct.key").write_text(held)
    with pytest.raises(ValueError, match="32-byte key.*remove it"):
        instance_secret(instance)
    before = usage.dropped()
    usage.record(client="router", task="source_claim_extraction", role="extraction", endpoint="192.0.2.1",
                 latency_ms=5, outcome="ok")
    assert recorded(instance) == [] and usage.dropped() == before + 1
    assert (instance / "instinct.key").read_text() == held


@pytest.mark.parametrize("held", ["", "ab12\n"])
async def test_a_damaged_secret_leaves_every_decision_running_under_a_key_no_one_can_guess(tmp_path, held):
    """The decision path reads the same secret for its keyed hashes. A file that holds no whole key refuses to key
    anything there too, but a decision is never lost to it: the point is still answered by its tiers, its state
    hashed under a key of this process alone (as when the secret cannot be read at all), never an empty one."""
    import hashlib
    import hmac
    from colonymind.instinct.log import Store
    from test_instinct_core import Provider, Tier, engine, record, yes_no
    (tmp_path / "instinct.key").write_text(held)
    store = Store(tmp_path / "instinct.db")
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),), fallback=lambda fields, rules: None)
    made = engine([point], fast=Provider(probe={"yes": 0.95, "no": 0.05}), records=[record(point)], store=store)
    decision = await made.decide("probe", {"text": "x"}, {"turn": "t-1"})
    await made.drain()
    assert (decision.label, decision.applied, decision.tier) == ("yes", True, "fast")
    [row] = store.rows()
    guessable = {hmac.new(bytes.fromhex(held.strip()), point.render({"text": "x"}).encode(), hashlib.sha256)
                 .hexdigest()[:len(row["state_hash"] or "")]}
    assert row["state_hash"] and row["state_hash"] not in guessable
    assert (tmp_path / "instinct.key").read_text() == held


def test_the_secret_appears_whole_or_not_at_all(tmp_path, monkeypatch):
    """The secret is written whole under a private name and linked into place, so a reader never finds the file
    before its bytes (the background writer would have keyed its aliases with nothing). Racing makers all keep the
    first one linked."""
    import os
    import threading
    from colonymind import instance as instance_module
    seen = []
    link = os.link

    def watched(source, target, *args, **kwargs):
        seen.append((os.path.exists(target), len(bytes.fromhex(open(source).read().strip()))))
        return link(source, target, *args, **kwargs)
    monkeypatch.setattr(instance_module.os, "link", watched)
    directory = tmp_path / "fresh"
    barrier, made = threading.Barrier(8), []

    def make():
        barrier.wait()
        made.append(instance_module.instance_secret(directory))
    threads = [threading.Thread(target=make) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen and all(size == 32 for _, size in seen) and seen[0][0] is False
    assert len(made) == 8 and len(set(made)) == 1 and len(made[0]) == 32
    assert [path.name for path in directory.iterdir()] == ["instinct.key"]


def test_the_report_names_the_alias_this_instance_records_for_a_name(instance, capsys):
    """``colonymind report usage --alias NAME`` says what this instance's rows record for a name: the name itself
    when it is plain, else its alias. It reads the secret and never makes one: before any alias was recorded it
    says there is none."""
    from colonymind.reports import cli

    def ask(value):
        code = cli.run(Namespace(report_command="usage", since="7d", json=False, format="md", out=None, alias=value))
        return code, capsys.readouterr().out
    code, out = ask("192.0.2.1")
    assert code == 0 and "no alias" in out and not (instance / "instinct.key").exists()
    usage.record(client="router", task="source_claim_extraction", role="extraction", endpoint="192.0.2.1",
                 latency_ms=5, outcome="ok")
    [row] = recorded(instance)
    assert ask("192.0.2.1") == (0, f"192.0.2.1 is recorded as {row['endpoint']}\n")
    assert ask("quick") == (0, "quick is recorded as quick\n")


def test_the_alias_report_finds_the_rows_an_earlier_build_wrote_before_a_write_keys_them(instance, capsys):
    """An earlier build's ledger keeps a name's plain digest prefix until this build's first write keys it in place.
    Until then ``--alias`` names that alias as well, and the keyed one those rows and new ones take at the next write
    (or, with no secret made yet, says the next write keys them); after the write only the keyed one. It never makes
    the secret."""
    import hashlib
    from colonymind.instance import instance_secret
    from colonymind.reports import cli
    from colonymind.util.temporal import now_utc
    earlier = "alias-" + hashlib.sha256(b"192.0.2.1").hexdigest()[:12]
    with closing(sqlite3.connect(instance / usage.DB_FILE)) as conn, conn:
        conn.executescript(usage.SCHEMA)
        conn.execute(f"INSERT INTO model_calls ({', '.join(usage.COLUMNS)}) VALUES (?,?,?,?,?,?,?,?,?)",
                     (usage.stamp(now_utc()), "router", "source_claim_extraction", "extraction", earlier, 10, 2, 5,
                      "ok"))

    def ask(value):
        code = cli.run(Namespace(report_command="usage", since="7d", json=False, format="md", out=None, alias=value))
        return code, capsys.readouterr().out
    code, out = ask("192.0.2.1")
    assert code == 0 and out == (f"192.0.2.1 is recorded as {earlier} in the rows an earlier build wrote; the next "
                                 "write keys them under this instance's own secret\n")
    assert not (instance / "instinct.key").exists()
    instance_secret(instance)                    # a decision made the secret; nothing has written a row since
    keyed = keyed_alias(instance, "192.0.2.1")
    assert ask("192.0.2.1") == (0, f"192.0.2.1 is recorded as {earlier} in the rows an earlier build wrote, and as "
                                  f"{keyed} from the next write, which keys those rows the same way\n")
    assert ask("worker.v1") == (0, f"worker.v1 is recorded as {keyed_alias(instance, 'worker.v1')}\n")
    usage.record(client="router", task="source_claim_extraction", role="extraction", endpoint="192.0.2.1",
                 latency_ms=5, outcome="ok")
    assert [row["endpoint"] for row in recorded(instance)] == [keyed, keyed]
    assert ask("192.0.2.1") == (0, f"192.0.2.1 is recorded as {keyed}\n")


# -- failure isolation and the hot path ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_ledger_that_cannot_be_written_never_fails_the_model_call(instance):
    (instance / usage.DB_FILE).mkdir()                      # a directory where the database file should be
    with ModelEndpoint(chat_answer()) as endpoint:
        assert (await extract(configured(endpoint.url))).content == "done"
    assert usage.flush()
    assert (instance / usage.DB_FILE).is_dir()


def test_recording_never_raises_and_a_full_queue_drops_the_row(instance, monkeypatch):
    before = usage.dropped()
    monkeypatch.setattr(usage._WRITER, "_queue", _Full())
    usage.record(client="router", task="mind_compose", role="reasoning", endpoint="quick", latency_ms=3,
                 outcome="ok")
    assert usage.dropped() == before + 1
    monkeypatch.setattr(usage._WRITER, "_queue", _Broken())
    usage.record(client="router", task="mind_compose", role="reasoning", endpoint="quick", latency_ms=3,
                 outcome="ok")


@pytest.mark.asyncio
async def test_a_writer_that_cannot_start_never_fails_the_model_call(instance, monkeypatch):
    """Starting the writer thread can fail (a process at its thread limit): the call's answer still comes back,
    and its row waits in the queue for the next writer that starts."""
    start = threading.Thread.start

    def refuse(thread):
        if thread.name == "colonymind-usage":
            raise RuntimeError("can't start new thread")
        return start(thread)
    monkeypatch.setattr(usage, "_WRITER", usage._Writer())       # a process whose writer has not started yet
    monkeypatch.setattr(threading.Thread, "start", refuse)
    with ModelEndpoint(chat_answer()) as endpoint:
        assert (await extract(configured(endpoint.url))).content == "done"
    assert usage.flush(timeout=0.5) is False                 # no writer yet; the flush says so and never raises
    monkeypatch.setattr(threading.Thread, "start", start)
    assert [row["outcome"] for row in recorded(instance)] == ["ok"]


class _Full:
    def put_nowait(self, item):
        import queue
        raise queue.Full


class _Broken:
    def put_nowait(self, item):
        raise RuntimeError("the queue is broken")


@pytest.mark.asyncio
async def test_the_calling_thread_never_opens_the_ledger(instance, monkeypatch):
    """A model call only hands its row to the ledger's writer thread: the database is written off the call's path."""
    opened_by = []
    connect = sqlite3.connect

    def watched(*args, **kwargs):
        opened_by.append(threading.current_thread().name)
        return connect(*args, **kwargs)
    monkeypatch.setattr(sqlite3, "connect", watched)
    with ModelEndpoint(chat_answer()) as endpoint:
        caller = threading.current_thread().name
        await extract(configured(endpoint.url))
        assert usage.flush()
    assert opened_by and caller not in opened_by


def test_nothing_is_recorded_without_a_named_state_directory(tmp_path, monkeypatch):
    """A harness or script that sets no state directory records nothing (never into a default instance)."""
    monkeypatch.delenv("COLONYMIND_STATE_DIR", raising=False)
    monkeypatch.delenv("COLONYMIND_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".colonymind").mkdir()
    (tmp_path / ".colonymind" / "colonymind.yaml").write_text("{}\n")
    usage.record(client="router", task="mind_compose", role="reasoning", endpoint="quick", latency_ms=3,
                 outcome="ok")
    assert usage.flush()
    assert not list(tmp_path.rglob(usage.DB_FILE))


# -- retention -------------------------------------------------------------------------------------------------


def _row(at, **values):
    base = {"at": at.isoformat(), "client": "router", "task": "source_claim_extraction", "role": "extraction",
            "endpoint": "quick", "input_tokens": 10, "output_tokens": 2, "latency_ms": 50, "outcome": "ok"}
    return {**base, **values}


def _write(directory, rows):
    with sqlite3.connect(directory / usage.DB_FILE) as conn:
        conn.executescript(usage.SCHEMA)
        conn.executemany(f"INSERT INTO model_calls ({', '.join(usage.COLUMNS)}) VALUES "
                         f"({', '.join('?' * len(usage.COLUMNS))})",
                         [tuple(row[key] for key in usage.COLUMNS) for row in rows])


@pytest.mark.asyncio
async def test_an_idle_instance_drops_rows_past_retention_in_the_daily_maintenance(instance):
    """The sidecar's daily maintenance drops usage rows and capture counts past retention, whether or not the
    instance makes another model call: an idle instance keeps no row past 30 days."""
    from colonymind.instinct import maintenance
    from colonymind.util.temporal import now_utc
    now = now_utc()
    _write(instance, [_row(now - usage.RETENTION - timedelta(hours=1), task="mind_compose"),
                      _row(now - usage.RETENTION + timedelta(hours=1), task="mind_deliberate")])
    with sqlite3.connect(instance / usage.DB_FILE) as conn:
        conn.executemany("INSERT INTO captures (at, messages) VALUES (?, ?)",
                         [(usage.stamp(now - usage.RETENTION - timedelta(hours=1)), 3),
                          (usage.stamp(now - usage.RETENTION + timedelta(hours=1)), 2)])
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        if len(slept) > 1:
            raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await maintenance(instance, sleep=sleep, first_delay=0)
    assert [row["task"] for row in usage.rows(instance)] == ["mind_deliberate"]
    assert usage.captured_since(instance, now - timedelta(days=60)) == 2
    assert slept[1] == 24 * 3600 and usage.RETENTION == timedelta(days=30)
    assert not (instance / "instinct.db").exists()          # the usage rows are pruned without a decision log


def test_recording_leaves_retention_to_the_maintenance(instance):
    """Retention is the daily maintenance's work alone: recording a call leaves older rows for it."""
    from colonymind.util.temporal import now_utc
    _write(instance, [_row(now_utc() - usage.RETENTION - timedelta(hours=1), task="mind_compose")])
    usage.record(client="router", task="mind_drift_check", role="reasoning", endpoint="quick", latency_ms=3,
                 outcome="ok")
    assert sorted(row["task"] for row in recorded(instance)) == ["mind_compose", "mind_drift_check"]
    assert usage.prune(instance) == 1


# -- the other model clients -----------------------------------------------------------------------------------


def _decision_server(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_a_decision_provider_records_a_row_per_question_it_asks(instance):
    from colonymind.instinct.providers.local_decision import LocalDecision

    def answer(request):
        body = json.loads(request.content)
        if "fail" in body["state"]:
            return httpx.Response(503, json={})
        if "garbled" in body["state"]:
            return httpx.Response(200, json={"protocol": "local-decision.v1"})
        return httpx.Response(200, json={"protocol": "local-decision.v1", "input_truncated": False,
                                         "answers": {"decision": {"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}}})
    provider = LocalDecision("local", url="http://decisions.invalid", transport=_decision_server(answer))
    question = {"type": "yes_no", "instructions": "Is it settled?", "labels": {"yes": "settled", "no": "open"}}
    await provider.answer(f"{SECRET_WORDS}", {"owner_send": question})
    for state in ("fail", "garbled"):
        with pytest.raises(Exception):
            await provider.answer(state, {"question_answer": question})
    rows = recorded(instance)
    assert [(row["client"], row["task"], row["role"], row["endpoint"], row["outcome"]) for row in rows] == [
        ("instinct", "owner_send", "local", "local", "ok"),
        ("instinct", "question_answer", "local", "local", "error"),
        ("instinct", "question_answer", "local", "local", "incomplete")]
    assert all(row["input_tokens"] is None for row in rows)
    assert b"Robin" not in (instance / usage.DB_FILE).read_bytes()


@pytest.mark.asyncio
async def test_a_registered_point_is_recorded_under_its_own_name(instance):
    """A point's name may hold digits (an operator's ``cron_job_1``): once the process registers the point, as the
    gates' engine does when it is built, its rows carry its own name, so two points never merge. A name no registry
    holds is still ``unnamed``, whatever its shape."""
    from colonymind.instinct import external
    from colonymind.instinct.providers.local_decision import LocalDecision
    external.build_engine({"external": {"cron_job_1": {"kind": "cron", "question": "Is there a reply?"},
                                        "cron_job_2": {"kind": "cron", "question": "Is there news?"}}})

    def answer(request):
        return httpx.Response(200, json={"protocol": "local-decision.v1", "input_truncated": False,
                                         "answers": {"decision": {"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}}})
    provider = LocalDecision("local", url="http://decisions.invalid", transport=_decision_server(answer))
    question = {"type": "yes_no", "instructions": "Is it settled?", "labels": {"yes": "settled", "no": "open"}}
    for point in ("cron_job_1", "cron_job_2", "cron_job_4417"):
        await provider.answer("a state", {point: question})
    assert [row["task"] for row in recorded(instance)] == ["cron_job_1", "cron_job_2", "unnamed"]
    assert usage.task_name("turn_4417") == "unnamed" and usage.task_name("owner_send") == "owner_send"


@pytest.mark.asyncio
async def test_the_reader_records_each_completions_call_with_its_tokens(instance):
    from colonymind.instinct.providers.logits import Logits

    def answer(request):
        return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [{" A": -0.1, " B": -2.5}]}}],
                                         "usage": {"prompt_tokens": 120, "completion_tokens": 1}})
    reader = Logits("reader", url="http://reader.invalid/v1", model="fixture", rotations=2,
                    transport=_decision_server(answer))
    question = {"type": "yes_no", "instructions": "Is it settled?", "labels": {"yes": "settled", "no": "open"}}
    await reader.answer("a state", {"claim_relation": question})
    rows = recorded(instance)
    assert [(row["task"], row["role"], row["input_tokens"], row["output_tokens"], row["outcome"]) for row in rows] \
        == [("claim_relation", "reader", 120, 1, "ok")] * 2


@pytest.mark.asyncio
async def test_query_embeddings_batches_and_reranks_are_recorded(instance):
    from model_endpoint import rerank_answer
    from colonymind.vector.config import EmbeddingConfig
    from colonymind.vector.openai_provider import OpenAIAPIEmbeddingProvider
    from colonymind.vector.reranker import OpenAIAPIRerankerProvider

    def embeddings(path, body):
        status, payload = embedding_answer(path, body)
        return status, {**payload, "usage": {"prompt_tokens": 9, "total_tokens": 9}}
    with ModelEndpoint(embeddings) as endpoint:
        embedder = OpenAIAPIEmbeddingProvider(EmbeddingConfig(provider="openai_api", model_id="fixture", dimensions=2))
        embedder.configure(endpoint.url, "")
        await embedder.embed_query(SECRET_WORDS)
        await embedder.embed_batch(["one", "two"])
        await embedder.close()
    with ModelEndpoint(lambda path, body: (200, {**rerank_answer(path, body)[1], "usage": {"total_tokens": 21}})) \
            as endpoint:
        reranker = OpenAIAPIRerankerProvider("fixture-reranker")
        reranker.configure(endpoint.url, "")
        await reranker.rerank("where", ["blue tin", "red box"], top_k=2)
        await reranker.close()
    rows = recorded(instance)
    # An embedding or a rerank generates no tokens: its output count is 0, not unknown.
    assert [(row["client"], row["task"], row["role"], row["endpoint"], row["input_tokens"], row["output_tokens"],
             row["outcome"]) for row in rows] == [
        ("embedder", "query_embedding", "embedding", "embed", 9, 0, "ok"),
        ("embedder", "text_embedding", "embedding", "embed", 9, 0, "ok"),
        ("reranker", "rerank", "rerank", "rerank", 21, 0, "ok")]


@pytest.mark.asyncio
async def test_a_rerank_whose_answer_cannot_be_used_is_not_recorded_as_ok(instance):
    """The row covers reading the answer too: an index outside the documents sent fails the call, and its row says
    the answer came but could not be used, with the tokens it still cost."""
    from colonymind.vector.reranker import OpenAIAPIRerankerProvider
    answer = {"results": [{"index": 7, "relevance_score": 0.9}], "usage": {"total_tokens": 5}}
    with ModelEndpoint(lambda path, body: (200, answer)) as endpoint:
        reranker = OpenAIAPIRerankerProvider("fixture-reranker")
        reranker.configure(endpoint.url, "")
        with pytest.raises(IndexError):
            await reranker.rerank("where", ["blue tin", "red box"], top_k=2)
        await reranker.close()
    assert [(row["client"], row["input_tokens"], row["outcome"]) for row in recorded(instance)] == [
        ("reranker", 5, "incomplete")]


def _unit_vectors(path, body):
    return 200, {"model": "fixture-served", "usage": {"prompt_tokens": 9, "total_tokens": 9},
                 "data": [{"index": i, "embedding": [1.0, 0.0, 0.0]} for i in range(len(body.get("input") or []))]}


@pytest.mark.asyncio
async def test_only_a_captured_sources_text_is_a_memory_embedding(instance):
    """An embedding's row says what it was for: a warmup and each health check are ``embedding_check``, a live
    query ``query_embedding``, and only the text of a captured source ``source_embedding``, the memory lane. The
    per-message cost counts that lane alone, however often the health route probes the embedder."""
    from colonymind.reports import usage as report
    from colonymind.vector.config import EmbeddingConfig
    from colonymind.vector.embedder import EmbeddingPipeline
    from colonymind.vector.openai_provider import OpenAIAPIEmbeddingProvider
    from test_source_vectors import drain, setup
    with ModelEndpoint(_unit_vectors) as endpoint:
        provider = OpenAIAPIEmbeddingProvider(EmbeddingConfig(provider="openai_api", model_id="fixture", dimensions=3))
        provider.configure(endpoint.url, "")
        pipeline = EmbeddingPipeline(provider)
        await pipeline.warmup()
        for _ in range(2):
            assert (await pipeline.health_check())["status"] == "ok"
        ledger, _, _, projection = await setup(instance, pipeline)
        ledger.record_source("t0", contact_id="owner", session_id="s",
                             messages=[{"role": "user", "content": "The office labeler is ready."}])
        await drain(projection)
        await pipeline.embed_query("where is the labeler")
        await pipeline.close()
    assert [row["task"] for row in recorded(instance)] == [
        "embedding_check", "embedding_check", "embedding_check", "source_embedding", "query_embedding"]
    memory = report.collect(instance, since="7d").memory
    assert memory["captured_messages"] == 1 and memory["calls"] == 1 and memory["input_tokens"] == 9


@pytest.mark.asyncio
async def test_reindexing_a_source_for_a_new_embedder_is_not_a_capture_cost(instance):
    """A new embedder re-embeds every source already indexed (``source_reindex``): maintenance, not the memory lane
    of the window it runs in. A source's first embedding is ``source_embedding``."""
    from colonymind.vector.migrate import migrate_tier
    from test_source_vectors import Pipeline, drain, setup

    class Declared(Pipeline):
        tasks: list = []

        async def embed_batch(self, texts):
            self.tasks.append(usage.embedding_task_of("undeclared"))
            return await super().embed_batch(texts)
    ledger, _, _, projection = await setup(instance, Declared("a"))
    ledger.record_source("t0", contact_id="owner", session_id="s",
                         messages=[{"role": "user", "content": "The hydrofoil code is ready."}])
    await drain(projection)
    _, replacement, pipeline, reprojection = await setup(instance, Declared("b", rotate=True))
    assert not (await migrate_tier(replacement, pipeline)).errors
    await drain(reprojection)
    # The rebuild's own pass over the retained rows names no task: other text, never the memory lane either.
    assert [task for task in Declared.tasks if task != "undeclared"] == ["source_embedding", "source_reindex"]
    from colonymind.reports.usage import MEMORY_TASKS
    assert "source_embedding" in MEMORY_TASKS and not {"source_reindex", "embedding_check", "query_embedding",
                                                       "text_embedding"} & set(MEMORY_TASKS)


class _Vectors(list):
    """An array of vectors as a local model returns one: it and each of its rows convert with ``tolist``."""

    def tolist(self):
        return [item.tolist() if isinstance(item, _Vectors) else item for item in self]


class _LocalModel:
    """A stand-in for a local model's weights: sentence-transformers' ``encode`` and ``predict``."""

    def encode(self, texts, **options):
        return _Vectors([_Vectors([1.0, 0.0]) for _ in texts])

    def predict(self, pairs):
        return [1.0 / (1 + i) for i in range(len(pairs))]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cpu", "cuda", "mlx", "native_mlx"])
async def test_a_local_embedder_records_its_calls_with_no_token_count(instance, kind, monkeypatch):
    """A local model's calls are model calls too: one row each, with its latency and no token count (a local model
    reports none), the warmup's as a check and a live query's as a query."""
    from colonymind.vector import embedder as local
    from colonymind.vector.config import EmbeddingConfig
    provider = local.make_provider(EmbeddingConfig(provider=kind, model_id="fixture", dimensions=2))
    model = _LocalModel()
    monkeypatch.setattr(provider, "_load_model", lambda: (model, None) if kind == "native_mlx" else model)
    if kind == "native_mlx":
        monkeypatch.setattr(provider, "_warmup_inference", lambda: provider._infer(["warmup"]))
        monkeypatch.setattr(provider, "_infer", lambda texts, max_length=512: model.encode(texts).tolist())
    pipeline = local.EmbeddingPipeline(provider)
    await pipeline.warmup()
    assert await pipeline.embed_query(SECRET_WORDS) == [1.0, 0.0]
    assert await pipeline.embed_batch(["one", "two"]) == [[1.0, 0.0]] * 2
    assert await provider.embed("three") == [1.0, 0.0]
    rows = recorded(instance)
    assert [(row["client"], row["task"], row["role"], row["endpoint"], row["input_tokens"], row["output_tokens"],
             row["outcome"]) for row in rows] == [
        ("embedder", "embedding_check", "embedding", "embed", None, 0, "ok"),
        ("embedder", "query_embedding", "embedding", "embed", None, 0, "ok"),
        ("embedder", "text_embedding", "embedding", "embed", None, 0, "ok"),
        ("embedder", "text_embedding", "embedding", "embed", None, 0, "ok")]
    assert all(isinstance(row["latency_ms"], int) for row in rows)
    assert b"Robin" not in (instance / usage.DB_FILE).read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["CPURerankerProvider", "CUDARerankerProvider", "MLXRerankerProvider",
                                  "NativeMLXRerankerProvider"])
async def test_a_local_reranker_records_its_calls(instance, name, monkeypatch):
    """A local reranker's calls are model calls too: one row each, with its latency, no token count and its
    outcome."""
    from colonymind.vector import reranker as local

    def failing(*args):
        raise RuntimeError("the model failed")
    provider = getattr(local, name)("fixture")
    provider._model = _LocalModel()
    if name == "NativeMLXRerankerProvider":
        monkeypatch.setattr(provider, "_score_documents", lambda query, documents: [0.2, 0.9])
    ranked = await provider.rerank(SECRET_WORDS, ["blue tin", "red box"], top_k=2)
    assert len(ranked) == 2
    monkeypatch.setattr(provider._model, "predict", failing)
    monkeypatch.setattr(provider, "_score_documents", failing, raising=False)
    with pytest.raises(RuntimeError):
        await provider.rerank("where", ["blue tin"], top_k=1)
    assert [(row["client"], row["task"], row["role"], row["endpoint"], row["input_tokens"], row["output_tokens"],
             row["outcome"]) for row in recorded(instance)] == [
        ("reranker", "rerank", "rerank", "rerank", None, 0, "ok"),
        ("reranker", "rerank", "rerank", "rerank", None, 0, "error")]


# -- the report ------------------------------------------------------------------------------------------------


def _usage_rows(now):
    at = now - timedelta(hours=2)
    return [
        _row(at, task="source_claim_extraction", input_tokens=300, output_tokens=40, latency_ms=900),
        _row(at, task="source_claim_extraction", input_tokens=280, output_tokens=38, latency_ms=1100),
        _row(at, task="source_claim_review", role="judging", input_tokens=200, output_tokens=10, latency_ms=700),
        _row(at, task="commitment_extract", input_tokens=150, output_tokens=None, latency_ms=400),
        _row(at, client="embedder", task="source_embedding", role="embedding", endpoint="embed", input_tokens=30,
             output_tokens=0, latency_ms=20),
        _row(at, task="mind_deliberate", role="reasoning", endpoint="steady", input_tokens=1000, output_tokens=200,
             latency_ms=3000),
        _row(at, task="mind_deliberate", role="reasoning", endpoint="steady", input_tokens=None,
             output_tokens=None, latency_ms=5000, outcome="timeout"),
        _row(at, client="instinct", task="owner_send", role="local", endpoint="local", input_tokens=None,
             output_tokens=None, latency_ms=30),
        _row(now - timedelta(days=9), task="mind_compose", role="reasoning", input_tokens=5, output_tokens=5),
    ]


def _captured(directory, count):
    from colonymind.turns.idempotency import TurnIdempotencyLedger
    ledger = TurnIdempotencyLedger(directory / "turn-idempotency.db")
    for n in range(count):
        ledger.record_source(f"t{n}", contact_id="owner", session_id="s", messages=[
            {"role": "user", "content": f"message {n}"}, {"role": "assistant", "content": "noted"}])
    # A checkpoint's whole transcript is session-scoped: its old messages were not captured now.
    ledger.record_source("checkpoint", contact_id="owner", session_id="s", scope="session", messages=[
        {"role": "user", "content": f"earlier {n}"} for n in range(3)])
    assert usage.flush()
    return ledger


def test_the_report_totals_match_the_rows_and_its_json_is_stable(instance):
    from colonymind.reports import usage as report
    from colonymind.util.temporal import now_utc
    now = now_utc()
    _write(instance, _usage_rows(now))
    _captured(instance, 2)
    collected = report.collect(instance, since="7d")
    data = json.loads(report.to_json(collected))
    again = json.loads(report.to_json(report.collect(instance, since="7d")))
    assert {**again, "window": None} == {**data, "window": None}      # the same rows give the same numbers
    assert data["totals"] == {"calls": 8, "failed": 1, "input_tokens": 1960, "output_tokens": 288,
                              "without_input_tokens": 2, "without_output_tokens": 3}
    claims = next(line for line in data["by_task"] if line["task"] == "source_claim_extraction")
    assert {key: claims[key] for key in ("role", "client", "calls", "input_tokens", "output_tokens")} == {
        "role": "extraction", "client": "router", "calls": 2, "input_tokens": 580, "output_tokens": 78}
    assert claims["latency_ms"] == {"p50": 1000, "p90": 1080}
    deliberate = next(line for line in data["by_task"] if line["task"] == "mind_deliberate")
    assert deliberate["outcomes"] == {"ok": 1, "timeout": 1}
    assert {line["endpoint"]: line["calls"] for line in data["by_endpoint"]} == {
        "quick": 4, "embed": 1, "steady": 2, "local": 1}
    memory = data["memory"]
    assert memory["captured_messages"] == 2
    assert (memory["calls"], memory["input_tokens"], memory["output_tokens"]) == (5, 960, 88)
    # The commitment call reported its input tokens but not its output: the output estimate is a lower bound.
    assert (memory["without_input_tokens"], memory["without_output_tokens"]) == (0, 1)
    assert memory["per_message"] == {"calls": 2.5, "input_tokens": 480.0, "output_tokens": 44.0,
                                     "tokens": {"input": "reported", "output": "partial"}}
    assert list(data) == sorted(data) and data["window"]["span"] == "the 7 days"


def test_a_memory_cost_is_unknown_or_partial_when_calls_reported_no_tokens(instance):
    """A call whose endpoint reported no token count has an unknown cost, not a zero one: the per-message estimate
    says it is unknown when no memory call reported a count, and partial (a lower bound) when only some did."""
    from colonymind.reports import text
    from colonymind.reports import usage as report
    from colonymind.util.temporal import now_utc
    at = now_utc() - timedelta(hours=1)
    _write(instance, [_row(at, input_tokens=None, output_tokens=None) for _ in range(2)])
    _captured(instance, 2)
    memory = json.loads(report.to_json(report.collect(instance, since="7d")))["memory"]
    assert (memory["without_input_tokens"], memory["without_output_tokens"]) == (2, 2)
    assert memory["per_message"] == {"calls": 1.0, "input_tokens": None, "output_tokens": None,
                                     "tokens": {"input": "unknown", "output": "unknown"}}
    words = text.render(report.document(report.collect(instance, since="7d")), "md")
    assert "token counts are unknown" in words and "0 tokens in" not in words
    _write(instance, [_row(at, input_tokens=300, output_tokens=40)] * 2)
    memory = json.loads(report.to_json(report.collect(instance, since="7d")))["memory"]
    assert (memory["without_input_tokens"], memory["without_output_tokens"]) == (2, 2)
    assert memory["per_message"] == {"calls": 2.0, "input_tokens": 300.0, "output_tokens": 40.0,
                                     "tokens": {"input": "partial", "output": "partial"}}
    words = text.render(report.document(report.collect(instance, since="7d")), "md")
    assert "at least 300 tokens in and at least 40 out" in words
    assert "2 calls reported no input count, 2 no output count" in words


def test_a_call_missing_one_count_leaves_that_count_uncertain(instance):
    """A call that reported its input tokens and not its output tokens (or the reverse) leaves that side unknown:
    the per-message estimate, the totals and the health line keep it apart and never read it as zero."""
    from colonymind.reports import text
    from colonymind.reports import usage as report
    from colonymind.util.temporal import now_utc
    at = now_utc() - timedelta(hours=1)
    _write(instance, [_row(at, task="commitment_extract", input_tokens=100, output_tokens=None)])
    _captured(instance, 1)
    data = json.loads(report.to_json(report.collect(instance, since="7d")))
    assert data["memory"]["per_message"] == {"calls": 1.0, "input_tokens": 100.0, "output_tokens": None,
                                             "tokens": {"input": "reported", "output": "unknown"}}
    assert (data["totals"]["without_input_tokens"], data["totals"]["without_output_tokens"]) == (0, 1)
    words = text.render(report.document(report.collect(instance, since="7d")), "md")
    assert "0 out" not in words and "100 tokens in and an unknown number out" in words
    assert usage.health_line(instance) == ("Model calls in the last 24 hours: 1 (1 ok), 100 tokens in and an unknown "
                                           "number out (1 call reported no output count).")
    _write(instance, [_row(at, task="commitment_extract", input_tokens=None, output_tokens=6)])
    data = json.loads(report.to_json(report.collect(instance, since="7d")))
    assert data["memory"]["per_message"] == {"calls": 2.0, "input_tokens": 100.0, "output_tokens": 6.0,
                                             "tokens": {"input": "partial", "output": "partial"}}
    assert usage.last_day(instance)["without_input_tokens"] == 1


def test_the_report_cli_prints_json_and_plain_words(instance, capsys):
    from colonymind.reports import cli
    from colonymind.util.temporal import now_utc
    _write(instance, _usage_rows(now_utc()))
    args = Namespace(report_command="usage", since="24h", json=True, format="md", out=None)
    assert cli.run(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli.run(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert {**second, "window": None} == {**first, "window": None} and first["totals"]["calls"] == 8
    assert cli.run(Namespace(report_command="usage", since="24h", json=False, format="md", out=None)) == 0
    text = capsys.readouterr().out
    assert "Model usage" in text and "source_claim_extraction" in text and "8 model calls" in text
    assert "No messages were captured" in text


def test_an_instance_with_no_usage_yet_reports_zero_and_writes_nothing(instance):
    from colonymind.reports import usage as report
    data = json.loads(report.to_json(report.collect(instance, since="7d")))
    assert data["found"] is False and data["totals"]["calls"] == 0 and data["memory"]["per_message"] is None
    assert not (instance / usage.DB_FILE).exists()


@pytest.fixture
def locked(instance, monkeypatch):
    """A ledger with rows that another connection holds locked past the readers' wait (shortened for the test)."""
    from colonymind.util.temporal import now_utc
    _write(instance, _usage_rows(now_utc()))
    monkeypatch.setattr(usage, "READ_WAIT", 0.05)
    holder = sqlite3.connect(instance / usage.DB_FILE, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    yield instance
    holder.execute("ROLLBACK")
    holder.close()


def test_a_ledger_that_cannot_be_read_is_never_read_as_no_usage(locked):
    """A locked or damaged ledger is unreadable, not empty: the readers raise instead of answering zero, while a
    ledger whose tables are not written yet still reads as no usage."""
    from colonymind.util.temporal import now_utc
    with pytest.raises(usage.Unreadable, match="locked"):
        usage.rows(locked)
    with pytest.raises(usage.Unreadable):
        usage.captured_since(locked, now_utc() - timedelta(days=1))
    with pytest.raises(usage.Unreadable):
        usage.last_day(locked)
    with pytest.raises(usage.Unreadable):
        usage.health_line(locked)
    damaged = locked / "damaged"
    damaged.mkdir()
    (damaged / usage.DB_FILE).write_bytes(b"not a database, " * 64)
    with pytest.raises(usage.Unreadable):
        usage.rows(damaged)
    empty = locked / "empty"
    empty.mkdir()
    sqlite3.connect(empty / usage.DB_FILE).close()                # the file exists, no table yet
    assert usage.rows(empty) == [] and usage.captured_since(empty, now_utc() - timedelta(days=1)) == 0
    assert usage.health_line(empty) == "Model calls in the last 24 hours: none recorded."


def test_the_report_fails_clearly_on_a_ledger_it_cannot_read(locked, capsys):
    from colonymind.reports import cli
    assert cli.run(Namespace(report_command="usage", since="7d", json=True, format="md", out=None)) == 1
    out, err = capsys.readouterr()
    assert out == "" and "colonymind report usage: the model usage ledger could not be read" in err
    assert "locked" in err


@pytest.mark.asyncio
async def test_the_health_route_says_an_unreadable_ledger_failed_to_read(locked, monkeypatch):
    """The health note says the ledger could not be read, in words ``colonymind status`` shows as a warning, and
    never "none recorded"; the route itself still answers."""
    from colonymind import vector
    from colonymind.api.routers import host
    for name in ("_telemetry", "_commitment_store", "_embedder", "_embed_failure"):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setattr(vector, "_store", None)
    result = await host.health()
    assert result.notes["model_usage"] == "reading the model usage ledger failed (database is locked)"


def test_a_window_past_retention_is_cut_there_and_says_so(instance):
    from colonymind.reports import usage as report
    collected = report.collect(instance, since="60d")
    assert collected.cut and collected.span == "the 30 days usage is kept"
    assert json.loads(report.to_json(collected))["window"]["cut"] is True


@pytest.mark.asyncio
async def test_the_health_route_carries_the_last_days_totals(instance, monkeypatch):
    """``/v1/host/health`` has a line with the last 24 hours' model calls, in counts only; it is never read as a
    problem (``colonymind status`` shows notes that mention a failure or an error)."""
    from colonymind import vector
    from colonymind.api.routers import host
    from colonymind.util.temporal import now_utc
    for name in ("_telemetry", "_commitment_store", "_embedder", "_embed_failure"):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setattr(vector, "_store", None)
    assert (await host.health()).notes["model_usage"] == "Model calls in the last 24 hours: none recorded."
    _write(instance, _usage_rows(now_utc()))
    result = await host.health()
    assert result.notes["model_usage"] == ("Model calls in the last 24 hours: 8 (7 ok), at least 1,960 tokens in and "
                                           "at least 288 out (2 calls reported no input count, 3 no output count).")
    assert result.status == "ok" and result.problems == []
    assert not any(word in result.notes["model_usage"].lower() for word in ("fail", "error"))


def test_the_health_line_gives_the_last_days_totals(instance):
    from colonymind.util.temporal import now_utc
    now = now_utc()
    _write(instance, _usage_rows(now))
    line = usage.health_line(instance, now=now)
    assert line == "Model calls in the last 24 hours: 8 (7 ok), at least 1,960 tokens in and at least 288 out " \
                   "(2 calls reported no input count, 3 no output count)."
    assert usage.last_day(instance, now=now) == {"calls": 8, "failed": 1, "input_tokens": 1960, "output_tokens": 288,
                                                  "without_input_tokens": 2, "without_output_tokens": 3}
    assert usage.health_line(instance / "empty", now=now) == "Model calls in the last 24 hours: none recorded."


from colonymind.memory import curated
def test_erasing_a_source_never_changes_what_its_capture_cost(instance):
    """The per-message cost divides by the messages captured then, counted apart from the sources: erasing one
    source, or all of them, leaves the window's counts as they were. A replay is not a second capture."""
    from colonymind.reports import usage as report
    from colonymind.util.temporal import now_utc
    _write(instance, [_row(now_utc() - timedelta(hours=1), input_tokens=100, output_tokens=10)] * 2)
    ledger = _captured(instance, 2)
    assert not ledger.record_source('t0', contact_id='owner', session_id='s', messages=[{'role': 'user', 'content': 'message 0'}, {'role': 'assistant', 'content': 'noted'}])
    assert usage.flush()
    before = report.collect(instance, since='7d').memory
    assert before['captured_messages'] == 2
    assert before['per_message'] == {'calls': 1.0, 'input_tokens': 100.0, 'output_tokens': 10.0, 'tokens': {'input': 'reported', 'output': 'reported'}}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['t0'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert report.collect(instance, since='7d').memory == before
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['t1'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert report.collect(instance, since='7d').memory == before
    with sqlite3.connect(instance / usage.DB_FILE) as conn:
        columns = [line[1] for line in conn.execute('PRAGMA table_info(captures)')]
    assert columns == ['at', 'messages']
    assert b'owner' not in (instance / usage.DB_FILE).read_bytes()
