"""``/v1/host/health`` says which build the process runs: the commit, and whether its tree was clean.

The note is the build stamp in the words of doctor's ``build-stamp`` row, without the branch name, because the
route answers without a key. A process looks for its stamp once and reports what it found then.
"""
import asyncio
import json
import threading

import pytest
from httpx import ASGITransport, AsyncClient

from onekey import keyed_host_app
from colonymind import build_stamp, doctor
from colonymind.api.routers import host
from colonymind.api.schemas.host import HostHealthResponse
from test_health_problems import quiet_host  # noqa: F401  (fixture)

COMMIT = "0a1b2c3d4e" * 4
NO_STAMP = "no readable build stamp: the commit this install was built from is not recorded"
CLEAN = {"commit": COMMIT, "branch": "work", "dirty": False}
CHANGED = {"commit": COMMIT + COMMIT[:24], "branch": "topic/one", "dirty": True}
CLEAN_NOTE = f"commit {COMMIT[:12]}, clean"
CHANGED_NOTE = f"commit {COMMIT[:12]}, changed: the tree had uncommitted changes"


@pytest.fixture
def stamp(quiet_host, monkeypatch, tmp_path):
    """A stamp path of this test's own, and a process that has not looked for its stamp yet."""
    path = tmp_path / "build_stamp.json"
    monkeypatch.setattr(build_stamp, "PATH", path)
    monkeypatch.setattr(host, "_build_note", None, raising=False)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("held,note", [(CLEAN, CLEAN_NOTE), ({**CLEAN, "branch": ""}, CLEAN_NOTE),
                                       (CHANGED, CHANGED_NOTE)])
async def test_a_readable_stamp_is_noted_in_doctors_words_without_the_branch(stamp, held, note):
    stamp.write_text(json.dumps(held))
    assert (await host.health()).notes["build"] == note
    branch = f"branch {held['branch']}" if held["branch"] else "detached"
    assert doctor.check_build_stamp().detail == note.replace(", ", f", {branch}, ", 1)


@pytest.mark.asyncio
async def test_no_stamp_is_said_in_a_plain_sentence(stamp):
    assert not stamp.exists()
    assert (await host.health()).notes["build"] == NO_STAMP == doctor.check_build_stamp().detail


@pytest.mark.asyncio
@pytest.mark.parametrize("damaged", [
    None, "", "{", "[]", json.dumps({"commit": COMMIT, "branch": "work"}),
    json.dumps({**CLEAN, "commit": COMMIT[:12]}), json.dumps({**CLEAN, "dirty": "no"}),
    json.dumps({**CLEAN, "built": "today"})])
async def test_a_damaged_stamp_reads_as_no_stamp(stamp, damaged):
    if damaged is None:
        stamp.mkdir()                                    # a directory in the file's place: it cannot be read
    else:
        stamp.write_text(damaged)
    assert (await host.health()).notes["build"] == NO_STAMP


@pytest.mark.asyncio
async def test_a_reader_that_fails_reads_as_no_stamp(stamp, monkeypatch):
    def broken():
        raise OSError("synthetic")

    monkeypatch.setattr(build_stamp, "read", broken)
    assert (await host.health()).notes["build"] == NO_STAMP


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "the vector store did not open: ImportError: synthetic"])
async def test_the_stamp_changes_nothing_but_its_own_note(stamp, monkeypatch, failure):
    monkeypatch.setattr(host, "_embed_failure", failure)
    answers = []
    for held in (None, json.dumps(CLEAN), json.dumps(CHANGED), "{"):
        if held is not None:
            stamp.write_text(held)
        monkeypatch.setattr(host, "_build_note", None, raising=False)       # a new process, which looks again
        answers.append(await host.health())
    assert [answer.notes["build"] for answer in answers] == [NO_STAMP, CLEAN_NOTE, CHANGED_NOTE, NO_STAMP]
    without = answers[0]
    assert without.status == ("degraded" if failure else "ok")
    assert without.problems == (["semantic recall is off: " + failure] if failure else [])
    for answer in answers[1:]:
        assert (answer.status, answer.problems, answer.capabilities) == (
            without.status, without.problems, without.capabilities)
        assert {**answer.notes, "build": ""} == {**without.notes, "build": ""}


@pytest.mark.asyncio
async def test_the_stamp_is_read_once_per_process_and_off_the_event_loop(stamp, monkeypatch):
    stamp.write_text(json.dumps(CLEAN))
    read, reads = build_stamp.read, []

    def counted():
        reads.append(threading.get_ident())
        return read()

    monkeypatch.setattr(build_stamp, "read", counted)
    answers = list(await asyncio.gather(*(host.health() for _ in range(4))))     # the first calls arrive together
    stamp.write_text(json.dumps(CHANGED))                                       # a later stamp is not looked at
    answers += [await host.health() for _ in range(4)]
    assert {answer.notes["build"] for answer in answers} == {CLEAN_NOTE}
    assert len(reads) == 1 and reads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_the_route_answers_without_a_key_and_the_note_names_no_branch(stamp):
    stamp.write_text(json.dumps(CHANGED))
    transport = ASGITransport(app=keyed_host_app(), client=("192.0.2.10", 50000))    # the host routes behind a key
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/host/health")                         # no key, from another machine
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["notes"]["build"] == CHANGED_NOTE
    assert "topic/one" not in response.text
    assert set(body) == set(HostHealthResponse.model_fields)                    # no field was added to the answer
