"""The instance's own question set: built from a synthetic history into the state directory, and recall scored
against it with one result line. The history, the model's answers and every question here are made up."""
import argparse
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind import cli as colonymind_cli
from colonymind import recall_cli, recorder
from colonymind.api.errors import install_exception_handlers
from colonymind.api.routers import host
from colonymind.config import DEFAULTS, save_config
from colonymind.memory import curated, question_set
from colonymind.turns import get_turn_idempotency_ledger

CLIENT = httpx.Client
HISTORY = (
    ("t-c1", "casey", "The harbour ferry leaves at seven on weekdays."),
    ("t-c2", "casey", "Good morning!"),
    ("t-c3", "casey", "My dentist appointment moved to the ninth of March."),
    ("t-c4", "casey", "I parked the van behind the old bakery."),
    ("t-r1", "robin", "Our book club meets in the library annex."),
    ("t-r2", "robin", "Thanks, bye."),
)
# What the scripted model writes for a turn that states one of these: the question and the expected answer.
WRITTEN = {
    "harbour ferry": ("When does the harbour ferry leave on weekdays?", "At seven"),
    "dentist appointment": ("Which day did the dentist appointment move to?", "The ninth of March"),
    "the van": ("Truck location?", "Behind the old bakery"),          # shares no word with its turn: recall misses it
    "book club": ("Where does the book club meet?", "In the library annex"),
}


class Model:
    """Writes the scripted question for each turn it is shown, and two questions that must not be kept."""

    def __init__(self):
        self.shown = []

    async def complete(self, messages, *, context=None, **kwargs):
        assert context["task"] == question_set.TASK
        self.shown.append(messages[1]["content"])
        questions = [{"question": "Cites a turn that was not shown?", "answer": "No", "turns": ["t99"]},
                     {"question": "Has no answer?", "answer": "", "turns": ["t1"]}]
        for line in self.shown[-1].splitlines():
            name, _, said = line.partition(" | ")
            questions += [{"question": question, "answer": answer, "turns": [name]}
                          for words, (question, answer) in WRITTEN.items() if words in said]
        return SimpleNamespace(content=json.dumps({"questions": questions}))


class Silent:
    async def complete(self, messages, *, context=None, **kwargs):
        return SimpleNamespace(content=json.dumps({"questions": []}))


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Two people's turns in a fresh state directory, the real recall with no reranker and no vectors."""
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    from colonymind import vector
    monkeypatch.setattr(vector, "get_store", lambda: None)
    monkeypatch.setattr(vector, "get_pipeline", lambda: None)
    model = Model()
    monkeypatch.setattr(host, "_llm_router", model)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(question_set.router)
    ledger = get_turn_idempotency_ledger(tmp_path)
    for turn, person, said in HISTORY:
        ledger.record_source(turn, contact_id=person, session_id="s-" + person, derive_claims=False, messages=[
            {"role": "user", "content": said}, {"role": "assistant", "content": f"Noted ({turn})."}])
    return SimpleNamespace(app=app, ledger=ledger, model=model, file=tmp_path / question_set.FILE)


async def scored(app, **body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/mind/recall-questions", json=body)


def command(monkeypatch, tmp_path, answer, *argv):
    """``colonymind recall-questions`` against a sidecar that gives ``answer``: its exit code and what it sent."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    save_config(dict(DEFAULTS), home)
    (home / "api.key").write_text("k\n")
    sent = []

    def handler(request):
        sent.append((request.url.path, json.loads(request.content)))
        return httpx.Response(answer.status_code, json=answer.json())

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: CLIENT(transport=httpx.MockTransport(handler)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=None)
    recall_cli.add_parser(parser.add_subparsers(dest="command"))
    return recall_cli.run_questions(parser.parse_args(["recall-questions", *argv])), sent


async def test_a_set_is_built_from_the_history_into_the_state_directory_and_recall_is_scored(
        world, monkeypatch, tmp_path, capsys):
    first = await scored(world.app)
    assert first.status_code == 200, first.text
    assert first.json() == {"questions": 4, "recalled": 3, "built": 4}
    # The set is one private file in the state directory: a question, its expected answer, the turns that hold it.
    assert world.file.parent == tmp_path and stat.S_IMODE(world.file.stat().st_mode) == 0o600
    assert [json.loads(line) for line in world.file.read_text(encoding="utf-8").splitlines()] == [
        {"question": "When does the harbour ferry leave on weekdays?", "answer": "At seven", "turns": ["t-c1"]},
        {"question": "Which day did the dentist appointment move to?", "answer": "The ninth of March",
         "turns": ["t-c3"]},
        {"question": "Truck location?", "answer": "Behind the old bakery", "turns": ["t-c4"]},
        {"question": "Where does the book club meet?", "answer": "In the library annex", "turns": ["t-r1"]}]
    # The package holds no questions.
    assert not list(Path(question_set.__file__).resolve().parents[1].rglob(question_set.FILE))
    # Each call showed one person's turns under short ids; a question that cites none of them, or has no answer,
    # was not kept.
    casey, robin = world.model.shown
    assert "t1 | They said: The harbour ferry leaves at seven on weekdays. / The agent said: Noted (t-c1)." in casey
    assert "book club" not in casey and "harbour" not in robin
    line = recorder.recent(1, kind="check", source=question_set.SOURCE)[0]
    assert (line["result"], line["detail"]) == ("done", "recalled-3-of-4") and line["took_ms"] is not None

    # The set stays as built: the next run asks the same questions and the model is not called.
    again = await scored(world.app)
    assert again.json() == {"questions": 4, "recalled": 3, "built": 0} and len(world.model.shown) == 2
    # ``new`` builds it again first.
    new = await scored(world.app, new=True)
    assert new.json() == {"questions": 4, "recalled": 3, "built": 4} and len(world.model.shown) == 4

    # The command prints the one result line.
    assert 'args.command == "recall-questions"' in open(colonymind_cli.__file__, encoding="utf-8").read()
    assert command(monkeypatch, tmp_path, again) == (0, [("/v1/mind/recall-questions", {"new": False})])
    assert capsys.readouterr().out == "recall questions: 3 of 4 recalled\n"
    assert command(monkeypatch, tmp_path, new, "--new") == (0, [("/v1/mind/recall-questions", {"new": True})])
    assert capsys.readouterr().out == "recall questions: 3 of 4 recalled (set built: 4 questions)\n"


async def test_a_question_leaves_the_file_when_a_turn_it_names_is_forgotten(world):
    assert (await scored(world.app)).json()["questions"] == 4

    curated.forget_sources(world.ledger, "casey", source_ids=["t-c3"])

    standing = question_set.load(world.ledger)                     # what every run and the nightly pass read
    assert [(row["turns"], person) for row, person in standing] == [
        (["t-c1"], "casey"), (["t-c4"], "casey"), (["t-r1"], "robin")]
    assert "dentist" not in world.file.read_text(encoding="utf-8")
    assert (await scored(world.app)).json() == {"questions": 3, "recalled": 2, "built": 0}


async def test_without_a_model_no_set_is_built_and_the_run_is_recorded_as_failed(
        world, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(host, "_llm_router", None)
    refused = await scored(world.app)
    assert refused.status_code == 503 and refused.json()["detail"]["code"] == "no_model"
    assert not world.file.exists()
    line = recorder.recent(1, kind="check", source=question_set.SOURCE)[0]
    assert (line["result"], line["detail"]) == ("failed", "nomodel")
    # A history the model writes no question for gives no set either, and the run says so.
    monkeypatch.setattr(host, "_llm_router", Silent())
    empty = await scored(world.app)
    assert empty.json() == {"questions": 0, "recalled": 0, "built": 0} and not world.file.exists()

    assert command(monkeypatch, tmp_path, refused)[0] == 1
    assert "none is configured" in capsys.readouterr().err
    assert command(monkeypatch, tmp_path, empty)[0] == 1
    assert capsys.readouterr().out == "recall questions: none (the history gave no question to ask)\n"
