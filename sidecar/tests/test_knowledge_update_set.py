"""KU20, the pre-registered knowledge-update set for the claim priors (``benchmarks/knowledge_update``).

Offline only: the set's own rules, and the runner end to end with a scripted extractor that links a change exactly
when its record is offered, and an embedding that knows which two messages belong together. The measurement
itself runs later against real endpoints; these tests show the set exercises the case it was built for (a keyword
hit on other claims, never on the record the change is about) and that the runner tells the two builds apart.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from colonymind.beliefs.source_claims import _CHANGE, norm_value
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.source_vectors import SourceVectors
from colonymind.vector.indexes import EmbeddingIdentity, IndexCatalog
from colonymind.vector.store import VectorStore

RUN = Path(__file__).resolve().parents[2] / "benchmarks" / "knowledge_update" / "run.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("knowledge_update_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ku(runner):
    return runner.load_set()


# The words a keyword search of a message looks for: longer than two characters, not one of its stop words.
SEARCH_STOP = {"the", "and", "that", "this", "what", "when", "where", "how", "you", "your", "was", "were", "are",
               "for", "with", "remember", "about"}


def words(text):
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) > 2 and word not in SEARCH_STOP}


def share_a_root(a, b):
    return a.startswith(b) or b.startswith(a) or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5])


def test_the_set_is_the_frozen_one(runner, tmp_path):
    assert hashlib.sha256(runner.SET.read_bytes()).hexdigest() == runner.SET_SHA256
    edited = tmp_path / "ku20.json"
    edited.write_text(runner.SET.read_text().replace("Berlin", "Bremen"))
    with pytest.raises(ValueError, match="frozen"):
        runner.load_set(edited)


def test_the_set_has_its_registered_shape(runner, ku):
    assert (ku["name"], len(ku["pairs"]), len(ku["controls"])) == ("KU20", 20, 10)
    assert len(ku["fillers"]) >= runner.MIN_OTHER_CLAIMS + 10
    ids = [row["id"] for row in ku["fillers"] + ku["pairs"] + ku["controls"]]
    texts = [row["text"] for row in ku["fillers"]] + [item[part]["text"] for item, _ in runner.items(ku)
                                                      for part in ("first", "second")]
    assert len(set(ids)) == len(ids) and len(set(texts)) == len(texts)
    last_formed = max(at for _, at, _ in runner.formation_records(ku))
    assert all(item["second"]["at"] > last_formed for item, _ in runner.items(ku))
    for item in ku["pairs"]:
        assert item["gold"]["first_value"] in item["first"]["text"]
        assert item["gold"]["second_value"] in item["second"]["text"]
    for item in ku["controls"]:
        assert item["gold"]["second_value"] in item["second"]["text"]


def test_each_second_shares_no_root_with_its_first_and_carries_a_change_cue(runner, ku):
    for item, _ in runner.items(ku):
        first, second = words(item["first"]["text"]), words(item["second"]["text"])
        assert not {(a, b) for a in first for b in second if share_a_root(a, b)}, item["id"]
        assert _CHANGE.search(item["second"]["text"]), item["id"]


def test_a_keyword_search_of_each_second_finds_other_records_and_never_its_first(runner, ku, tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    for turn, at, text in runner.formation_records(ku):
        ledger.record_source(turn, contact_id=ku["contact_id"], session_id=turn, occurred_at=at,
                             messages=[{"role": "user", "content": text}], derive_claims=False)
    for item, _ in runner.items(ku):
        hits = ledger.search_sources(item["second"]["text"], contact_id=ku["contact_id"],
                                     session_id=runner.SESSION, limit=10)
        assert hits, item["id"]
        assert runner.first_turn(item) not in {hit["turn_id"] for hit in hits}, item["id"]


# -- the runner, end to end, offline -----------------------------------------------------------------------------

def subject(text):
    return "I" if re.search(r"\b(i|my|mine)\b", text, re.I) else " ".join(text.split()[:2])


def proposal(text, subj, predicate, value, **changes):
    return {"subject": subj, "predicate": predicate, "value": value, "evidence": text, "operation": "assert",
            "prior_claim_id": None, "memory_kind": "personal_context",
            "recall_reason": "Use this stated detail when helping the person later.",
            "valid_from_text": None, "valid_to_text": None, "event_at_text": None, **changes}


class IdealExtractor:
    """Links a change to the offered record of its property, exactly when that record is offered."""

    def __init__(self, ku):
        self.outputs = {}
        for filler in ku["fillers"]:
            text = filler["text"]
            self.outputs[text] = proposal(text, subject(text), "note " + filler["id"], text.rstrip(".").split()[-1])
        for item in ku["pairs"]:
            first, second, gold = item["first"]["text"], item["second"]["text"], item["gold"]
            self.outputs[first] = proposal(first, subject(first), gold["property"], gold["first_value"])
            self.outputs[second] = proposal(second, subject(first), gold["property"], gold["second_value"],
                                            operation="change", choose=gold["property"])
        for item in ku["controls"]:
            first, second, gold = item["first"]["text"], item["second"]["text"], item["gold"]
            self.outputs[first] = proposal(first, subject(first), gold["first_property"], gold["first_value"])
            self.outputs[second] = proposal(second, subject(second), gold["second_property"], gold["second_value"])

    def tier_config(self, tier):
        return SimpleNamespace(base_url="http://127.0.0.1:8080/v1")

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        if kwargs.get("context", {}).get("task") == "source_claim_review":
            return SimpleNamespace(model_id="fixture", usage=usage, content=json.dumps({
                str(row["index"]): {"keep": True, "reason": "Scripted review keeps it."} for row in payload["proposals"]}))
        result = dict(self.outputs[payload["message"]])
        wanted = result.pop("choose", None)
        if wanted:
            result["prior_claim_id"] = next((row["id"] for row in payload["prior_assertions"]
                                             if row["predicate"] == norm_value(wanted)), None)
        return SimpleNamespace(model_id="fixture", usage=usage, content=json.dumps([result]))


class Knowing:
    """An embedding that puts each item's two messages in one direction and every other message in another."""

    def __init__(self, ku):
        self.index_identity = EmbeddingIdentity("fixture", "fixture", "unknown", 31)
        self.dimension = {}
        for index, item in enumerate(ku["pairs"] + ku["controls"]):
            self.dimension[item["first"]["text"]] = self.dimension[item["second"]["text"]] = index

    def vector(self, text):
        index = self.dimension.get(text, 30)
        return [float(i == index) for i in range(31)]

    async def embed_batch(self, texts):
        return [self.vector(text) for text in texts]

    async def embed_query(self, text):
        return self.vector(text)


@pytest.mark.asyncio
async def test_the_runner_tells_a_build_with_the_vector_lane_from_one_without(runner, ku, tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    pipeline = Knowing(ku)
    store = VectorStore(str(tmp_path / "lancedb"), identity=pipeline.index_identity, catalog=IndexCatalog(ledger))
    await store.connect(31)
    await store.ensure_collections(31)
    vectors = SourceVectors(ledger, store, pipeline)
    model = IdealExtractor(ku)
    projection = SourceClaimProjection(ledger, vectors=vectors)
    await runner.form_store(ledger, projection, model, ku, vectors=vectors)
    firsts, total = runner.formed_store(ledger, ku)
    assert all(len(ids) == 1 for ids in firsts.values())
    other = min(total - len(ids) for ids in firsts.values())
    assert other == len(ku["fillers"]) + len(firsts) - 1 >= runner.MIN_OTHER_CLAIMS
    targets = runner.target_claims(ledger, ku)
    assert targets == {ident: {"status": "found", "ids": ids} for ident, ids in firsts.items()}

    with_lane = runner.score(await runner.measure_dry_run(projection, model, ku, targets),
                             other_claims_min=other, build={"path": "dry_run"})
    assert (with_lane["offered"], with_lane["superseded"], with_lane["wrong_target"]) == (20, 20, 0)
    assert with_lane["over_linked"] == 0 and with_lane["store"]["valid"]
    assert with_lane["review_rejection_rate"] == 0 and with_lane["usage"]["calls"] == 60
    assert with_lane["not_complete"] == [] and with_lane["not_scored"] == {} and with_lane["change_pairs"] == 20

    without = runner.score(await runner.measure_by_commit(ledger, model, ku, targets, tmp_path / "pairs"),
                           other_claims_min=other, build={"path": "commit"})
    assert (without["offered"], without["superseded"], without["over_linked"]) == (0, 0, 0)
    # Nothing either measurement did reached the formed store.
    assert runner.formed_store(ledger, ku) == (firsts, total)


def test_the_score_counts_over_linking_wrong_targets_and_a_thin_store(runner):
    def row(ident, kind, **changes):
        return {"id": ident, "kind": kind, "status": "complete", "target": "found", "offered": True,
                "proposed_link": True, "superseded": False, "retired_other": [], "held_for_owner": [],
                "reviewed": 2, "review_rejected": 0, "rejections": [], "usage": {"calls": 2}, **changes}
    rows = [row("p01", "change", superseded=True), row("p02", "change", retired_other=["claim:x"]),
            row("p03", "change", target="ambiguous", superseded=False, retired_other=["claim:z"]),
            row("c01", "control", retired_other=["claim:y"], review_rejected=1), row("c02", "control")]
    summary = runner.score(rows, other_claims_min=49, build={})
    assert (summary["superseded"], summary["wrong_target"], summary["over_linked"]) == (1, 1, 1)
    assert summary["superseded_share"] == 0.5 and summary["over_link_share"] == 0.5
    # A pair whose target could not be told apart is not scored, and is listed.
    assert summary["change_pairs"] == 2 and summary["not_scored"] == {"p03": "target_ambiguous"}
    assert summary["review_rejection_rate"] == 0.1
    assert summary["store"] == {"other_claims_min": 49, "floor": 50, "valid": False}


def test_the_campaign_reads_only_benchmark_variables_and_refuses_a_used_directory(runner, tmp_path, monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("COLONYMIND_BENCH_"):
            monkeypatch.delenv(name)
    with pytest.raises(ValueError, match="COLONYMIND_BENCH_CHAT_BASE_URL"):
        runner.environment("commit")
    monkeypatch.setenv("COLONYMIND_BENCH_CHAT_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("COLONYMIND_BENCH_CHAT_MODEL", "fixture")
    assert runner.environment("commit") == {"CHAT_BASE_URL": "http://127.0.0.1:9/v1", "CHAT_MODEL": "fixture"}
    with pytest.raises(ValueError, match="COLONYMIND_BENCH_EMBED_BASE_URL"):
        runner.environment("dry_run")
    config = runner.router_config(runner.environment("commit"))
    assert set(config["functionRoles"]) == {"extraction", "judging"}
    used = tmp_path / "used"
    used.mkdir()
    (used / "something").write_text("x")
    import asyncio
    with pytest.raises(ValueError, match="new empty"):
        asyncio.run(runner.main(["--state-dir", str(used), "--out", str(tmp_path / "out.json")]))


class Several(IdealExtractor):
    """The ideal extractor, except that a first message can state several properties (``several``)."""

    def __init__(self, ku):
        super().__init__(ku)
        self.several = {}

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        if kwargs.get("context", {}).get("task") != "source_claim_review" and payload["message"] in self.several:
            return SimpleNamespace(model_id="fixture", usage={}, content=json.dumps(self.several[payload["message"]]))
        return await super().complete(messages, **kwargs)


def claim_value(ledger, ident):
    with closing(ledger._connect()) as conn:
        return json.loads(conn.execute("SELECT data_json FROM source_claims WHERE id=?", (ident,)).fetchone()[0])["value"]


async def formed(runner, ku, tmp_path, model):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    pipeline = Knowing(ku)
    store = VectorStore(str(tmp_path / "lancedb"), identity=pipeline.index_identity, catalog=IndexCatalog(ledger))
    await store.connect(31)
    await store.ensure_collections(31)
    vectors = SourceVectors(ledger, store, pipeline)
    projection = SourceClaimProjection(ledger, vectors=vectors)
    await runner.form_store(ledger, projection, model, ku, vectors=vectors)
    return ledger, projection


@pytest.mark.asyncio
async def test_only_the_claim_of_the_property_a_pair_is_about_counts_as_its_target(runner, ku, tmp_path):
    pairs = {item["id"]: item for item in ku["pairs"]}
    small = dict(ku, fillers=ku["fillers"][:5], pairs=[pairs[ident] for ident in ("p01", "p03", "p08")],
                 controls=[])
    model = Several(small)
    # p03's first message states two properties: the gym, and the day of training. Its change retires the day.
    gym, switched = pairs["p03"]["first"]["text"], pairs["p03"]["second"]["text"]
    model.several[gym] = [proposal(gym, "I", "gym", "Riverside gym"), proposal(gym, "I", "training day", "Tuesdays")]
    model.outputs[switched] = proposal(switched, "I", "training day", "climbing hall", operation="change",
                                       choose="training day")
    # p08: two claims hold the value, one exactly: that one is the target. p01: none holds it (missing).
    dentist = pairs["p08"]["first"]["text"]
    model.several[dentist] = [proposal(dentist, "I", "dentist", "Dr. Okafor"),
                              proposal(dentist, "I", "dentist practice", "Dr. Okafor on Elm Street")]
    office = pairs["p01"]["first"]["text"]
    model.outputs[office] = proposal(office, "I", "office location", "the office")
    ledger, projection = await formed(runner, small, tmp_path, model)
    firsts, _ = runner.formed_store(ledger, small)
    targets = runner.target_claims(ledger, small)
    day = next(ident for ident in firsts["p03"] if ident not in targets["p03"]["ids"])
    assert targets["p03"]["status"] == "found" and len(firsts["p03"]) == 2
    exact = next(ident for ident in firsts["p08"] if claim_value(ledger, ident) == "Dr. Okafor")
    assert targets["p08"] == {"status": "found", "ids": [exact]}
    assert targets["p01"] == {"status": "missing", "ids": []}
    measured = await runner.measure_dry_run(projection, model, small, targets)
    row = next(row for row in measured if row["id"] == "p03")
    assert (row["offered"], row["superseded"], row["retired_other"]) == (True, False, [day])
    summary = runner.score(measured, other_claims_min=50, build={})
    assert (summary["superseded"], summary["wrong_target"]) == (1, 1)          # p08, and p03
    assert summary["change_pairs"] == 2 and summary["not_scored"] == {"p01": "target_missing"}
    # By commit the same targets are scored: retiring p03's day never counts as changing its gym.
    measured = await runner.measure_by_commit(ledger, model, small, targets, tmp_path / "pairs")
    row = next(row for row in measured if row["id"] == "p03")
    assert row["superseded"] is False and set(row["retired_other"]) <= {day}
    summary = runner.score(measured, other_claims_min=50, build={})
    assert summary["change_pairs"] == 2 and summary["not_scored"] == {"p01": "target_missing"}


def test_a_target_held_by_two_claims_equally_is_ambiguous(runner, ku, tmp_path):
    from colonymind.beliefs.source_projection import SourceClaimProjection as _  # noqa: F401  (the sidecar is importable)
    item = next(item for item in ku["pairs"] if item["id"] == "p12")
    small = dict(ku, fillers=[], pairs=[item], controls=[])
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    text = item["first"]["text"]
    ledger.record_source(runner.first_turn(item), contact_id=ku["contact_id"], session_id="s",
                         messages=[{"role": "user", "content": text}], derive_claims=False)
    with closing(ledger._connect()) as conn, conn:
        for ident, predicate, value in (("claim:a", "instrument", "the cello"), ("claim:b", "hobby", "cello quartet")):
            conn.execute("INSERT INTO source_claims(id,turn_id,message_hash,subject_key,predicate,value_key,data_json) "
                         "VALUES (?,?,?,?,?,?,?)", (ident, runner.first_turn(item), "h", "speaker", predicate,
                                                     norm_value(value), json.dumps({"value": value})))
    assert runner.target_claims(ledger, small)["p12"] == {"status": "ambiguous", "ids": ["claim:a", "claim:b"]}


class Linking:
    """Scripted extraction that links a proposal marked ``choose`` to the first record it is offered."""

    def __init__(self, outputs):
        self.outputs = outputs

    def tier_config(self, tier):
        return SimpleNamespace(base_url="http://127.0.0.1:8080/v1")

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        if kwargs.get("context", {}).get("task") == "source_claim_review":
            return SimpleNamespace(model_id="fixture", usage={}, content=json.dumps({
                str(row["index"]): {"keep": True, "reason": "Scripted review keeps it."} for row in payload["proposals"]}))
        result = dict(self.outputs[payload["message"]])
        if result.pop("choose", None):
            result["prior_claim_id"] = payload["prior_assertions"][0]["id"]
        return SimpleNamespace(model_id="fixture", usage={}, content=json.dumps([result]))


@pytest.mark.asyncio
async def test_a_preference_change_held_for_the_owner_is_reported_by_both_arms(runner, tmp_path, monkeypatch):
    from colonymind.memory import learning
    from test_source_quoted_preferences import TEXT, preference
    monkeypatch.setenv(learning.FLAG, "1")
    # The change names nothing of what the handoff preference was about, so it waits for the owner.
    changed = "Correction: I prefer quiet rooms."
    held = {"contact_id": "owner", "timezone": "UTC", "fillers": [], "controls": [], "pairs": [{
        "id": "h01", "first": {"at": "2026-05-01T10:00:00+00:00", "text": TEXT},
        "second": {"at": "2026-06-01T10:00:00+00:00", "text": changed},
        "gold": {"property": "handoff format", "first_value": TEXT, "second_value": changed}}]}
    model = Linking({TEXT: preference(), changed: preference(changed, operation="correct") | {"choose": True}})
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    projection = SourceClaimProjection(ledger)
    await runner.form_store(ledger, projection, model, held)
    targets = runner.target_claims(ledger, held)
    first, = targets["h01"]["ids"]
    for path, measured in (("dry_run", await runner.measure_dry_run(projection, model, held, targets)),
                           ("commit", await runner.measure_by_commit(ledger, model, held, targets, tmp_path / "pairs"))):
        row, = measured
        assert row["proposed_link"] and not row["superseded"] and row["held_for_owner"] == [first], (path, row)
        assert runner.score(measured, other_claims_min=0, build={})["held_for_owner"] == 1
