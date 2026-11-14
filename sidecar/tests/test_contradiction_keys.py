"""The bound for contradiction candidates by meaning: the labelled dev set and the bound's selection.

The night pairs two claims of one person under different keys when the keys' embeddings are at least the bound
measured for its embedding alike (``KEY_SIMILARITY_BOUND`` for one never measured). A bound is chosen on a frozen set
of 60 synthetic labelled pairs; these tests pin
the set (its size, balance, shape, labels and hash), that it is scored with exactly the key text the night embeds,
and the selection rule, and that every measured bound is tied to its embedding and recomputes from its recorded
scores. Here every embedder is a stub or a mock endpoint.
"""

from __future__ import annotations

import asyncio
import json
import math
import re

import pytest

from colonymind.beliefs.source_claims import literal_subject, norm_value
from colonymind.mind import contradiction_keys as bound
from colonymind.mind.consolidate import KEY_SIMILARITY_BOUND, key_text


class TableEmbedder:
    """Unit vectors by topic, recording each batch: key texts on one topic are identical, others orthogonal."""

    def __init__(self, topics=None):
        self.topics, self.batches, self.axes = dict(topics or {}), [], {}

    async def embed_batch(self, texts):
        self.batches.append(list(texts))
        vectors = []
        for text in texts:
            values = [0.0] * 256
            values[self.axes.setdefault(self.topics.get(text, text), len(self.axes))] = 1.0
            vectors.append(values)
        return vectors


# ---------------------------------------------------------------------------
# The dev set: 60 labelled pairs, frozen by hash
# ---------------------------------------------------------------------------

def test_the_dev_set_is_sixty_labelled_pairs_half_the_same_fact():
    pairs = bound.load_dev_set()
    assert len(pairs) == 60 and len({pair["id"] for pair in pairs}) == 60
    assert sum(pair["same_fact"] for pair in pairs) == 30
    assert all(isinstance(pair["same_fact"], bool) for pair in pairs)


def test_every_pair_is_two_keys_two_values_and_statements_that_ground_them():
    """Each pair is what the night would compare: two claims of one person under two different keys (as the claim
    extractor files them) with two different values, each value and subject found in its own statement."""
    for pair in bound.load_dev_set():
        a, b = pair["a"], pair["b"]
        assert bound.claim_key(a["subject"], a["predicate"]) != bound.claim_key(b["subject"], b["predicate"]), pair
        assert norm_value(a["value"]) != norm_value(b["value"]), pair
        for side in (a, b):
            assert side["value"].casefold() in side["said"].casefold(), side
            assert literal_subject(side["subject"], side["said"]), side
            assert "\u2014" not in side["said"]


UPDATE = re.compile(r"\b(now|actually)\b", re.I)


def test_every_same_fact_pair_is_an_unambiguous_update_and_no_other_pair_reads_as_one():
    """A same-fact pair is one fact whose newer statement replaces the older, and says so: the value changed ("now")
    or was corrected ("actually"). A pair whose statements could both stand as said (a team and a department, two
    weekly meetings) is not the same fact, and a two-facts pair never reads as an update."""
    for pair in bound.load_dev_set():
        said = [pair["a"]["said"], pair["b"]["said"]]
        if pair["same_fact"]:
            assert UPDATE.search(said[1]), pair["id"]
        else:
            assert not any(UPDATE.search(text) for text in said), pair["id"]


def test_the_claim_key_is_the_extractors_key():
    assert bound.claim_key("I", "works_at") == ("speaker", "works at")
    assert bound.claim_key("i", "Home_City") == ("speaker", "home city")
    assert bound.claim_key("My Sister", "employer") == ("my sister", "employer")


def test_a_changed_dev_set_is_refused(tmp_path):
    text = bound.DEV_SET.read_text()
    changed = tmp_path / "dev.json"
    changed.write_text(text.replace('"same_fact": true', '"same_fact": false', 1))
    with pytest.raises(ValueError, match="sha256"):
        bound.load_dev_set(changed)
    assert len(bound.load_dev_set(changed, sha256=None)) == 60     # an explicit unpinned read, for a new set


def test_the_dev_set_uses_only_made_up_numbers_and_addresses():
    """Fixtures use 555 numbers and example domains."""
    text = bound.DEV_SET.read_text()
    numbers = re.findall(r"\b\d{3}-\d{4}\b", text)
    assert numbers and all(number.startswith("555-") for number in numbers)
    domains = re.findall(r"@([\w-]+(?:\.[\w-]+)+)", text)
    assert domains and set(domains) <= {"example.com", "example.org"}
    assert "\u2014" not in text


# ---------------------------------------------------------------------------
# Scoring: the night's key text, one batch
# ---------------------------------------------------------------------------

async def test_the_dev_set_is_scored_with_the_key_text_the_night_embeds_in_one_batch():
    pairs = bound.load_dev_set()
    embedder = TableEmbedder()
    scored = await bound.score_dev_set(embedder, pairs)
    batch, = embedder.batches
    expected = {key_text(*bound.claim_key(side["subject"], side["predicate"]))
                for pair in pairs for side in (pair["a"], pair["b"])}
    assert sorted(batch) == sorted(expected) and len(batch) == len(set(batch))
    assert [row["id"] for row in scored] == [pair["id"] for pair in pairs]
    assert all(row["score"] == 0.0 for row in scored)                     # every key its own topic


async def test_a_pair_scores_the_cosine_of_its_two_keys():
    pair = {"id": "p", "same_fact": True,
            "a": {"subject": "I", "predicate": "works_at", "value": "Acme", "said": "I work at Acme."},
            "b": {"subject": "I", "predicate": "employer", "value": "Globex", "said": "My employer is Globex."}}
    embedder = TableEmbedder({key_text("speaker", "works at"): "job", key_text("speaker", "employer"): "job"})
    row, = await bound.score_dev_set(embedder, [pair])
    assert row == {"id": "p", "same_fact": True, "score": pytest.approx(1.0)}


async def test_an_embedder_that_answers_the_wrong_number_of_vectors_is_an_error():
    class Short(TableEmbedder):
        async def embed_batch(self, texts):
            return (await super().embed_batch(texts))[:-1]
    with pytest.raises(ValueError, match="vectors"):
        await bound.score_dev_set(Short(), bound.load_dev_set())


# ---------------------------------------------------------------------------
# Selection: the lowest bound that keeps the precision floor, highest recall
# ---------------------------------------------------------------------------

def rows(positives, negatives):
    return ([{"id": f"p{i}", "same_fact": True, "score": s} for i, s in enumerate(positives)]
            + [{"id": f"n{i}", "same_fact": False, "score": s} for i, s in enumerate(negatives)])


def test_the_bound_is_the_lowest_that_keeps_the_precision_floor_set_between_two_scores():
    chosen = bound.select_bound(rows([0.9, 0.85, 0.7, 0.6], [0.8, 0.5, 0.4]), precision_floor=0.8)
    assert chosen["floor_met"] is True
    assert chosen["bound"] == pytest.approx(0.55)                          # between 0.6 (kept) and 0.5 (left out)
    at = chosen["at_bound"]
    assert (at["tp"], at["fp"], at["fn"], at["tn"]) == (4, 1, 0, 2)
    assert at["precision"] == pytest.approx(0.8) and at["recall"] == 1.0
    thresholds = [row["threshold"] for row in chosen["curve"]]
    assert thresholds == sorted(thresholds, reverse=True) and len(thresholds) == 7


def test_ties_between_equal_recall_keep_the_higher_bound():
    """0.77 and 0.5 both keep every same-fact pair above the floor; the higher lets no other pair through."""
    chosen = bound.select_bound(rows([0.9, 0.8, 0.79, 0.78, 0.77], [0.5, 0.4]), precision_floor=0.8)
    assert chosen["bound"] == pytest.approx(0.635) and chosen["at_bound"]["fp"] == 0


def test_without_a_bound_that_keeps_the_floor_the_best_f1_is_chosen_and_says_so():
    chosen = bound.select_bound(rows([0.6, 0.4], [0.9, 0.8, 0.7]), precision_floor=0.8)
    assert chosen["floor_met"] is False
    best = max(row["f1"] for row in chosen["curve"])
    assert chosen["at_bound"]["f1"] == pytest.approx(best)


def test_no_bound_is_chosen_when_no_threshold_finds_a_same_fact_pair():
    chosen = bound.select_bound(rows([], [0.4, 0.3]), precision_floor=0.8)
    assert chosen["bound"] is None and chosen["floor_met"] is False and chosen["at_bound"] is None
    assert len(chosen["curve"]) == 2


async def test_the_report_names_the_set_its_hash_the_chosen_and_the_default_bound():
    topics = {}
    pairs = bound.load_dev_set()
    for pair in pairs:
        if pair["same_fact"]:
            for side in (pair["a"], pair["b"]):
                topics[key_text(*bound.claim_key(side["subject"], side["predicate"]))] = pair["id"]
    report = await bound.run(TableEmbedder(topics), label="stub")
    assert report["dev_set"]["sha256"] == bound.DEV_SET_SHA256 and report["dev_set"]["pairs"] == 60
    assert report["embedder"] == "stub" and report["precision_floor"] == bound.PRECISION_FLOOR
    assert report["default_bound"] == KEY_SIMILARITY_BOUND and "at_default_bound" in report
    assert len(report["pairs"]) == 60
    json.dumps(report)                                                    # a plain JSON document
    assert math.isfinite(report["bound"])


def test_the_command_line_writes_the_report_without_printing_the_key(tmp_path, monkeypatch, capsys):
    made = {}

    async def fake_embedder(provider, base_url, model, dims, request_dims, revision):
        made.update(provider=provider, base_url=base_url, model=model, dims=dims, request_dims=request_dims)
        return TableEmbedder()

    for name in ("COLONYMIND_EMBED_PROVIDER", "COLONYMIND_EMBED_DIMS", "COLONYMIND_EMBED_REQUEST_DIMS",
                 "COLONYMIND_EMBED_REVISION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bound, "_sidecar_embedder", fake_embedder)
    monkeypatch.setenv("COLONYMIND_EMBED_API_KEY", "fixture-key-never-printed")
    out = tmp_path / "report.json"
    assert bound.main(["--base-url", "http://127.0.0.1:9/v1", "--model", "embed-model", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert made == {"provider": "openai_api", "base_url": "http://127.0.0.1:9/v1", "model": "embed-model", "dims": 0,
                    "request_dims": None}
    assert report["embedder"] == "embed-model" and report["dev_set"]["pairs"] == 60
    assert report["embedding"] is None                                    # a stub has no embedding identity
    printed = capsys.readouterr().out
    assert "fixture-key-never-printed" not in printed and "fixture-key-never-printed" not in out.read_text()
    assert json.loads(printed)["bound"] == report["bound"]


def endpoint(monkeypatch, native=8):
    """An OpenAI-compatible embedding endpoint behind the sidecar's own HTTP client: deterministic vectors of the
    endpoint's native width, or of the width a request asks for (``dimensions``). Every request body is kept."""
    import hashlib as _hashlib

    import httpx

    requests = []

    def vector(text, width):
        digest = _hashlib.sha256(text.encode()).digest()
        return [1.0] + [float(digest[index % len(digest)] % 7 + 1) for index in range(width - 1)]

    def response(request):
        body = json.loads(request.content)
        requests.append(body)
        width = body.get("dimensions", native)
        return httpx.Response(200, json={"model": "served-embed", "data": [
            {"index": index, "embedding": vector(text, width)} for index, text in enumerate(body["input"])]})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(response), **kw))
    for name in ("COLONYMIND_EMBED_PROVIDER", "COLONYMIND_EMBED_BASE_URL", "COLONYMIND_EMBED_MODEL",
                 "COLONYMIND_EMBED_DIMS", "COLONYMIND_EMBED_REQUEST_DIMS", "COLONYMIND_EMBED_REVISION",
                 "COLONYMIND_EMBED_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return requests


async def production_identity(model, *, dims, request_dims):
    """The embedding identity the sidecar gives an ``openai_api`` pipeline configured this way (``P/server.py``:
    the declared width is the configured one, else the requested one, else learned from the first answer)."""
    from colonymind.vector.config import EmbeddingConfig
    from colonymind.vector.embedder import EmbeddingPipeline, make_provider
    provider = make_provider(EmbeddingConfig(provider="openai_api", model_id=model,
                                             dimensions=dims or request_dims or 0, request_dimensions=request_dims))
    provider.configure("http://fixture/v1", "")
    pipeline = EmbeddingPipeline(provider)
    await pipeline.warmup()
    return pipeline.index_identity


@pytest.mark.parametrize("how", ["environment", "flag"])
def test_the_command_line_measures_the_width_the_sidecar_requests_and_records_it(tmp_path, monkeypatch, how):
    """A deployment that asks its endpoint for fewer dimensions than it natively answers is measured on those
    vectors, not on the native ones and not refused as a width mismatch; the report names the embedding it
    measured by the same identity the sidecar's vector index keeps."""
    requests = endpoint(monkeypatch)
    args = ["--base-url", "http://fixture/v1", "--model", "embed-model", "--out", str(tmp_path / "report.json")]
    if how == "environment":
        monkeypatch.setenv("COLONYMIND_EMBED_REQUEST_DIMS", "4")               # as the sidecar reads it
    else:
        args += ["--request-dims", "4"]
    assert bound.main(args) == 0
    assert requests and all(body.get("dimensions") == 4 for body in requests)
    report = json.loads((tmp_path / "report.json").read_text())
    embedding = report["embedding"]
    assert embedding["request_dimensions"] == 4 and embedding["dimensions"] == 4
    assert embedding["requested_model"] == "embed-model" and embedding["served_model"] == "served-embed"
    requests.clear()
    identity = asyncio.run(production_identity("embed-model", dims=0, request_dims=4))
    assert embedding["fingerprint"] == identity.fingerprint


def test_without_a_requested_width_the_native_one_is_measured_and_recorded(tmp_path, monkeypatch):
    requests = endpoint(monkeypatch)
    monkeypatch.setenv("COLONYMIND_EMBED_BASE_URL", "http://fixture/v1")          # the sidecar's own settings
    monkeypatch.setenv("COLONYMIND_EMBED_MODEL", "embed-model")
    monkeypatch.setenv("COLONYMIND_EMBED_DIMS", "8")
    assert bound.main(["--out", str(tmp_path / "report.json")]) == 0
    assert requests and all("dimensions" not in body for body in requests)
    embedding = json.loads((tmp_path / "report.json").read_text())["embedding"]
    assert embedding["request_dimensions"] is None and embedding["dimensions"] == 8
    identity = asyncio.run(production_identity("embed-model", dims=8, request_dims=None))
    assert embedding["fingerprint"] == identity.fingerprint


# ---------------------------------------------------------------------------
# Measured bounds: tied to the embedding they were measured on, reproducible from their evidence
# ---------------------------------------------------------------------------

def test_every_measured_bound_is_reproducible_from_its_evidence():
    """Each recorded measurement is on the pinned dev set, names the embedding it measured, keeps every pair's score
    under the set's own labels, and its bound is what the selection rule gives on those scores. The default for an
    embedding without a measurement is no looser than any bound measured."""
    data = json.loads(bound.BOUNDS.read_text())
    labels = {pair["id"]: pair["same_fact"] for pair in bound.load_dev_set()}
    assert data["measurements"]
    for entry in data["measurements"]:
        assert entry["dev_set"]["sha256"] == bound.DEV_SET_SHA256 and entry["precision_floor"] == bound.PRECISION_FLOOR
        embedding = entry["embedding"]
        assert embedding["requested_model"] and embedding["dimensions"] > 0 and embedding["fingerprint"]
        assert {row["id"]: row["same_fact"] for row in entry["pairs"]} == labels
        chosen = bound.select_bound(entry["pairs"], precision_floor=entry["precision_floor"])
        assert (chosen["bound"], chosen["floor_met"], chosen["at_bound"]) == (
            entry["bound"], entry["floor_met"], entry["at_bound"])
    assert KEY_SIMILARITY_BOUND >= max(entry["bound"] for entry in data["measurements"] if entry["floor_met"])


def identity(model="embed-model", dims=64, request=None):
    from colonymind.vector.indexes import EmbeddingIdentity
    return EmbeddingIdentity(requested_model=model, served_model="served-embed", declared_revision="unknown",
                             dimensions=dims, request_dimensions=request)


class Identified(TableEmbedder):
    """A stub embedder that carries an embedding identity, as the sidecar's pipeline does."""

    def __init__(self, ident, topics=None):
        super().__init__(topics)
        self.index_identity = ident


def measurement(ident, *, bound_value, floor_met=True, sha256=None):
    return {"dev_set": {"sha256": sha256 or bound.DEV_SET_SHA256}, "precision_floor": bound.PRECISION_FLOOR,
            "embedding": {**__import__("dataclasses").asdict(ident), "fingerprint": ident.fingerprint},
            "bound": bound_value, "floor_met": floor_met, "at_bound": None, "pairs": []}


def test_the_bound_for_an_embedder_is_the_one_measured_for_its_embedding(tmp_path):
    path = tmp_path / "bounds.json"
    path.write_text(json.dumps({"measurements": [
        measurement(identity("embed-model"), bound_value=0.7),
        measurement(identity("weak-model"), bound_value=0.5, floor_met=False),
        measurement(identity("old-set-model"), bound_value=0.6, sha256="0" * 64)]}))
    assert bound.bound_for(Identified(identity("embed-model")), path=path) == (0.7, "measured")
    # the same model at another width, a measurement on another dev set, no identity, no file: not measured
    for embedder in (Identified(identity("embed-model", dims=32, request=32)), Identified(identity("old-set-model")),
                     TableEmbedder()):
        assert bound.bound_for(embedder, path=path) == (KEY_SIMILARITY_BOUND, "unmeasured")
    assert bound.bound_for(Identified(identity("embed-model")), path=tmp_path / "missing.json") == (
        KEY_SIMILARITY_BOUND, "unmeasured")
    # a measurement that kept no bound above the precision floor: no pairs by meaning for that embedding
    assert bound.bound_for(Identified(identity("weak-model")), path=path) == (None, "below_floor")


def test_a_deployment_keeps_its_own_measurement_and_the_night_reads_it_before_the_shipped_one(tmp_path, monkeypatch):
    """A deployment that measured its own embedder, at a precision floor it chose, records the measurement in a file
    of its own (``--record``) and names that file in COLONYMIND_CONTRADICTION_KEY_BOUNDS: for that embedding its
    measurement replaces the shipped one (or the default), and every other embedding keeps the shipped bounds. Unset,
    or naming a file that is not there, the shipped file alone decides."""
    shipped, own = tmp_path / "shipped.json", tmp_path / "own.json"
    shipped.write_text(json.dumps({"measurements": [measurement(identity("embed-model"), bound_value=0.9),
                                                     measurement(identity("other-model"), bound_value=0.8)]}))
    own.write_text(json.dumps({"measurements": [
        {**measurement(identity("embed-model"), bound_value=0.78), "precision_floor": 0.6},
        measurement(identity("new-model"), bound_value=0.75)]}))
    monkeypatch.setattr(bound, "BOUNDS", shipped)
    monkeypatch.delenv("COLONYMIND_CONTRADICTION_KEY_BOUNDS", raising=False)
    assert bound.bound_for(Identified(identity("embed-model"))) == (0.9, "measured")
    assert bound.bound_for(Identified(identity("new-model"))) == (KEY_SIMILARITY_BOUND, "unmeasured")
    monkeypatch.setenv("COLONYMIND_CONTRADICTION_KEY_BOUNDS", str(own))
    assert bound.bound_for(Identified(identity("embed-model"))) == (0.78, "measured")
    assert bound.bound_for(Identified(identity("new-model"))) == (0.75, "measured")
    assert bound.bound_for(Identified(identity("other-model"))) == (0.8, "measured")
    assert bound.bound_for(TableEmbedder()) == (KEY_SIMILARITY_BOUND, "unmeasured")
    monkeypatch.setenv("COLONYMIND_CONTRADICTION_KEY_BOUNDS", str(tmp_path / "missing.json"))
    assert bound.bound_for(Identified(identity("embed-model"))) == (0.9, "measured")


def test_the_command_line_records_its_measurement_for_the_night(tmp_path, monkeypatch):
    """--record adds the report to a bounds file as the measurement for its embedding on its dev set (a second run
    replaces it), and the night then reads that bound for a pipeline of the same embedding."""
    endpoint(monkeypatch)
    record = tmp_path / "bounds.json"
    args = ["--base-url", "http://fixture/v1", "--model", "embed-model", "--dims", "8", "--record", str(record)]
    assert bound.main(args) == 0 and bound.main(args) == 0
    entry, = json.loads(record.read_text())["measurements"]
    assert entry["dev_set"]["sha256"] == bound.DEV_SET_SHA256 and len(entry["pairs"]) == 60
    assert entry["embedding"]["requested_model"] == "embed-model" and entry["embedding"]["dimensions"] == 8
    pipeline = asyncio.run(production_identity("embed-model", dims=8, request_dims=None))
    status = "measured" if entry["floor_met"] else "below_floor"
    assert bound.bound_for(Identified(pipeline), path=record) == (entry["bound"] if entry["floor_met"] else None,
                                                                   status)


def test_the_command_line_embeds_with_a_local_provider_as_the_sidecar_does(tmp_path, monkeypatch):
    """A sidecar that embeds in-process (a local provider, no endpoint) is measured the same way: the provider and
    the declared width follow the sidecar's settings."""
    made = {}

    async def fake(provider, base_url, model, dims, request_dims, revision):
        made.update(provider=provider, base_url=base_url, model=model, dims=dims, request_dims=request_dims)
        return TableEmbedder()

    for name in ("COLONYMIND_EMBED_BASE_URL", "COLONYMIND_EMBED_DIMS", "COLONYMIND_EMBED_REQUEST_DIMS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLONYMIND_EMBED_PROVIDER", "cpu")
    monkeypatch.setenv("COLONYMIND_EMBED_MODEL", "local-embed")
    monkeypatch.setattr(bound, "_sidecar_embedder", fake)
    assert bound.main(["--out", str(tmp_path / "report.json")]) == 0
    assert made == {"provider": "cpu", "base_url": None, "model": "local-embed", "dims": 384, "request_dims": None}
    with pytest.raises(SystemExit):
        bound.main(["--request-dims", "4"])                                 # a local model is not asked a width
