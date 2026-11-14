"""Two uses of the embedding model beyond recall are faculties with a switch of their own and a line in
``colonymind report faculties``.

- ``mind.faculties.contradictions_by_meaning``: each night a person's statements filed under two different names
  whose embeddings are close are compared too (the night's contradiction candidates by meaning). Off, the night
  compares statements under the very same name only.
- ``mind.faculties.priors_by_meaning``: when a message's statements are recorded, the extractor is also offered
  earlier records found by meaning (the vector lane of the prior records). Off, it is offered the records a keyword
  search finds only. The claim projection runs outside the mind, so the switch reaches it as an environment flag
  (``COLONYMIND_PRIORS_BY_MEANING``), the way the learning rules do.

Both ship on (on unless shown harmful) and are read from ``mind.faculties`` when the file names them.
"""
from datetime import datetime, timedelta, timezone

import pytest

import colonymind.vector as vector
from colonymind.beliefs import source_projection
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.config import ConfigError, apply_environment, load_config
from colonymind.mind.consolidate import TASK_DIGEST
from colonymind.reports import faculties as report
from colonymind.reports.registry import BY_KEY, FACULTIES
from report_fixtures import write_instance
from test_mind_consolidate import OWNER, Fixture
from test_mind_contradiction_candidates import EMPLOYER, WORKS_AT, KeyEmbedder, relation_calls
from test_source_claim_priors import FILLERS, MOVED, OFFICE, ChoosingModel, Pipeline, claim_rows, outputs, person, store

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


# -- contradictions by meaning -----------------------------------------------------------------------------------

@pytest.fixture
def night(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setattr(vector, "_pipeline", KeyEmbedder({WORKS_AT: "job", EMPLOYER: "job"}))
    made = []

    def build(**faculties):
        fixture = Fixture(tmp_path / f"night-{len(made)}", config={"faculties": faculties})
        fixture.router.answers[TASK_DIGEST] = None
        made.append(fixture)
        return fixture

    yield build
    for fixture in made:
        fixture.store.close()


async def two_names(fx):
    await fx.fact("t-1", OWNER, "s-1", "I work at Acme.", "Acme", predicate="works_at")
    fx.shift(days=1)
    await fx.fact("t-2", OWNER, "s-2", "My employer is Globex.", "Globex", predicate="employer")
    return await fx.mind.consolidate()


@pytest.mark.asyncio
async def test_the_night_compares_names_by_meaning_unless_the_switch_is_off(night):
    shipped = night()
    result = await two_names(shipped)
    assert len(relation_calls(shipped)) == 1 and result["counts"].get("keys_embedded") == 2
    off = night(contradictions_by_meaning=False)
    result = await two_names(off)
    assert relation_calls(off) == [] and "keys_embedded" not in result["counts"]


@pytest.mark.asyncio
async def test_switched_off_the_night_still_decides_statements_under_one_name(night):
    fx = night(contradictions_by_meaning=False)
    await fx.fact("t-1", OWNER, "s-1", "My office is in room 4.", "room 4")
    fx.shift(days=1)
    await fx.fact("t-2", OWNER, "s-2", "My office is in room 7.", "room 7")
    await fx.mind.consolidate()
    assert len(relation_calls(fx)) == 1


# -- prior records by meaning -------------------------------------------------------------------------------------

class Searches:
    """The source vector search, counted."""

    def __init__(self, vectors):
        self.vectors, self.calls = vectors, 0

    async def search(self, *args, **kwargs):
        self.calls += 1
        return await self.vectors.search(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("flag, offered", [(None, True), ("1", True), ("0", False), ("off", False)])
async def test_the_extractor_is_offered_earlier_records_by_meaning_unless_the_switch_is_off(
        tmp_path, monkeypatch, flag, offered):
    if flag is None:
        monkeypatch.delenv(source_projection.PRIORS_BY_MEANING_FLAG, raising=False)
    else:
        monkeypatch.setenv(source_projection.PRIORS_BY_MEANING_FLAG, flag)
    ledger, vectors = await store(tmp_path, Pipeline())
    searches = Searches(vectors)
    projection = SourceClaimProjection(ledger, vectors=searches)
    model = ChoosingModel(outputs())
    await person(ledger, projection, vectors, model, [OFFICE, *FILLERS])
    berlin, = [row for row in claim_rows(ledger).values() if row["value"] == "Berlin"]
    model.calls.clear()
    searches.calls = 0
    await person(ledger, projection, vectors, model, [MOVED], start=100)
    assert (berlin["id"] in [row["id"] for row in model.calls[0][0]["prior_assertions"]]) is offered
    assert (searches.calls > 0) is offered            # switched off, the lane never searches


def test_the_install_exports_the_priors_switch_and_reserves_its_name(tmp_path):
    (tmp_path / "colonymind.yaml").write_text("owner: {contact_id: owner-1}\n")
    assert apply_environment(load_config(tmp_path), environ={})[source_projection.PRIORS_BY_MEANING_FLAG] == "1"
    (tmp_path / "colonymind.yaml").write_text("owner: {contact_id: owner-1}\n"
                                             "mind: {faculties: {priors_by_meaning: false}}\n")
    assert apply_environment(load_config(tmp_path), environ={})[source_projection.PRIORS_BY_MEANING_FLAG] == "0"
    (tmp_path / "colonymind.yaml").write_text("owner: {contact_id: owner-1}\n"
                                             "environment: {COLONYMIND_PRIORS_BY_MEANING: '0'}\n")
    with pytest.raises(ConfigError, match="mind.faculties.priors_by_meaning"):
        load_config(tmp_path)


# -- the faculty report -------------------------------------------------------------------------------------------

def section(document, title):
    lines, inside = [], False
    for block in document.blocks:
        if block.kind == "heading" and block.level == 2:
            inside = block.text.split(":")[0] == title
            continue
        if inside:
            lines.extend(block.items if block.kind == "list" else [block.text])
    return "\n".join(lines)


def heading(document, title):
    return next(block.text for block in document.blocks if block.kind == "heading" and block.text.startswith(title))


def test_each_has_a_line_an_arm_and_a_switch():
    for key, title in (("contradictions_by_meaning", "Contradictions by meaning"),
                       ("priors_by_meaning", "Earlier statements by meaning")):
        faculty = BY_KEY[key]
        assert faculty.title == title and faculty in FACULTIES
        assert faculty.setting == f"mind.faculties.{key}" and faculty.arm == f"all-on-{key}"


def test_the_report_says_they_ship_on_and_how_to_switch_each_off(tmp_path):
    home = tmp_path / "fresh"
    home.mkdir()
    (home / "colonymind.yaml").write_text("owner:\n  contact_id: owner-1\n")
    document = report.document(report.collect(home, now=NOW))
    for title, key in (("Contradictions by meaning", "contradictions_by_meaning"),
                       ("Earlier statements by meaning", "priors_by_meaning")):
        assert heading(document, title) == f"{title}: on"
        text = section(document, title)
        assert f"To switch it off, set mind.faculties.{key} to false in colonymind.yaml." in text
        assert "no embedding endpoint is set" in text


def test_the_report_reads_what_the_nights_compared_by_meaning(tmp_path):
    from colonymind.initiatives.store import InitiativeStore
    home = write_instance(tmp_path / "instance", NOW)
    store = InitiativeStore(state_dir=home)
    for day, counts in ((1, {"keys_embedded": 6, "key_bound_unmeasured": 1}),
                        (2, {"keys_embedded": 4, "key_embedding_failed": 1, "key_bound_unmeasured": 1})):
        at = NOW - timedelta(days=day, hours=1)
        row, _ = store.create_intention(kind="note", type="consolidation", title="consolidation work", drive="upkeep",
                                        cls="internal", decision="act", decision_reason="fixture", status="done",
                                        dedup_key=None, context={"counts": counts}, created_at=at)
        store.transition(row.id, "done", action="fixture", at=at, completed_at=at + timedelta(minutes=5))
    store.close()
    document = report.document(report.collect(home, now=NOW))
    text = section(document, "Contradictions by meaning")
    assert "It compared statements filed under different names by meaning on 2 nights (10 names)." in text
    assert "On 1 night the embedding model failed, so that night compared statements under one name only." in text
    assert "never measured" in text


def test_the_report_names_what_each_needs(tmp_path):
    home = write_instance(tmp_path / "instance", NOW)
    config = (home / "colonymind.yaml").read_text()
    (home / "colonymind.yaml").write_text(config.replace("  faculties:\n", "  faculties:\n    semantic_recall: false\n"
                                                                            "    consolidation: false\n"))
    document = report.document(report.collect(home, now=NOW))
    assert heading(document, "Contradictions by meaning") == "Contradictions by meaning: off"
    assert "nightly consolidation" in section(document, "Contradictions by meaning")
    assert heading(document, "Earlier statements by meaning") == "Earlier statements by meaning: off"
    assert "semantic recall is off" in section(document, "Earlier statements by meaning")
    (home / "colonymind.yaml").write_text(config.replace(
        "  faculties:\n", "  faculties:\n    contradictions_by_meaning: false\n    priors_by_meaning: false\n"))
    document = report.document(report.collect(home, now=NOW))
    for title, key in (("Contradictions by meaning", "contradictions_by_meaning"),
                       ("Earlier statements by meaning", "priors_by_meaning")):
        assert heading(document, title) == f"{title}: off"
        assert f"To switch it on, set mind.faculties.{key} to true in colonymind.yaml." in section(document, title)
