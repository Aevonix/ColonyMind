"""KU20: is a paraphrased knowledge update linked to the record it changes?

The pre-registered set (``ku20.json``, frozen by its SHA-256 below) and its runner. One person's memory is
formed for real: 60 filler messages and the first message of every pair, extracted and reviewed by the model
under test, with source vectors projected as the sidecar projects them. Then each second message is run through
the claim dry run (``SourceClaimProjection.rehearse``, the route ``POST /v1/host/claims/dry-run``), so every pair
is measured against the same formed store and nothing a second message would write reaches the next.

- 20 change pairs: the second states a change of the first's property without sharing a content word with it,
  while sharing one with a filler, so a keyword search of the second finds the person's other claims and never
  the first. Each pair's **target** is the one claim of its first message that holds the value the pair is about
  (``target_claims``), found after formation; a first message can state more than one property. Measured: the
  share whose target is **offered** to the extractor, and the share it would **supersede** (retract or supersede
  the target), plus changes that would retire any other claim, the first message's other claims included. A pair
  whose target is missing or ambiguous is not scored, and is listed.
- 10 controls: the second is a change of a different, nearby property. Measured: the share that would retire
  any claim (over-linking).
- The review rejection rate over all 30 dry runs, and the tokens they cost.
- The store must hold at least 50 other claims of the person for every pair (checked on the formed store and
  reported; a run below it is invalid).

A build without the dry run (extraction version 17 had neither it nor the priors' vector lane) is measured with
``--path commit``: each second message is committed for real into its own copy of the formed store, and the
priors offered are read from the extraction call. ``--path commit`` on a later build measures it without its
vector lane. The gold fields of the set are used only here, after the calls; they never reach extraction.

Endpoints (never a model or embedder a live instance uses): ``COLONYMIND_BENCH_CHAT_BASE_URL``,
``COLONYMIND_BENCH_CHAT_MODEL`` (``COLONYMIND_BENCH_CHAT_API_KEY`` optional) for extraction and review; for the
dry-run path also ``COLONYMIND_BENCH_EMBED_BASE_URL``, ``COLONYMIND_BENCH_EMBED_MODEL``,
``COLONYMIND_BENCH_EMBED_DIMS`` (``COLONYMIND_BENCH_EMBED_API_KEY`` and ``COLONYMIND_BENCH_EMBED_QUERY_INSTRUCTION``
optional).

    python benchmarks/knowledge_update/run.py --state-dir <new empty directory> --out <results.json>

Run it with the interpreter of the build under test: it imports that interpreter's ``colonymind`` and falls back
to this checkout's sidecar only when none is installed.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
SET = HERE / "ku20.json"
SET_SHA256 = "e8d14a4a326ebda21b766baa7799a4ac4b15e38053c115d49f47fa055c62bab4"
#: Other claims of the person the formed store must hold for each pair (the pre-registered floor).
MIN_OTHER_CLAIMS = 50
SESSION = "ku20-later"
PATHS = ("auto", "dry_run", "commit")


def load_set(path: Path = SET, *, frozen: bool = True) -> dict:
    """The set; refused when its bytes are not the frozen ones (a changed set is a new pre-registration)."""
    raw = Path(path).read_bytes()
    if frozen and hashlib.sha256(raw).hexdigest() != SET_SHA256:
        raise ValueError("ku20.json is not the frozen set: a changed set needs a new pre-registration")
    return json.loads(raw)


def items(ku: dict):
    """Every measured item, change pairs first, with its kind."""
    return [(item, "change") for item in ku["pairs"]] + [(item, "control") for item in ku["controls"]]


def first_turn(item) -> str:
    return "ku20:" + item["id"] + "-first"


def formation_records(ku: dict):
    """What is committed for real before any measurement, in the order it was said: fillers and first messages."""
    rows = [("ku20:" + filler["id"], filler["at"], filler["text"]) for filler in ku["fillers"]]
    rows += [(first_turn(item), item["first"]["at"], item["first"]["text"]) for item, _ in items(ku)]
    return sorted(rows, key=lambda row: row[1])


async def form_store(ledger, projection, router, ku: dict, *, vectors=None) -> None:
    """Record and extract every formation message in turn; project each into the vector index when there is one."""
    for turn, at, text in formation_records(ku):
        ledger.record_source(turn, contact_id=ku["contact_id"], session_id=turn, occurred_at=at,
                             timezone_name=ku["timezone"], messages=[{"role": "user", "content": text}])
        while await projection.process_one(router):
            pass
        if vectors is not None:
            while await vectors.process_one():
                pass


def _within(part: list, whole: list) -> bool:
    """Whether the words ``part`` occur, in order and together, in ``whole``."""
    return bool(part) and any(whole[i:i + len(part)] == part for i in range(len(whole) - len(part) + 1))


def target_claims(ledger, ku: dict) -> dict:
    """Each item's target after formation: the live claim of its first message that holds the value the item is
    about (``gold.first_value``). ``{"status": "found", "ids": [id]}``, or ``missing`` (no claim of the first
    message holds it) or ``ambiguous`` (more than one does), with the candidates.

    A claim holds the value when their normalized words are equal, or else when one contains the other as whole
    words in order ("the Riverside gym" holds "Riverside gym"; "Tuesdays" does not). An exact match wins over a
    containing one. The rule is a builder's choice, fixed before any run: it only has to tell the property an
    item is about from the other properties its first message states."""
    from colonymind.beliefs.source_claims import norm_value
    result = {}
    with closing(ledger._connect()) as conn:
        for item, _ in items(ku):
            rows = [(row[0], norm_value(json.loads(row[1]).get("value")).split()) for row in conn.execute(
                "SELECT id,data_json FROM source_claims WHERE turn_id=? AND superseded_by IS NULL "
                "AND retracted_by IS NULL ORDER BY id", (first_turn(item),))]
            gold = norm_value(item["gold"]["first_value"]).split()
            found = ([ident for ident, words in rows if words == gold]
                     or [ident for ident, words in rows if _within(gold, words) or _within(words, gold)])
            result[item["id"]] = {"status": "found" if len(found) == 1 else "ambiguous" if found else "missing",
                                  "ids": found}
    return result


def formed_store(ledger, ku: dict):
    """Each item's live first claims, and the person's live claims in all."""
    with closing(ledger._connect()) as conn:
        firsts = {item["id"]: sorted(row[0] for row in conn.execute(
            "SELECT id FROM source_claims WHERE turn_id=? AND superseded_by IS NULL AND retracted_by IS NULL",
            (first_turn(item),))) for item, _ in items(ku)}
        total = conn.execute("""SELECT count(*) FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id
            WHERE s.contact_id=? AND c.superseded_by IS NULL AND c.retracted_by IS NULL""",
                             (ku["contact_id"],)).fetchone()[0]
    return firsts, total


def verdict(item, kind, target, *, offered, accepted, rejected, status, diagnostics, usage) -> dict:
    """One item's result from what the extractor was offered and what the commit would do. ``target``: the
    item's entry of ``target_claims``; any retired claim that is not the found target is ``retired_other``."""
    goal = set(target["ids"]) if target["status"] == "found" else set()
    retired = {ident for row in accepted for ident in [*row.get("retracts", []), *row.get("supersedes", [])]}
    proposed = {row.get("claim", {}).get("prior_claim_id") for row in accepted}
    proposed |= {row["proposal"].get("prior_claim_id") for row in rejected if isinstance(row.get("proposal"), dict)}
    return {"id": item["id"], "kind": kind, "status": status, "target": target["status"],
            "offered": bool(goal & set(offered)), "proposed_link": bool(goal & proposed),
            "superseded": bool(goal & retired), "retired_other": sorted(retired - goal),
            "held_for_owner": sorted({ident for row in accepted for ident in row.get("held_for_owner", [])}),
            "reviewed": int(diagnostics.get("reviewed_count") or 0),
            "review_rejected": int(diagnostics.get("review_rejected_count") or 0),
            "rejections": sorted({(row.get("stage"), str(row.get("reason"))[:120]) for row in rejected}),
            "usage": usage}


async def measure_dry_run(projection, router, ku: dict, targets: dict) -> list[dict]:
    """Each second message through the dry run, against the same formed store."""
    results = []
    for item, kind in items(ku):
        response = await projection.rehearse(router, contact_id=ku["contact_id"], session_id=SESSION,
                                             text=item["second"]["text"], occurred_at=item["second"]["at"],
                                             timezone_name=ku["timezone"])
        results.append(verdict(item, kind, targets[item["id"]],
                               offered=[row["id"] for row in response.get("offered_priors", [])],
                               accepted=response.get("accepted", []), rejected=response.get("rejected", []),
                               status=response["status"], diagnostics=response.get("diagnostics") or {},
                               usage=response.get("usage") or {}))
    return results


class _Watched:
    """The router, keeping what each extraction call was offered and the tokens every call reported."""

    def __init__(self, router):
        self._router, self.offered, self.usage = router, [], {"calls": 0}

    def __getattr__(self, name):
        return getattr(self._router, name)

    async def complete(self, messages, **kwargs):
        if (kwargs.get("context") or {}).get("task") == "source_claim_extraction":
            payload = json.loads(messages[-1]["content"])
            self.offered.extend(row["id"] for row in payload.get("prior_assertions", []))
        response = await self._router.complete(messages, **kwargs)
        self.usage["calls"] += 1
        for key, value in (getattr(response, "usage", None) or {}).items():
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value
        return response


async def measure_by_commit(ledger, router, ku: dict, targets: dict, scratch: Path) -> list[dict]:
    """Each second message committed for real into its own copy of the formed store (a build with no dry run)."""
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import TurnIdempotencyLedger
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for item, kind in items(ku):
        copy = scratch / (item["id"] + ".db")
        with closing(sqlite3.connect(ledger.db_path)) as source, closing(sqlite3.connect(copy)) as target:
            source.backup(target)
        pair = TurnIdempotencyLedger(copy)
        watched = _Watched(router)
        turn = "ku20:" + item["id"] + "-second"
        pair.record_source(turn, contact_id=ku["contact_id"], session_id=SESSION, occurred_at=item["second"]["at"],
                           timezone_name=ku["timezone"], messages=[{"role": "user", "content": item["second"]["text"]}])
        await SourceClaimProjection(pair).process_one(watched)
        with closing(pair._connect()) as conn:
            made = [dict(row) for row in conn.execute("SELECT id,data_json FROM source_claims WHERE turn_id=?", (turn,))]
            # What each new claim retired, and the changes it left for the owner to decide (as the dry run reports).
            accepted = [{"claim": json.loads(row["data_json"]), **{key: [r[0] for r in conn.execute(sql, (row["id"],))]
                         for key, sql in (
                             ("retracts", "SELECT id FROM source_claims WHERE retracted_by=? ORDER BY id"),
                             ("supersedes", "SELECT id FROM source_claims WHERE superseded_by=? ORDER BY id"),
                             ("held_for_owner", "SELECT target_id FROM learning_pending WHERE by_id=? "
                                                "AND status='pending' ORDER BY target_id"))}}
                        for row in made]
            job = conn.execute("SELECT status,diagnostics_json FROM source_claim_jobs WHERE turn_id=?", (turn,)).fetchone()
        diagnostics = json.loads(job["diagnostics_json"]) if job and job["diagnostics_json"] else {}
        results.append(verdict(item, kind, targets[item["id"]], offered=watched.offered, accepted=accepted,
                               rejected=[], status="complete" if job and job["status"] == "complete" else "failed",
                               diagnostics=diagnostics, usage=watched.usage))
    return results


def score(results: list[dict], *, other_claims_min: int, build: dict) -> dict:
    """The pre-registered summary. Shares are of the scored change pairs (offered, superseded: those whose
    target was found) and of the controls (over-linked: would retire any claim)."""
    change = [row for row in results if row["kind"] == "change" and row["target"] == "found"]
    controls = [row for row in results if row["kind"] == "control"]

    def share(count, total):
        return round(count / total, 4) if total else None
    offered = sum(row["offered"] for row in change)
    superseded = sum(row["superseded"] for row in change)
    over = sum(row["superseded"] or bool(row["retired_other"]) for row in controls)
    reviewed = sum(row["reviewed"] for row in results)
    refused = sum(row["review_rejected"] for row in results)
    tokens = {key: sum(row["usage"].get(key, 0) for row in results)
              for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")}
    return {"set": "KU20", "set_sha256": SET_SHA256, "build": build,
            "store": {"other_claims_min": other_claims_min, "floor": MIN_OTHER_CLAIMS,
                      "valid": other_claims_min >= MIN_OTHER_CLAIMS},
            "change_pairs": len(change),
            "not_scored": {row["id"]: "target_" + row["target"] for row in results
                           if row["kind"] == "change" and row["target"] != "found"},
            "offered": offered, "offered_share": share(offered, len(change)),
            "superseded": superseded, "superseded_share": share(superseded, len(change)),
            "wrong_target": sum(bool(row["retired_other"]) for row in change),
            "controls": len(controls), "over_linked": over, "over_link_share": share(over, len(controls)),
            "reviewed": reviewed, "review_rejected": refused, "review_rejection_rate": share(refused, reviewed),
            "held_for_owner": sum(bool(row["held_for_owner"]) for row in results),
            "not_complete": [row["id"] for row in results if row["status"] != "complete"], "usage": tokens}


async def measure(ledger, projection, router, ku: dict, *, path: str, vectors=None, scratch: Path | None = None,
                  build: dict | None = None) -> dict:
    """Form the store, measure every item on ``path`` and score it: ``{summary, results}``."""
    await form_store(ledger, projection, router, ku, vectors=vectors)
    firsts, total = formed_store(ledger, ku)
    other_min = min(total - len(ids) for ids in firsts.values())
    targets = target_claims(ledger, ku)
    if path == "dry_run":
        results = await measure_dry_run(projection, router, ku, targets)
    else:
        results = await measure_by_commit(ledger, router, ku, targets, scratch)
    return {"summary": score(results, other_claims_min=other_min, build=dict(build or {}, path=path,
                                                                                 vector_lane=vectors is not None)),
            "results": results}


def environment(path: str) -> dict:
    required = ["CHAT_BASE_URL", "CHAT_MODEL"] + (["EMBED_BASE_URL", "EMBED_MODEL", "EMBED_DIMS"]
                                                   if path == "dry_run" else [])
    missing = ["COLONYMIND_BENCH_" + name for name in required if not os.environ.get("COLONYMIND_BENCH_" + name)]
    if missing:
        raise ValueError("Missing benchmark variables: " + ", ".join(missing))
    names = [*required, "CHAT_API_KEY", "EMBED_API_KEY", "EMBED_QUERY_INSTRUCTION"]
    return {name: os.environ["COLONYMIND_BENCH_" + name] for name in names if "COLONYMIND_BENCH_" + name in os.environ}


def router_config(config: dict) -> dict:
    """Extraction and review both on the model under test (timeouts: a builder's choice for a benchmark)."""
    return {"provider": "custom", "protocol": "openai-chat",
            "modelPool": {"candidate": {"model": config["CHAT_MODEL"], "baseUrl": config["CHAT_BASE_URL"],
                                        "apiKey": config.get("CHAT_API_KEY", ""), "maxTokens": 4096,
                                        "supportsTools": True, "supportsJsonSchema": True}},
            "functionRoles": {role: {"candidates": ["candidate"], "timeoutSeconds": 120, "deadlineSeconds": 120}
                              for role in ("extraction", "judging")}}


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state-dir", type=Path, required=True, help="a new empty directory for this run's stores")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--path", choices=PATHS, default="auto")
    args = parser.parse_args(argv)
    try:
        import colonymind  # noqa: F401  (the build under test: the interpreter's own)
    except ImportError:
        sys.path.insert(0, str(HERE.parents[1] / "sidecar"))
    from colonymind.beliefs.source_claims import EXTRACTION_VERSION
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.router.router import LLMRouter
    from colonymind.turns import TurnIdempotencyLedger
    path = args.path if args.path != "auto" else ("dry_run" if hasattr(SourceClaimProjection, "rehearse") else "commit")
    state = args.state_dir.resolve()
    if state.exists() and any(state.iterdir()):
        raise ValueError("Use a new empty disposable state directory")
    state.mkdir(parents=True, exist_ok=True)
    config = environment(path)
    ku = load_set()
    os.environ["COLONYMIND_STATE_DIR"] = str(state)
    ledger = TurnIdempotencyLedger(state / "turn-idempotency.db")
    router = LLMRouter(tiers={})
    router.configure(router_config(config))
    vectors = None
    build = {"extraction_version": EXTRACTION_VERSION, "model": config["CHAT_MODEL"]}
    if path == "dry_run":
        from colonymind.turns.source_vectors import SourceVectors
        from colonymind.vector.config import EmbeddingConfig
        from colonymind.vector.embedder import EmbeddingPipeline
        from colonymind.vector.indexes import IndexCatalog
        from colonymind.vector.openai_provider import OpenAIAPIEmbeddingProvider
        from colonymind.vector.store import VectorStore
        if "EMBED_QUERY_INSTRUCTION" in config:
            os.environ["COLONYMIND_EMBED_QUERY_INSTRUCTION"] = config["EMBED_QUERY_INSTRUCTION"]
        provider = OpenAIAPIEmbeddingProvider(EmbeddingConfig(provider="openai_api", model_id=config["EMBED_MODEL"],
                                                              dimensions=int(config["EMBED_DIMS"])))
        provider.configure(config["EMBED_BASE_URL"], config.get("EMBED_API_KEY", ""))
        pipeline = EmbeddingPipeline(provider)
        await pipeline.warmup()
        store = VectorStore(str(state / "lancedb"), identity=pipeline.index_identity, catalog=IndexCatalog(ledger))
        await store.connect(pipeline.dimensions)
        await store.ensure_collections(pipeline.dimensions)
        vectors = SourceVectors(ledger, store, pipeline)
        build["embedding_model"] = config["EMBED_MODEL"]
        projection = SourceClaimProjection(ledger, vectors=vectors)
    else:
        projection = SourceClaimProjection(ledger)
    report = await measure(ledger, projection, router, ku, path=path, vectors=vectors, scratch=state / "pairs",
                           build=build)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=1))
    return 0 if report["summary"]["store"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
