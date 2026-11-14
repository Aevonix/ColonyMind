"""The bound for contradiction candidates by meaning: a labelled dev set and the rule that chooses the bound.

The night's contradictions stage (``P/mind/consolidate.py``, docs/CONSOLIDATION.md) puts two live claims of one
person filed under different keys to the relation decision when their keys' embeddings are at least
``KEY_SIMILARITY_BOUND`` alike (the cosine of the two ``key_text``). This module chooses that bound, offline:

- the dev set (``contradiction_keys_dev.json``, pinned by its SHA-256): 60 synthetic pairs of claims under two
  different keys, 30 labelled the same fact (one value at a time: the newer statement replaces the older and says
  so) and 30 two facts that can both hold, among them near misses (a home city and a birth city, one's own employer
  and a sister's);
- ``score_dev_set``: each pair's cosine, from one batch of exactly the key texts the night embeds;
- ``select_bound``: every score is tried as a threshold; of those whose candidates are at least ``PRECISION_FLOOR``
  same-fact pairs, the one that finds the most (a tie keeps the higher threshold), placed halfway to the next lower
  score so the chosen side of every dev pair keeps a margin. When no threshold keeps the floor, the one with the
  best F1, reported as below the floor. With no same-fact pair found at any threshold, no bound.

A bound belongs to the embedding it was measured on: cosines of one model say nothing about another's. The measured
bounds (``contradiction_keys_bounds.json``) are each a trimmed report: the dev set's hash, the embedding measured
(its identity and fingerprint, as the sidecar's vector index keeps them) and every pair's score, so the choice can
be recomputed. The night (``bound_for``) uses the bound measured for its own embedding (the same model at the same
width, on the pinned dev set); an embedding whose measurement kept no bound above the floor gets no pairs by
meaning; one never measured gets ``KEY_SIMILARITY_BOUND``, a default no looser than any bound measured.

A deployment may keep measurements of its own (``DEPLOYMENT_BOUNDS``: a bounds file it recorded, named in
``COLONYMIND_CONTRADICTION_KEY_BOUNDS``), for an embedder the release did not measure or at a precision floor it
chose; each entry records its floor. For the embeddings it measured, its file replaces the shipped measurements; a
setting that names no readable file leaves the shipped ones alone, and says so once.

Run it on the embedder the sidecar uses, built as the sidecar builds it (``P/server.py``): every option defaults to
the sidecar's own setting (``COLONYMIND_EMBED_PROVIDER``, ``_BASE_URL``, ``_MODEL``, ``_DIMS``, ``_REQUEST_DIMS``,
``_REVISION``, and an endpoint's key, if it needs one, from ``COLONYMIND_EMBED_API_KEY``), so in the sidecar's
environment no option is needed and the vectors measured are the ones the night gets, at the width it asks for:

    python -m colonymind.mind.contradiction_keys [--provider P] [--base-url URL] [--model MODEL] [--dims N]
                                                [--request-dims N] [--out report.json] [--record bounds.json]

It prints the chosen bound with its precision and recall and the default bound's, writes the full report (every
pair's score, the whole curve, and ``embedding``) to ``--out``, and with ``--record`` adds the measurement to a
bounds file (replacing an earlier one of the same embedding on the same dev set). Nothing else is written.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from colonymind.beliefs.source_claims import norm_value
from colonymind.memory.learning import cosine

from .consolidate import KEY_SIMILARITY_BOUND, key_text

logger = logging.getLogger(__name__)

DEV_SET = Path(__file__).with_name("contradiction_keys_dev.json")
BOUNDS = Path(__file__).with_name("contradiction_keys_bounds.json")
#: The environment setting that names a deployment's own bounds file (``--record`` writes one).
DEPLOYMENT_BOUNDS = "COLONYMIND_CONTRADICTION_KEY_BOUNDS"
DEV_SET_SHA256 = "51bbbf9c0de8e7eebe62125fa568646c7e047f49f45ef7102b2656cd2ad30f92"
PRECISION_FLOOR = 0.8


def claim_key(subject: str, predicate: str) -> Tuple[str, str]:
    """The key the claim extractor files a claim under (``P/beliefs/source_claims.py``): ``speaker`` for "I", else
    the normalised subject, and the normalised words of the predicate."""
    subject = str(subject or "").strip()
    return ("speaker" if subject.lower() == "i" else norm_value(subject),
            norm_value(str(predicate or "").replace("_", " ")))


def load_dev_set(path: Path | str = DEV_SET, *, sha256: Optional[str] = DEV_SET_SHA256) -> List[Dict[str, Any]]:
    """The dev set's pairs. A file whose SHA-256 is not the pinned one is refused: the set is pre-registered, and a
    changed set is a new set, read only on purpose (``sha256=None``)."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"the dev set's sha256 is {digest}, not the pinned {sha256}")
    return list(json.loads(raw)["pairs"])


def _text(side: Mapping[str, Any]) -> str:
    return key_text(*claim_key(side["subject"], side["predicate"]))


async def score_dev_set(embedder: Any, pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Each pair's cosine of its two keys' embeddings (``id``, ``same_fact``, ``score``), from one ``embed_batch``
    of the distinct key texts, which are the texts the night embeds (``key_text``)."""
    texts = list(dict.fromkeys(_text(side) for pair in pairs for side in (pair["a"], pair["b"])))
    vectors = list(await embedder.embed_batch(texts))
    if len(vectors) != len(texts):
        raise ValueError("the embedder answered a different number of vectors than key texts")
    by_text = dict(zip(texts, vectors))
    return [{"id": str(pair["id"]), "same_fact": bool(pair["same_fact"]),
             "score": cosine(by_text[_text(pair["a"])], by_text[_text(pair["b"])])} for pair in pairs]


def _point(scored: Sequence[Mapping[str, Any]], threshold: float) -> Dict[str, Any]:
    """What a bound of ``threshold`` does on the scored pairs: a pair is a candidate at or above it. Precision is
    None when nothing is a candidate."""
    tp = sum(1 for row in scored if row["same_fact"] and row["score"] >= threshold)
    fp = sum(1 for row in scored if not row["same_fact"] and row["score"] >= threshold)
    fn = sum(1 for row in scored if row["same_fact"] and row["score"] < threshold)
    tn = sum(1 for row in scored if not row["same_fact"] and row["score"] < threshold)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"threshold": threshold, "candidates": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def select_bound(scored: Sequence[Mapping[str, Any]], *, precision_floor: float = PRECISION_FLOOR) -> Dict[str, Any]:
    """The bound the scored dev pairs support (the rule in the module docstring): ``bound`` (None when no threshold
    finds a same-fact pair), ``floor_met``, ``at_bound`` (the bound's own counts) and ``curve`` (every score as a
    threshold, highest first)."""
    scores = sorted({float(row["score"]) for row in scored}, reverse=True)
    curve = [_point(scored, threshold) for threshold in scores]
    kept = [point for point in curve if point["tp"] and point["precision"] >= precision_floor]
    if kept:
        best, floor_met = max(kept, key=lambda point: (point["recall"], point["threshold"])), True
    else:
        found = [point for point in curve if point["tp"]]
        best = max(found, key=lambda point: (point["f1"], point["threshold"])) if found else None
        floor_met = False
    if best is None:
        return {"bound": None, "floor_met": False, "precision_floor": precision_floor, "at_bound": None,
                "curve": curve}
    lower = next((score for score in scores if score < best["threshold"]), None)
    chosen = best["threshold"] if lower is None else (best["threshold"] + lower) / 2
    return {"bound": chosen, "floor_met": floor_met, "precision_floor": precision_floor,
            "at_bound": _point(scored, chosen), "curve": curve}


def embedding_identity(embedder: Any) -> Optional[Dict[str, Any]]:
    """What was measured, for the report: an embedding pipeline's identity (model asked for and served, revision,
    width, the width requested of the endpoint) and its fingerprint, the same the sidecar's vector index keeps.
    None for an embedder without one."""
    identity = getattr(embedder, "index_identity", None)
    if identity is None or not dataclasses.is_dataclass(identity):
        return None
    return {**dataclasses.asdict(identity), "fingerprint": identity.fingerprint}


def embedding_key(embedding: Mapping[str, Any]) -> Tuple[str, int, Optional[int]]:
    """What decides the vectors of a key text: the model asked for, the width of its vectors and the width asked of
    the endpoint (the query instruction, and the endpoint's address, do not reach a batch of key texts)."""
    request = embedding.get("request_dimensions")
    return (str(embedding.get("requested_model") or ""), int(embedding.get("dimensions") or 0),
            None if request is None else int(request))


def measured_bounds(path: Path | str | None = None) -> Dict[Tuple[str, int, Optional[int]], Optional[float]]:
    """Per embedding, the bound measured for it on the pinned dev set (None when that measurement kept no bound above
    its precision floor). A missing or unreadable file is no measurement."""
    try:
        data = json.loads(Path(BOUNDS if path is None else path).read_text())
    except (OSError, ValueError):
        return {}
    found: Dict[Tuple[str, int, Optional[int]], Optional[float]] = {}
    entries = data.get("measurements") if isinstance(data, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        try:
            if entry["dev_set"]["sha256"] != DEV_SET_SHA256:
                continue
            found[embedding_key(entry["embedding"])] = (float(entry["bound"]) if entry.get("floor_met")
                                                        and entry.get("bound") is not None else None)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return found


_UNREAD: set = set()


def deployment_bounds() -> Dict[Tuple[str, int, Optional[int]], Optional[float]]:
    """The deployment's own measurements (the file ``DEPLOYMENT_BOUNDS`` names), or none; a named file that cannot be
    read is reported once and leaves the shipped measurements in force."""
    named = os.environ.get(DEPLOYMENT_BOUNDS, "").strip()
    if not named:
        return {}
    if not Path(named).is_file() and named not in _UNREAD:
        _UNREAD.add(named)
        logger.warning("%s names %s, which is not a file: the shipped key similarity bounds apply", DEPLOYMENT_BOUNDS,
                       named)
    return measured_bounds(named)


def bound_for(embedder: Any, *, path: Path | str | None = None) -> Tuple[Optional[float], str]:
    """The night's bound for key vectors from this embedder, and why: ``measured`` (the bound measured for its
    embedding), ``below_floor`` (None: its measurement kept no bound above the precision floor, so it finds no
    pairs by meaning) or ``unmeasured`` (``KEY_SIMILARITY_BOUND``: no measurement of its embedding, or an embedder
    without an identity). Without ``path``, the deployment's own measurements (``deployment_bounds``) replace the
    shipped ones for the embeddings they measured."""
    try:
        embedding = embedding_identity(embedder)
    except Exception:
        embedding = None
    bounds = measured_bounds(path) if embedding is not None else {}
    if embedding is not None and path is None:
        bounds.update(deployment_bounds())
    key = embedding_key(embedding) if embedding is not None else None
    if key not in bounds:
        return KEY_SIMILARITY_BOUND, "unmeasured"
    return (bounds[key], "measured") if bounds[key] is not None else (None, "below_floor")


def record(report: Mapping[str, Any], path: Path | str) -> Dict[str, Any]:
    """Adds a report to a bounds file as the measurement of its embedding on its dev set, trimmed to what
    reproduces the choice, replacing an earlier measurement of the same embedding on the same dev set."""
    if not report.get("embedding"):
        raise ValueError("a measurement is recorded only for an embedder with an embedding identity")
    entry = {key: report[key] for key in ("dev_set", "embedding", "precision_floor", "bound", "floor_met",
                                          "at_bound", "pairs")}
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {"about": BOUNDS_ABOUT, "measurements": []}
    same = (embedding_key(entry["embedding"]), entry["dev_set"]["sha256"])
    data["measurements"] = [old for old in data.get("measurements") or []
                            if (embedding_key(old["embedding"]), old["dev_set"]["sha256"]) != same] + [entry]
    path.write_text(json.dumps(data, indent=1) + "\n")
    return entry


BOUNDS_ABOUT = ("Key similarity bounds for the night's contradiction candidates by meaning, each measured on the "
                "labelled dev set for one embedding (python -m colonymind.mind.contradiction_keys --record). The night "
                "uses the bound measured for its own embedding (the same model at the same width); each entry keeps "
                "every pair's score, so its choice can be recomputed.")


async def run(embedder: Any, *, path: Path | str = DEV_SET, sha256: Optional[str] = DEV_SET_SHA256,
              precision_floor: float = PRECISION_FLOOR, label: str = "") -> Dict[str, Any]:
    """The whole measurement on one embedder: the dev set scored, the bound chosen, and the default bound's
    (``KEY_SIMILARITY_BOUND``, an unmeasured embedding's) precision and recall beside it. ``label`` names the embedder in the report, ``embedding`` its identity."""
    pairs = load_dev_set(path, sha256=sha256)
    raw = Path(path).read_bytes()
    scored = await score_dev_set(embedder, pairs)
    chosen = select_bound(scored, precision_floor=precision_floor)
    return {"dev_set": {"name": json.loads(raw).get("name"), "sha256": hashlib.sha256(raw).hexdigest(),
                        "pairs": len(pairs), "same_fact": sum(1 for pair in pairs if pair["same_fact"])},
            "embedder": label, "embedding": embedding_identity(embedder),
            "precision_floor": precision_floor, "bound": chosen["bound"],
            "floor_met": chosen["floor_met"], "at_bound": chosen["at_bound"], "default_bound": KEY_SIMILARITY_BOUND,
            "at_default_bound": _point(scored, KEY_SIMILARITY_BOUND), "curve": chosen["curve"], "pairs": scored}


PROVIDERS = ("openai_api", "cpu", "cuda", "mlx", "native_mlx")


def declared_width(provider: str, dims: int, request_dims: Optional[int]) -> int:
    """The width the sidecar declares for its embedder (``P/server.py``): the configured one, else the one asked of
    the endpoint, else, for an endpoint, learned from its first answer (0), and 384 for a local model."""
    return int(dims or request_dims or (0 if provider == "openai_api" else 384))


async def _sidecar_embedder(provider: str, base_url: Optional[str], model: str, dims: int,
                            request_dims: Optional[int] = None, revision: Optional[str] = None) -> Any:
    """The embedding pipeline the sidecar builds from these settings (``P/server.py``): a local model in process, or
    an OpenAI-compatible endpoint (``openai_api``) asked for ``request_dims`` (None asks for none, and the endpoint
    answers its native width), its vectors checked against ``declared_width``."""
    from colonymind.vector.config import EmbeddingConfig
    from colonymind.vector.embedder import EmbeddingPipeline, make_provider
    made = make_provider(EmbeddingConfig(provider=provider, model_id=model,
                                         dimensions=declared_width(provider, dims, request_dims),
                                         revision=revision or None, request_dimensions=request_dims or None))
    if provider == "openai_api":
        made.configure(base_url or "", os.environ.get("COLONYMIND_EMBED_API_KEY", ""))
    pipeline = EmbeddingPipeline(made)
    await pipeline.warmup()
    return pipeline


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m colonymind.mind.contradiction_keys",
                                     description="Choose the bound for contradiction candidates by meaning on the "
                                                 "labelled dev set.")
    env = os.environ.get

    def width(value: Any) -> int:
        number = int(value)
        if number < 0:
            raise argparse.ArgumentTypeError("a width is a whole number of dimensions, 0 or more")
        return number

    parser.add_argument("--provider", choices=PROVIDERS, default=env("COLONYMIND_EMBED_PROVIDER") or "openai_api",
                        help="how the sidecar embeds: an OpenAI-compatible endpoint (openai_api) or a local model "
                             "(default: COLONYMIND_EMBED_PROVIDER, else openai_api)")
    parser.add_argument("--base-url", default=env("COLONYMIND_EMBED_BASE_URL") or None,
                        help="an OpenAI-compatible embedding endpoint (default: COLONYMIND_EMBED_BASE_URL)")
    parser.add_argument("--model", default=env("COLONYMIND_EMBED_MODEL") or None,
                        help="the embedding model the night uses (default: COLONYMIND_EMBED_MODEL)")
    parser.add_argument("--dims", type=width, default=width(env("COLONYMIND_EMBED_DIMS") or 0),
                        help="the width its vectors are checked against (default: COLONYMIND_EMBED_DIMS; 0: the "
                             "requested width, else an endpoint's first answer, else 384 for a local model)")
    parser.add_argument("--request-dims", type=width, default=width(env("COLONYMIND_EMBED_REQUEST_DIMS") or 0),
                        help="the width asked of the endpoint (default: COLONYMIND_EMBED_REQUEST_DIMS; 0: none, "
                             "the endpoint's native width)")
    parser.add_argument("--precision-floor", type=float, default=PRECISION_FLOOR)
    parser.add_argument("--out", help="write the full report here")
    parser.add_argument("--record", help="add the measurement to this bounds file (the shipped one is "
                                         "colonymind/mind/contradiction_keys_bounds.json)")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("a model is needed: --model, or the sidecar's COLONYMIND_EMBED_MODEL")
    if args.provider == "openai_api" and not args.base_url:
        parser.error("an endpoint is needed: --base-url, or the sidecar's COLONYMIND_EMBED_BASE_URL")
    if args.provider != "openai_api" and args.request_dims:
        parser.error("a width is asked only of an endpoint (openai_api): a local model answers its own")

    async def measure() -> Dict[str, Any]:
        embedder = await _sidecar_embedder(args.provider, args.base_url if args.provider == "openai_api" else None,
                                           args.model, declared_width(args.provider, args.dims, args.request_dims),
                                           args.request_dims or None,
                                           env("COLONYMIND_EMBED_REVISION") or None)
        return await run(embedder, precision_floor=args.precision_floor, label=args.model)

    report = asyncio.run(measure())
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if args.record:
        record(report, args.record)
    print(json.dumps({key: report[key] for key in ("dev_set", "embedder", "embedding", "precision_floor", "bound",
                                                   "floor_met", "at_bound", "default_bound", "at_default_bound")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
