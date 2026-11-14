"""Lesson selection by meaning: the texts compared, the bound and its dev set, and the lesson vectors.

Which lesson reaches a task body or an owner turn (``P/mind/lessons.py``, ``Lessons.for_task`` and
``Lessons.for_turn``) is decided by meaning: the cosine of the lesson's vector and the work's vector, at or above the
bound measured for the running embedding.

- **Texts.** A lesson's document is its title and when-to-use (``lesson_text``), embedded in the index's document
  format (``embed_batch``). The work is an owner turn's query text, or a task's title, topic and concern
  (``task_text``), embedded in the index's query format (``embed_query``: the query instruction, then the text).
- **Lesson vectors** (``LessonVectors``) are a derived cache: a small file in the mind's state directory
  (``VECTORS_FILE``), keyed by lesson id, the content hash of the two fields and the embedding identity's
  fingerprint. ``P/mind/lessons.py`` writes it when a lesson is admitted, superseded, revived or relayed and in the
  night's lesson pass for any that are missing; every write keeps only the lessons that may be served (``served()``),
  so a retired, superseded, cooled or erased lesson loses its vector in the call that closes it; a new embedding
  identity drops the whole file. It is never a source of truth: losing it costs one batch embed.
- **The bound** is chosen offline, by contradiction keys' scheme (``P/mind/contradiction_keys.py``): the dev set
  (``lesson_match_dev.json``, pinned by its SHA-256) of 80 synthetic (lesson, work) pairs, 40 related and 40 not,
  half of those near misses that share terms but not the situation (80 and the halves are my assumption); every pair
  scored by the cosine of its two texts' vectors; ``select_bound`` at ``PRECISION_FLOOR`` (0.8, my assumption,
  contradiction keys' floor): the threshold that finds the most related pairs while at least that share of what it
  finds is related, placed halfway to the next lower score. A bound belongs to the embedding it was measured on (its
  identity but the normalization, which no cosine sees: ``embedding_key``). The shipped
  measurements are in ``lesson_match_bounds.json``; a deployment keeps its own in the file
  ``COLONYMIND_LESSON_MATCH_BOUNDS`` names, which replaces the shipped measurement for each embedding it measured. An
  embedding never measured gets ``DEFAULT_BOUND``, or the tightest bound measured (shipped or the deployment's) when
  that is higher, so it is never related more loosely than a measured one; one whose measurement kept no
  bound above its floor gets nothing by meaning.
- **One rule.** Lexical relevance (shared terms of the lesson's title and when-to-use with the work), meaning, and
  meaning or lexical were measured on the dev set at the floor, on the two embedders the release measured, and
  compared embedder by embedder (the bounds file's ``choice`` keeps the counts): no rule found more than meaning on
  either, and meaning ships alone. Meaning found 16 of
  the 40 related pairs at precision 0.8 on Qwen3-Embedding-8B; on Qwen3-Embedding-0.6B no bound keeps the floor
  (near misses score as high as related pairs; its best, precision 0.67 at recall 1.0), so that embedding gets
  nothing by meaning until a deployment records a measurement of its own. The lexical rule (precision 0.28, recall
  0.18) and the union kept the floor on neither.

Run it on the embedder the sidecar uses, built as the sidecar builds it (every option defaults to the sidecar's own
setting, as in ``P/mind/contradiction_keys.py``):

    python -m colonymind.mind.lesson_match [--provider P] [--base-url URL] [--model MODEL] [--dims N]
                                          [--request-dims N] [--out report.json] [--record bounds.json]

It prints the chosen bound with its precision and recall and the default bound's, writes the full report to
``--out`` and with ``--record`` adds the measurement to a bounds file (replacing an earlier one of the same
embedding on the same dev set). Nothing else is written.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEV_SET = Path(__file__).with_name("lesson_match_dev.json")
BOUNDS = Path(__file__).with_name("lesson_match_bounds.json")
#: The environment setting that names a deployment's own bounds file (``--record`` writes one).
DEPLOYMENT_BOUNDS = "COLONYMIND_LESSON_MATCH_BOUNDS"
DEV_SET_SHA256 = "9fc08b0d29dbcb0d51a67ddec99e8b9917dd8143c9c7876834f87f52c99984cc"
#: My assumption: contradiction keys' floor. At least this share of the pairs a bound finds are related.
PRECISION_FLOOR = 0.8
#: The bound of an embedding never measured (my assumption): no looser than any shipped bound, and above every near
#: miss either shipped measurement scored (at most 0.706), so an unmeasured embedding on a similar scale serves only
#: matches closer than any near miss of the dev set. A test checks it against the shipped bounds; a deployment's own
#: tighter measurement raises it (``default_bound``).
DEFAULT_BOUND = 0.72
#: The file of lesson vectors in the mind's state directory.
VECTORS_FILE = "lesson_vectors.json"
VECTORS_FORMAT = "lesson-vectors-1"
#: How a served lesson matched its work: its own failure class (a task's), or by meaning.
MATCH_KINDS = ("own_class", "meaning")
#: The count of calls that found no work vector (no embedder, a timeout, an identity mismatch).
UNAVAILABLE = "lesson_match_unavailable"


# -- the texts -----------------------------------------------------------------------------------

def _words(text: Any) -> str:
    return " ".join(str(text or "").split())


def lesson_text(title: Any, when_to_use: Any) -> str:
    """A lesson's document: its title and when it applies (the fields today's relevance read)."""
    return f"{_words(title)}. When {_words(when_to_use)}"


def task_text(title: Any, topic: Any, concern: Any) -> str:
    """A task's work: its title, topic and concern."""
    return " ".join(part for part in (_words(title), _words(topic), _words(concern)) if part)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import numpy as np
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        return 0.0
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right) / norm if norm else 0.0


@dataclasses.dataclass(frozen=True)
class Work:
    """A work's vector (query format) and the embedding identity it came from (``index_identity``)."""
    vector: Tuple[float, ...]
    identity: Any

    @property
    def fingerprint(self) -> str:
        return str(getattr(self.identity, "fingerprint", "") or "")


# -- the bound -----------------------------------------------------------------------------------

def load_dev_set(path: Path | str = DEV_SET, *, sha256: Optional[str] = DEV_SET_SHA256) -> List[Dict[str, Any]]:
    """The dev set's pairs. A file whose SHA-256 is not the pinned one is refused (``sha256=None`` reads a changed
    set on purpose)."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"the dev set's sha256 is {digest}, not the pinned {sha256}")
    return list(json.loads(raw)["pairs"])


def pair_texts(pair: Mapping[str, Any]) -> Tuple[str, str]:
    """(the lesson's document, the work's text) of a dev pair, exactly as the runtime builds them."""
    lesson, work = pair["lesson"], pair["work"]
    text = (_words(work.get("text")) if work.get("kind") == "turn"
            else task_text(work.get("title"), work.get("topic"), work.get("concern")))
    return lesson_text(lesson["title"], lesson["when_to_use"]), text


async def score_dev_set(embedder: Any, pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Each pair's cosine (``id``, ``related``, ``score``): the lessons' documents in one ``embed_batch`` (the
    document format), each work through ``embed_query`` (the query format), as the runtime embeds them."""
    texts = [pair_texts(pair) for pair in pairs]
    documents = list(dict.fromkeys(document for document, _ in texts))
    vectors = list(await embedder.embed_batch(documents))
    if len(vectors) != len(documents):
        raise ValueError("the embedder answered a different number of vectors than lesson documents")
    by_document = dict(zip(documents, vectors))
    works: Dict[str, Any] = {}
    for _, work in texts:
        if work not in works:
            works[work] = await embedder.embed_query(work)
    return [{"id": str(pair["id"]), "related": bool(pair["related"]),
             "score": cosine(by_document[document], works[work])} for pair, (document, work) in zip(pairs, texts)]


def _labelled(scored: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The scored pairs in contradiction keys' shape (its label is ``same_fact``)."""
    return [{"same_fact": bool(row["related"]), "score": float(row["score"])} for row in scored]


def select_bound(scored: Sequence[Mapping[str, Any]], *, precision_floor: float = PRECISION_FLOOR) -> Dict[str, Any]:
    """Contradiction keys' rule on the scored pairs (``contradiction_keys.select_bound``): ``bound``, ``floor_met``,
    ``at_bound`` and ``curve``, where ``tp`` counts related pairs found."""
    from .contradiction_keys import select_bound as choose
    return choose(_labelled(scored), precision_floor=precision_floor)


def point(scored: Sequence[Mapping[str, Any]], threshold: float) -> Dict[str, Any]:
    """What a bound of ``threshold`` does on the scored pairs."""
    from .contradiction_keys import _point
    return _point(_labelled(scored), threshold)


def embedding_identity(embedder: Any) -> Optional[Dict[str, Any]]:
    """An embedding pipeline's identity with its fingerprint, as the sidecar's vector index keeps it; None for an
    embedder without one."""
    identity = getattr(embedder, "index_identity", None)
    if identity is None or not dataclasses.is_dataclass(identity):
        return None
    return {**dataclasses.asdict(identity), "fingerprint": identity.fingerprint}


#: An embedding as a measurement names it (``embedding_key``).
Key = Tuple[str, str, str, int, Optional[int], str, str, str, str]


def embedding_key(embedding: Mapping[str, Any]) -> Key:
    """What decides the cosines a bound is chosen on: the model asked for and the one that answered, its declared
    revision and quantization, the width of its vectors and the width asked of the endpoint, and the document and
    query formats (the lesson and the work are embedded in them). Everything of the embedding identity but its
    normalization, which a cosine does not see; the endpoint's address is not part of it. A bound is read only for the
    same key, so another served model, revision, quantization or format is unmeasured."""
    request = embedding.get("request_dimensions")
    return (str(embedding.get("requested_model") or ""), str(embedding.get("served_model") or ""),
            str(embedding.get("declared_revision") or ""), int(embedding.get("dimensions") or 0),
            None if request is None else int(request), str(embedding.get("document_format") or ""),
            str(embedding.get("query_format") or ""), str(embedding.get("quantization") or ""),
            str(embedding.get("query_transform", "identity")))


_PARSED: Dict[Tuple[str, int], Dict[Key, Optional[float]]] = {}


def measured_bounds(path: Path | str | None = None) -> Dict[Key, Optional[float]]:
    """Per embedding, the bound measured for it on the pinned dev set (None when that measurement kept no bound
    above its floor). A missing or unreadable file is no measurement. Parsed once per version of the file."""
    path = Path(BOUNDS if path is None else path)
    try:
        stamp = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return {}
    if stamp in _PARSED:
        return dict(_PARSED[stamp])
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    found: Dict[Key, Optional[float]] = {}
    entries = data.get("measurements") if isinstance(data, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        try:
            if entry["dev_set"]["sha256"] != DEV_SET_SHA256:
                continue
            found[embedding_key(entry["embedding"])] = (float(entry["bound"]) if entry.get("floor_met")
                                                        and entry.get("bound") is not None else None)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    _PARSED[stamp] = dict(found)
    return found


_UNREAD: set = set()


def deployment_bounds() -> Dict[Key, Optional[float]]:
    """The deployment's own measurements (the file ``DEPLOYMENT_BOUNDS`` names), or none; a named file that cannot
    be read is reported once and leaves the shipped measurements in force."""
    named = os.environ.get(DEPLOYMENT_BOUNDS, "").strip()
    if not named:
        return {}
    if not Path(named).is_file() and named not in _UNREAD:
        _UNREAD.add(named)
        logger.warning("%s names %s, which is not a file: the shipped lesson match bounds apply", DEPLOYMENT_BOUNDS,
                       named)
    return measured_bounds(named)


def default_bound(bounds: Mapping[Key, Optional[float]]) -> float:
    """The bound of an embedding never measured: ``DEFAULT_BOUND``, or the tightest of the measured ``bounds`` in
    force when that is higher, so no unmeasured embedding is related more loosely than a measured one."""
    return max([DEFAULT_BOUND, *(bound for bound in bounds.values() if bound is not None)])


def bound_for(embedding: Optional[Mapping[str, Any]], *, path: Path | str | None = None) -> Tuple[Optional[float], str]:
    """The bound for work and lesson vectors of this embedding (an identity as ``embedding_identity`` gives it), and
    why: ``measured``, ``below_floor`` (None: nothing by meaning) or ``unmeasured`` (``default_bound``; also for no
    identity). Without ``path`` the deployment's own measurements replace the shipped ones they measured."""
    bounds = measured_bounds(path)
    if path is None:
        bounds.update(deployment_bounds())
    key = embedding_key(embedding) if embedding is not None else None
    if key is None or key not in bounds:
        return default_bound(bounds), "unmeasured"
    return (bounds[key], "measured") if bounds[key] is not None else (None, "below_floor")


def identity_mapping(identity: Any) -> Optional[Dict[str, Any]]:
    """An ``EmbeddingIdentity`` (or a mapping of one) as ``bound_for`` reads it."""
    if identity is None:
        return None
    if isinstance(identity, Mapping):
        return dict(identity)
    if dataclasses.is_dataclass(identity):
        return {**dataclasses.asdict(identity), "fingerprint": getattr(identity, "fingerprint", "")}
    return None


BOUNDS_ABOUT = ("Lesson match bounds: for each embedding measured, the cosine at or above which a lesson and a work "
                "(a task or an owner turn) are related, chosen on the labelled dev set (python -m "
                "colonymind.mind.lesson_match --record). Each entry keeps every pair's score, so its choice can be "
                "recomputed.")


def record(report: Mapping[str, Any], path: Path | str) -> Dict[str, Any]:
    """Adds a report to a bounds file as the measurement of its embedding on its dev set, trimmed to what
    reproduces the choice, replacing an earlier measurement of the same embedding on the same dev set. Every other
    key of the file is kept."""
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


async def run(embedder: Any, *, path: Path | str = DEV_SET, sha256: Optional[str] = DEV_SET_SHA256,
              precision_floor: float = PRECISION_FLOOR, label: str = "") -> Dict[str, Any]:
    """The whole measurement on one embedder: the dev set scored, the bound chosen, and the default bound's
    precision and recall beside it."""
    raw = Path(path).read_bytes()
    pairs = load_dev_set(path, sha256=sha256)
    scored = await score_dev_set(embedder, pairs)
    chosen = select_bound(scored, precision_floor=precision_floor)
    return {"dev_set": {"name": json.loads(raw).get("name"), "sha256": hashlib.sha256(raw).hexdigest(),
                        "pairs": len(pairs), "related": sum(1 for pair in pairs if pair["related"])},
            "embedder": label, "embedding": embedding_identity(embedder), "precision_floor": precision_floor,
            "bound": chosen["bound"], "floor_met": chosen["floor_met"], "at_bound": chosen["at_bound"],
            "default_bound": DEFAULT_BOUND, "at_default_bound": point(scored, DEFAULT_BOUND),
            "curve": chosen["curve"], "pairs": scored}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .contradiction_keys import PROVIDERS, _sidecar_embedder, declared_width
    parser = argparse.ArgumentParser(prog="python -m colonymind.mind.lesson_match",
                                     description="Choose the bound for lesson selection by meaning on the labelled "
                                                 "dev set.")
    env = os.environ.get

    def width(value: Any) -> int:
        number = int(value)
        if number < 0:
            raise argparse.ArgumentTypeError("a width is a whole number of dimensions, 0 or more")
        return number

    parser.add_argument("--provider", choices=PROVIDERS, default=env("COLONYMIND_EMBED_PROVIDER") or "openai_api",
                        help="how the sidecar embeds (default: COLONYMIND_EMBED_PROVIDER, else openai_api)")
    parser.add_argument("--base-url", default=env("COLONYMIND_EMBED_BASE_URL") or None,
                        help="an OpenAI-compatible embedding endpoint (default: COLONYMIND_EMBED_BASE_URL)")
    parser.add_argument("--model", default=env("COLONYMIND_EMBED_MODEL") or None,
                        help="the embedding model the sidecar uses (default: COLONYMIND_EMBED_MODEL)")
    parser.add_argument("--dims", type=width, default=width(env("COLONYMIND_EMBED_DIMS") or 0),
                        help="the width its vectors are checked against (default: COLONYMIND_EMBED_DIMS)")
    parser.add_argument("--request-dims", type=width, default=width(env("COLONYMIND_EMBED_REQUEST_DIMS") or 0),
                        help="the width asked of the endpoint (default: COLONYMIND_EMBED_REQUEST_DIMS; 0: none)")
    parser.add_argument("--precision-floor", type=float, default=PRECISION_FLOOR)
    parser.add_argument("--out", help="write the full report here")
    parser.add_argument("--record", help="add the measurement to this bounds file (the shipped one is "
                                         "colonymind/mind/lesson_match_bounds.json)")
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
                                           args.request_dims or None, env("COLONYMIND_EMBED_REVISION") or None)
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


# -- the lesson vectors -----------------------------------------------------------------------------

def _encode(vector: Sequence[float]) -> str:
    import numpy as np
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def _decode(text: str) -> Tuple[float, ...]:
    import numpy as np
    return tuple(float(value) for value in np.frombuffer(base64.b64decode(text.encode("ascii")), dtype="<f4"))


class LessonVectors:
    """The lesson vectors: ``{lesson id: (text hash, vector)}`` for one embedding identity (its fingerprint), in a
    small file (``path``; None keeps them in memory only). A vector is read only for the same lesson id, text hash
    and fingerprint; every write keeps only the entries of ``keep`` (the served lessons' ids and text hashes), and a
    write for another fingerprint starts from nothing. The file is replaced atomically, readable by its owner only.
    Memory follows a write at once; when the file could not follow (``pending``), the next write retries it, so an
    erased lesson's vector does not stay on disk because memory had already let it go."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self._fingerprint: Optional[str] = None
        self._vectors: Dict[str, Tuple[str, Tuple[float, ...]]] = {}
        self._loaded = False
        self._pending = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict) or data.get("format") != VECTORS_FORMAT:
                return
            vectors = {str(ident): (str(entry["hash"]), _decode(entry["vector"]))
                       for ident, entry in (data.get("vectors") or {}).items()}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return
        self._fingerprint, self._vectors = (str(data.get("fingerprint") or "") or None), vectors

    @property
    def fingerprint(self) -> Optional[str]:
        self._load()
        return self._fingerprint

    @property
    def pending(self) -> bool:
        """Whether the file may still hold vectors memory no longer does: its last write failed, and the next write
        retries it."""
        return self._pending

    def ids(self) -> List[str]:
        self._load()
        return sorted(self._vectors)

    def get(self, ident: str, digest: str, fingerprint: str) -> Optional[Tuple[float, ...]]:
        self._load()
        if not fingerprint or fingerprint != self._fingerprint:
            return None
        entry = self._vectors.get(str(ident))
        return entry[1] if entry is not None and entry[0] == digest else None

    def write(self, keep: Mapping[str, str], *, fingerprint: Optional[str] = None,
              add: Optional[Mapping[str, Tuple[str, Sequence[float]]]] = None) -> bool:
        """Keep only ``keep``'s entries (``{id: text hash}``), adding ``add`` (vectors for ``fingerprint``). True when
        the file was written: anything changed (it is then rewritten, or removed when nothing is left), or an earlier
        write of it failed (``pending``). A write that fails raises and leaves it pending."""
        self._load()
        fingerprint = fingerprint or self._fingerprint
        vectors = dict(self._vectors) if fingerprint == self._fingerprint else {}
        for ident, (digest, vector) in (add or {}).items():
            vectors[str(ident)] = (str(digest), tuple(float(value) for value in vector))
        vectors = {ident: entry for ident, entry in vectors.items() if keep.get(ident) == entry[0]}
        if fingerprint == self._fingerprint and vectors == self._vectors and not self._pending:
            return False
        self._fingerprint, self._vectors = (fingerprint if vectors else None), vectors
        self._pending = True
        self._persist()
        self._pending = False
        return True

    def _persist(self) -> None:
        if self.path is None:
            return
        if not self._vectors:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return
        payload = json.dumps({"format": VECTORS_FORMAT, "fingerprint": self._fingerprint,
                              "vectors": {ident: {"hash": digest, "vector": _encode(vector)}
                                          for ident, (digest, vector) in sorted(self._vectors.items())}})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".lesson-vectors-", dir=str(self.path.parent))
        try:
            with os.fdopen(handle, "w") as out:
                out.write(payload)
            os.chmod(name, 0o600)
            os.replace(name, self.path)
        except BaseException:
            try:
                os.unlink(name)
            except OSError:
                pass
            raise


def served_hashes(lessons: Iterable[Any]) -> Dict[str, str]:
    """``{lesson id: text hash}`` of lessons, as ``LessonVectors.write`` keeps them."""
    return {str(lesson.id): text_hash(lesson_text(lesson.title, lesson.when_to_use)) for lesson in lessons}


__all__ = ["BOUNDS", "DEFAULT_BOUND", "DEPLOYMENT_BOUNDS", "DEV_SET", "DEV_SET_SHA256", "LessonVectors",
           "MATCH_KINDS", "PRECISION_FLOOR", "UNAVAILABLE", "VECTORS_FILE", "Work", "bound_for", "cosine",
           "default_bound", "embedding_identity", "embedding_key", "identity_mapping", "lesson_text", "load_dev_set",
           "measured_bounds", "record", "run", "score_dev_set", "select_bound", "served_hashes", "task_text",
           "text_hash"]


if __name__ == "__main__":
    sys.exit(main())
