"""Recall at a memory budget: what the host would inject for each question, captured for offline re-scoring.

For one question, one arm and one budget the runner copies the arm's store (a replay's instance, see
``replay.py``) into a fresh directory, serves it in process with the clock at the question's time, and asks
``POST /v1/host/context/assemble`` what a host injects: the question as the incoming message, in a new session,
``inline_chars`` set to the budget, the recall trace on (the instance key manages the instance), and a host wait
long enough that it never cuts recall short. Recall's own ceiling (``COLONYMIND_RECALL_CONTEXT_MAX_CHARS``) is set
to the budget as well, up to the ceiling recall accepts (``recall_ceiling``): the host's allowance alone would
leave the packet at its default, so a larger budget could not fill. The copy is served with claim extraction off,
so reading calls no chat model and never changes the formed store. A question is read at its time, or at the
store's last moment when the replayed history ends after it (sessions stamped close together are replayed one
after another; the row records the shift). The clock stands at that time while the question is read
(``paired_body.set_clock``).

A store whose formation is incomplete (``replay.formation``: a refused turn, a drain that ran out its budget, a
night that failed) is refused unless the read is told to measure it anyway (``allow_incomplete``), and every row
carries the store's formation record; ``replay`` and ``import`` never count an incomplete store as formed.

A capture row (JSON lines) keeps everything a re-score needs: the sections the host would inject, the block fit's
report, the recall trace for stage attribution, every call recall made to the reranker (``ProviderCapture``),
and the question's gold identifiers (``replay.gold``). ``score`` reads a row only: reach is structural
(``scoring.reach``) between the gold identifiers and the identifiers each shown item cites, item by item in the order
the host renders the sections (``paired_rescore.section_items``), and ``chars`` is what the sections cost against the
budget (``memory.blocks.cost``). ``rescore`` reads the capture file alone, so a run re-scores identically offline.

**Arms** (``ARMS``) are the recall-time ColonyMind arms, each one change from all-on: the store (the formed store,
the store before the night, the import-only store) or a setting (no embedder, no reranker, no resident-block fit,
the mind off). **Budgets** default to 4,000 and 24,000 characters (``BUDGETS``); 10,000 is the host's default
inline cap. Baselines use the serving packer to fit ranked items to the budget. A context over its budget
(the arm with the block fit off returns what it
assembled) is cut the same way before it is scored (``fit``): the row keeps what the host returned, and the score
says what was cut.

    python benchmarks/public_memory/runner.py replay --dataset longmemeval_s --question ID [...] --out DIR
    python benchmarks/public_memory/runner.py import --dataset longmemeval_s --question ID [...] --out DIR
    python benchmarks/public_memory/runner.py read --stores DIR --arm all-on --budget 4000 --capture FILE
    python benchmarks/public_memory/runner.py rescore --capture FILE --output FILE
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.public_memory import replay
from benchmarks.public_memory import scoring
from benchmarks.public_memory import paired_rescore as rescoring

from colonymind.qualification import paired_body  # noqa: E402  (replay.py puts the sidecar on the path)
from colonymind.plasticity.recall_qualify import QueryAdapterPipeline  # noqa: E402

PROTOCOL = 'public-memory-capture-1'
BUDGETS = (4000, 24000)
HOST_INLINE_CAP = 10000
# Request the largest allowance; the instance's configured assembly deadline still applies.
WAIT_MS = 600_000
# Reading never runs a model projection: whatever a formed store still owes is left as it is.
READ_ONLY = {'COLONYMIND_SOURCE_CLAIMS': 'off'}
STORES = {'formed': ('', replay.MANIFEST), 'before_night': (replay.BEFORE_NIGHT, replay.MANIFEST),
          'import_only': ('', replay.MANIFEST)}
# Each arm is one change from all-on: the store it reads, or a setting read at recall time.
ARMS: Dict[str, Dict[str, Any]] = {
    'all-on': {'store': 'formed'},
    'lexical-only': {'store': 'formed', 'overrides': {'COLONYMIND_EMBED_PROVIDER': 'skip'}},
    'reranker-off': {'store': 'formed', 'overrides': {'COLONYMIND_RECALL_RERANK': 'off'}},
    'no-night': {'store': 'before_night'},
    'memory-core': {'store': 'formed', 'mind': {'enabled': False}},
    'import-only': {'store': 'import_only'},
}
_SAFE = re.compile(r'[^A-Za-z0-9._-]+')


def safe_name(value: str) -> str:
    """A directory name for an id: its readable part and a short hash, so two ids never share a directory."""
    return (_SAFE.sub('-', value).strip('-')[:64] or 'item') + '-' + hashlib.sha256(value.encode()).hexdigest()[:8]


# ------------------------------------------------------------------ scoring a row

def section_cost(section: Mapping[str, Any]) -> int:
    from colonymind.memory import blocks
    return blocks.cost(section.get('title') or section['id'], section['body'])


def shown(sections: Iterable[Mapping[str, Any]]) -> list:
    """The identifiers each shown item cites, in the order the host renders the sections."""
    return [item['refs'] for section in sections
            for item in rescoring.section_items(section['body'], section['id'])]


def fit(sections: Iterable[Mapping[str, Any]], budget: int) -> list:
    """The sections a reader is shown within ``budget`` characters (``section_cost``), in the order the host renders
    them: each whole section while it fits, then the whole lines of the first one that does not, up to the first
    line that does not fit, which ends the context. A context within its budget is kept as it
    is."""
    kept, used = [], 0
    for section in sections:
        cost = section_cost(section)
        if used + cost <= budget:
            kept.append(section)
            used += cost
            continue
        lines = []
        for line in section['body'].split('\n'):
            if used + section_cost({**section, 'body': '\n'.join([*lines, line])}) > budget:
                break
            lines.append(line)
        if lines:
            kept.append({**section, 'body': '\n'.join(lines)})
        break
    return kept


def score(row: Mapping[str, Any]) -> dict:
    """A capture row scored from the row alone: ``scoring.summarize``'s row fields, for what fits the row's budget
    (``fit``). ``returned_chars`` is what the host returned, and ``cut`` whether the budget cut it."""
    budget = int(row['budget_chars'])
    sections = fit(row['sections'], budget)
    items = shown(sections)
    return {'question': row['question'], 'conversation': row['conversation'], 'type': row['type'],
            'arm': row['arm'], 'reach': scoring.reach(row['gold'], items), 'items': len(items),
            'chars': sum(section_cost(section) for section in sections), 'budget': budget,
            **({'query_adapter': row['query_adapter']} if 'query_adapter' in row else {}),
            'returned_chars': sum(section_cost(section) for section in row['sections']),
            'cut': sections != list(row['sections'])}


def rescore(capture: Path) -> list:
    """Every row of a capture file, scored again from the file alone."""
    rows = []
    for line in Path(capture).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('protocol') != PROTOCOL:
                raise ValueError('Not a public memory capture row')
            rows.append(score(row))
    return rows


# ------------------------------------------------------------------ reading a store

def arm_config(home: Path, arm: Mapping[str, Any]) -> None:
    """Apply an arm's configuration change to a copied instance's ``colonymind.yaml``."""
    from colonymind.config import read_raw
    config = read_raw(home / 'colonymind.yaml')
    if arm.get('context'):
        config['context'] = {**(config.get('context') or {}), **arm['context']}
    if arm.get('mind'):
        config['mind'] = {**(config.get('mind') or {}), **arm['mind']}
    replay.write_config(home, config)


def _store_home(store: Path, name: str) -> Path:
    directory, _ = STORES[name]
    home = Path(store) / directory / replay.INSTANCE if directory else Path(store) / replay.INSTANCE
    if not (home / 'colonymind.yaml').is_file():
        raise replay.ReplayError(f'{home} holds no replayed instance')
    return home


def store_formation(manifest: Mapping[str, Any]) -> dict:
    """A manifest's formation record; a manifest without one is incomplete."""
    return dict(manifest.get('formation') or {'complete': False, 'problems': ['no formation record']})


def formed_already(root: Path) -> bool:
    """Whether ``root`` holds a store formed as specified, one a campaign may reuse."""
    path = Path(root) / replay.MANIFEST
    return path.is_file() and store_formation(replay.load_manifest(root)).get('complete') is True


def _store_end(manifest: Mapping[str, Any]) -> float:
    from datetime import datetime
    return max(datetime.fromisoformat(event['clock']).timestamp() for event in manifest['events'])


def recall_ceiling(budget: int) -> int:
    """Recall's own character ceiling for ``budget``: the budget, within what recall accepts (the memory block's
    limit, which is the packet's own ceiling in ``memory/search.py``)."""
    from colonymind.memory import blocks
    return max(0, min(int(budget), blocks.LIMITS['colonymind-memory']))


def _request(owner: str, question, session_id: str, budget: int) -> dict:
    return {'identity': dict(replay.HOST), 'context': {'session_id': session_id, 'contact_id': owner,
                                                       'channel_id': replay.CHANNEL, 'timezone': 'UTC'},
            'incoming_message': {'role': 'user', 'content': question.question},
            'inline_chars': int(budget), 'trace': True, 'wait_ms': WAIT_MS}


async def read_question(store: Path, question, *, arm: str, budget: int, work: Path, reranker=None,
                        keep: bool = False, allow_incomplete: bool = False, query_adapter: Path | None = None) -> dict:
    """One capture row: ``question`` read from ``store`` (a replay's directory) under ``arm`` at ``budget``.
    ``work`` is a new or empty directory for the copy; it is removed afterwards unless ``keep``. ``reranker``
    replaces the configured one (tests). An incomplete store is refused unless ``allow_incomplete``."""
    spec = ARMS[arm]
    manifest = replay.load_manifest(store)
    if spec['store'] == 'import_only' and manifest.get('store') != 'import_only':
        raise replay.ReplayError(f'the {arm} arm reads an import-only store')
    if spec['store'] != 'import_only' and manifest.get('store') != 'formed':
        raise replay.ReplayError(f'the {arm} arm reads a formed store')
    formation = store_formation(manifest)
    if not formation.get('complete') and not allow_incomplete:
        raise replay.ReplayError(f'{store} is incomplete ({"; ".join(formation.get("problems") or [])}): form it '
                                 'again, or read it with allow_incomplete to measure it as it is')
    source = _store_home(store, spec['store'])
    work = replay._fresh(work)
    asked = question.asked_at.timestamp()
    read_at = max(asked, _store_end(manifest))
    # The copy's one move back, before it exists: to when the question is asked.
    paired_body.set_clock(read_at, state=work)
    home = work / replay.INSTANCE
    shutil.copytree(source, home, symlinks=True)
    arm_config(home, spec)
    instance = replay.open_instance(work, home)
    session_id = f'question:{question.id}'
    # A reranker handed in is a configured one, which recall is told to use (``config.apply_environment``).
    # Recall's ceiling follows the budget, as the host's allowance does: both are the budget under test.
    ceiling = recall_ceiling(budget)
    overrides = {**READ_ONLY, **({'COLONYMIND_RECALL_RERANK': 'on'} if reranker is not None else {}),
                 **spec.get('overrides', {}), 'COLONYMIND_RECALL_CONTEXT_MAX_CHARS': str(ceiling)}
    try:
        async with replay.serve(instance, overrides=overrides, capture_reranker=reranker is None) as served:
            if reranker is not None:
                from colonymind.api.routers import host
                from colonymind.qualification.recall_capture import ProviderCapture
                served.reranker = ProviderCapture(reranker)
                host.set_reranker(served.reranker)
            started = time.perf_counter()
            with _query_adapter(query_adapter) as adapter:
                response = await served.client.post('/v1/host/context/assemble',
                                                    json=_request(instance.owner, question, session_id, budget))
            elapsed = (time.perf_counter() - started) * 1000
            data = response.json() if response.status_code == 200 else {}
            calls = served.reranker.recorder.take() if served.reranker is not None else []
            if reranker is not None:
                from colonymind.api.routers import host
                host.set_reranker(None)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    sections = [{'id': section['id'], 'title': section.get('title'), 'body': section['body']}
                for section in data.get('sections') or []]
    row = {'protocol': PROTOCOL, 'question': question.id, 'conversation': question.conversation,
           'type': question.type, 'abstention': bool(question.abstention), 'arm': arm, 'store': spec['store'],
           'budget_chars': int(budget), 'recall_max_chars': ceiling, 'asked_at': question.asked_at.isoformat(),
           'read_at': replay.iso(read_at), 'shift_seconds': read_at - asked, 'formation': formation,
           'session_id': session_id, 'status': response.status_code,
           'elapsed_ms': round(elapsed, 1), 'sections': sections, 'budget': data.get('budget'),
           'trace': data.get('trace'), 'rerank_calls': calls,
           'evidence': [[evidence.session, evidence.turn] for evidence in question.evidence],
           'gold': [sorted(item) for item in replay.gold(manifest, question)]}
    if adapter is not None:
        row['query_adapter'] = adapter.adapter_digest
    return row


@contextmanager
def _query_adapter(directory):
    from colonymind import vector
    from colonymind.api.routers import host
    if directory is None:
        yield None
        return
    original, embedder = vector.get_pipeline(), host._embedder
    if original is None:
        raise ValueError('--query-adapter requires the configured embedding pipeline')
    adapted = QueryAdapterPipeline(original, directory)
    vector.set_pipeline(adapted)
    host.set_embedder(adapted)
    try:
        yield adapted
    finally:
        vector.set_pipeline(original)
        host.set_embedder(embedder)


async def read(store: Path, questions, *, arm: str, budgets=BUDGETS, capture: Path, work: Path, reranker=None,
               keep: bool = False, allow_incomplete: bool = False, query_adapter: Path | None = None) -> list:
    """Read every question of ``store`` under ``arm`` at every budget, in the order they are asked; each row is
    appended to ``capture`` as it is read. Returns the rows' scores."""
    scores = []
    for budget in budgets:
        for question in sorted(questions, key=lambda item: (item.asked_at, item.id)):
            try:
                row = await read_question(store, question, arm=arm, budget=budget, reranker=reranker, keep=keep,
                                          work=Path(work) / safe_name(f'{arm}:{budget}:{question.id}'),
                                          allow_incomplete=allow_incomplete, query_adapter=query_adapter)
            finally:
                paired_body.uninstall_clock()
            with Path(capture).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            scores.append(score(row))
    return scores


# ------------------------------------------------------------------ the command line

def load_dataset(name: str, data_dir: Optional[Path], offline: bool):
    adapter = importlib.import_module(f'benchmarks.public_memory.datasets.{name}')
    return adapter.load(data_dir, download=not offline)


def _questions(dataset, ids) -> Dict[str, list]:
    wanted = set(ids)
    chosen = [question for question in dataset.questions if not wanted or question.id in wanted]
    missing = wanted - {question.id for question in chosen}
    if missing:
        raise SystemExit(f'no question {sorted(missing)[0]} in {dataset.name}')
    by_conversation: Dict[str, list] = {}
    for question in chosen:
        by_conversation.setdefault(question.conversation, []).append(question)
    return by_conversation


def _ids(args) -> list:
    ids = list(args.question or [])
    if args.subset:
        ids += json.loads(Path(args.subset).read_text(encoding='utf-8'))['ids']
    return ids


def _endpoints(path: Optional[Path]) -> 'replay.Endpoints':
    if path is None:
        return replay.Endpoints()
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return replay.Endpoints(**{key: data[key] for key in ('embed_url', 'embed_model', 'embed_dims', 'rerank_url',
                                                          'rerank_model', 'settings') if key in data})


def _form(args, *, imported: bool) -> int:
    dataset = load_dataset(args.dataset, args.data_dir, args.offline)
    llm_config = None if imported else json.loads(Path(args.llm_config).read_text(encoding='utf-8'))
    endpoints = _endpoints(args.endpoints)
    incomplete = 0
    for conversation_id, questions in _questions(dataset, _ids(args)).items():
        root = Path(args.out) / safe_name(conversation_id)
        if formed_already(root):
            print(json.dumps({'conversation': conversation_id, 'store': str(root), 'status': 'already formed'}))
            continue
        if (root / replay.MANIFEST).is_file():
            # Never counted as formed, and never formed again in place: the directory must be new or empty.
            incomplete += 1
            print(json.dumps({'conversation': conversation_id, 'store': str(root), 'status': 'incomplete',
                              'formation': store_formation(replay.load_manifest(root))}), flush=True)
            continue
        described = {'name': dataset.name, 'sha256': dataset.sha256}
        try:
            if imported:
                manifest = asyncio.run(replay.import_only(dataset.conversations[conversation_id], questions, root,
                                                          endpoints=endpoints, dataset=described))
            else:
                manifest = asyncio.run(replay.replay(dataset.conversations[conversation_id], questions, root,
                                                     llm_config=llm_config, endpoints=endpoints,
                                                     dataset=described))
        finally:
            paired_body.uninstall_clock()
        incomplete += not manifest['formation']['complete']
        print(json.dumps({'conversation': conversation_id, 'store': str(root), 'counts': manifest['counts'],
                          'refused': len(manifest.get('refused') or []), 'formation': manifest['formation']}),
              flush=True)
    return 1 if incomplete else 0


def _read(args) -> int:
    dataset = load_dataset(args.dataset, args.data_dir, args.offline)
    budgets = tuple(args.budget or BUDGETS)
    for conversation_id, questions in _questions(dataset, _ids(args)).items():
        store = Path(args.stores) / safe_name(conversation_id)
        scores = asyncio.run(read(store, questions, arm=args.arm, budgets=budgets, capture=args.capture,
                                  work=args.work, keep=args.keep, allow_incomplete=args.allow_incomplete,
                                  query_adapter=getattr(args, 'query_adapter', None)))
        for item in scores:
            print(json.dumps({key: item[key] for key in ('question', 'arm', 'budget', 'chars', 'items')}
                             | {'reached': item['reach']['any']}), flush=True)
    return 0


def _rescore(args) -> int:
    rows = rescore(args.capture)
    arms = sorted({row['arm'] for row in rows})
    report = {'protocol': PROTOCOL, 'rows': rows,
              'summary': {arm: {budget: scoring.summarize([row for row in rows if row['arm'] == arm
                                                           and row['budget'] == budget])
                                for budget in sorted({row['budget'] for row in rows if row['arm'] == arm})}
                          for arm in arms}}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding='utf-8')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--dataset', required=True, choices=('longmemeval_s', 'beam'))
    common.add_argument('--data-dir', type=Path)
    common.add_argument('--offline', action='store_true')
    common.add_argument('--question', action='append', help='a question id; repeat for more')
    common.add_argument('--subset', type=Path, help='a subset record (datasets subset --out)')
    form = commands.add_parser('replay', parents=[common], help='form the stores (S1) with the benchmark model')
    form.add_argument('--llm-config', type=Path, required=True,
                      help='the host model configuration, as the sidecar persists it; kept in memory only')
    form.add_argument('--endpoints', type=Path, help='embedder and reranker (Endpoints fields, JSON)')
    form.add_argument('--out', type=Path, required=True)
    importer = commands.add_parser('import', parents=[common], help='the import-only stores (S2)')
    importer.add_argument('--endpoints', type=Path)
    importer.add_argument('--out', type=Path, required=True)
    reader = commands.add_parser('read', parents=[common], help='read every question under one arm')
    reader.add_argument('--stores', type=Path, required=True)
    reader.add_argument('--arm', choices=sorted(ARMS), default='all-on')
    reader.add_argument('--budget', type=int, action='append')
    reader.add_argument('--capture', type=Path, required=True)
    reader.add_argument('--work', type=Path, required=True, help='where each read copies its store')
    reader.add_argument('--keep', action='store_true', help='keep each copy after it is read')
    reader.add_argument('--query-adapter', type=Path, help='Portable global query revision directory')
    reader.add_argument('--allow-incomplete', action='store_true',
                        help='read a store whose formation is incomplete; each row says so')
    again = commands.add_parser('rescore', help='score a capture file again, from the file alone')
    again.add_argument('--capture', type=Path, required=True)
    again.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'replay':
        return _form(args, imported=False)
    if args.command == 'import':
        return _form(args, imported=True)
    if args.command == 'read':
        return _read(args)
    return _rescore(args)


if __name__ == '__main__':
    sys.exit(main())
