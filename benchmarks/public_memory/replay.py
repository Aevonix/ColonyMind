"""Replay a public dataset's history into ColonyMind on its own dates: the formed store, and the import-only store.

**The formed store (S1).** Every (user, assistant) pair of every session goes through ``POST /v1/host/turns/sync``
of the sidecar itself (``colonymind.server.create_app`` and its lifespan, served in process), with the instance
key, which the credential table grants the one synthetic owner on the replay's platform (the benchmark credential
of the paired track), in person scope. Claim extraction, appraisal, capture, opinions and vectors run as they do
in production: the sidecar's own background worker, on the model router the replay installs.

The clock is absolute (``paired_body.set_clock``): it moves back once, to the first session, before the instance
exists, and only forward after that. Before each session the background queues drain (bounded: ``drain``), then
the clock moves to the session's end, and each pair's ``occurred_at`` is the session's start plus k seconds, so
the sidecar's "never later than now" rule never clamps a replayed time. A session's start is its dataset time,
or the previous session's end when that is later (sessions stamped in the same minute): the schedule
(``schedule``) depends only on the data. The clock stands between these moves: processing a turn, a drain or the
night takes real time but no replayed time, so the store's own stamps (``ingested_at``, a claim's
``recorded_at``), the night's date and the time a question is read at depend only on the data, never on how fast
the model answered. The harness's own waits (``drain``) run on the monotonic clock. The manifest records the clock
at every step.

After the last session and a final drain the sidecar stops, the instance is copied (``before-night``: the store
without the night), and it starts again for one night (``POST /v1/mind/consolidate``) at the earliest question's
time. The mind's own timer does not run during a replay, as in the paired harness: the question-time night is
its only one. The manifest keeps the night's own record, its failed calls (``errors``) and whether it stopped
early (``exhausted``) included: the route answers 200 for a night whose every model call failed.

Whether a store was formed as specified is the manifest's ``formation`` (``formation``): a turn the sidecar
refused, a drain that ran out its budget with work still running, or a night that did not run or whose calls
failed leaves it incomplete, with each problem named. The runner never counts an incomplete store as formed.

**The import-only store (S2).** The same history as a host's recorded messages (a SQLite snapshot in the shape
the history importer reads, one reviewed binding per session to the owner) through
``colonymind.turns.hermes_history.import_history``, run in the instance's own environment (``environment``):
sources with no claims and no model call. With an embedder configured, the sidecar then starts with claim
extraction off and indexes the sources' vectors.

One disposable directory per conversation (a LongMemEval question is its own conversation). The directory must
be new or empty. Its ``replay.json`` manifest maps every dataset turn to the turn id and message hash it was
stored under (``gold`` turns a question's evidence into those identifiers), and records every drain, clock move,
refused turn and the night. Nothing in it is a secret: the instance key stays in the instance's ``api.key``, a
model or endpoint credential only in this process's environment (``SECRET_VARIABLES``; ``Endpoints`` refuses one
among its settings).

A replay needs the process's wall clock: run one replay at a time per process.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import sys
import time
from typing import Any, Dict, Mapping, Optional

try:
    from colonymind.qualification import paired_body
except ImportError:  # a checkout without the package installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sidecar'))
    from colonymind.qualification import paired_body

PROTOCOL = 'public-memory-replay-1'
MANIFEST = 'replay.json'
INSTANCE = 'instance'           # configuration, key, identity and every store of the replayed instance
BEFORE_NIGHT = 'before-night'   # the instance copied before the night
IMPORT = 'import'               # the import-only store's snapshot and mapping
HERMES = 'hermes'               # an empty host home: the sidecar never reads a real one
HOME = 'home'                   # the user home while served: nothing is read from or written to a real one
NAMESPACE = 'replay'
# The owner's lane: the benchmark credential grants the instance key the owner on the CLI
# (``native_memory_worker.BENCHMARK_CREDENTIALS``), as ``colonymind init --owner-platform cli`` does.
CHANNEL = 'cli:replay'
HOST = {'host_id': 'replay'}
# The import-only store's binding platform: a direct conversation, never an automation lane.
IMPORT_PLATFORM = 'replay'
# The bounded drain before each session. The budget is a harness choice, generous for a session of about ten turns
# with several background model calls each; a drain that reaches it is recorded, never hidden. The idle rule and the
# poll are the paired track's (``paired_worker.DRAIN_IDLE_SECONDS``): a queue nothing works (a job held back, a lane the
# configured model cannot serve) is not waited for past it. Both are recorded in the manifest.
DRAIN_SECONDS, DRAIN_IDLE_SECONDS, DRAIN_POLL_SECONDS = 1800.0, 5.0, 0.25
# The all-on configuration: every faculty flag on but the feeling rules (``native_memory_worker.mind_section``).
ALL_ON = {'full': True, 'all_on': True}
# The benchmark endpoints' credentials are read from these process variables, never written anywhere.
SECRET_VARIABLES = {'COLONYMIND_EMBED_API_KEY': 'COLONYMIND_BENCH_EMBED_API_KEY',
                    'COLONYMIND_RERANKER_API_KEY': 'COLONYMIND_BENCH_RERANKER_API_KEY'}
# The settings ``Endpoints.settings`` may carry: the recall settings no ``router`` key covers, for a remote embedder
# and reranker. How the reranker is prompted and identified, its calibrated cutoff, the fingerprint that cutoff was
# fitted under and the tie margin, the recall deadlines, and how a query is embedded. Each is written into the
# instance's configuration and the manifest, so none is a credential; anything else is refused (the list is this
# harness's choice: a setting a campaign needs is added here, never passed through).
SETTINGS = frozenset({
    'COLONYMIND_RERANKER_PROMPT_STYLE', 'COLONYMIND_RERANKER_REVISION', 'COLONYMIND_RECALL_RERANK_MIN_SCORE',
    'COLONYMIND_RECALL_RERANK_CALIBRATION', 'COLONYMIND_RECALL_RERANK_TIE_MARGIN', 'COLONYMIND_RECALL_RERANK_TIMEOUT_MS',
    'COLONYMIND_RERANK_HOLD_LIMIT_MS', 'COLONYMIND_RECALL_INDEX_GENERATION', 'COLONYMIND_RECALL_EMBED_TIMEOUT_MS',
    'COLONYMIND_EMBED_QUERY_INSTRUCTION', 'COLONYMIND_EMBED_QUERY_MAX_TOKENS', 'COLONYMIND_EMBED_REQUEST_DIMS',
    'COLONYMIND_EMBED_REVISION'})


class ReplayError(RuntimeError):
    """A replay that cannot start or finish as specified."""


@dataclass(frozen=True)
class Endpoints:
    """The embedder and reranker a store is formed and read with (``colonymind.yaml``'s ``router`` keys), and the
    ``COLONYMIND_*`` settings recall reads that no key covers (``SETTINGS``: a reranker prompt style, the calibrated
    cutoff and its fingerprint, a query instruction). None of it is a secret: a credential in ``settings`` is
    refused, and credentials come only from ``SECRET_VARIABLES``. Without an embedder only lexical recall runs, and
    the manifest says so."""
    embed_url: str = ''
    embed_model: str = ''
    embed_dims: int = 0
    rerank_url: str = ''
    rerank_model: str = ''
    settings: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        from colonymind.config import looks_secret
        for name in map(str, self.settings):
            if name in SETTINGS:
                continue
            if name in SECRET_VARIABLES or name in SECRET_VARIABLES.values() or looks_secret(name):
                # Neither the value nor the name is repeated: a message can reach a log.
                raise ReplayError('Endpoints.settings holds a credential, which a replay never records: set it in '
                                  + ' or '.join(sorted(SECRET_VARIABLES.values())) + ' instead')
            raise ReplayError(f'Endpoints.settings: {name} is not a recall setting a replay records (SETTINGS)')

    def router(self) -> dict:
        return {'embed_url': self.embed_url, 'embed_model': self.embed_model, 'embed_dims': int(self.embed_dims or 0),
                'rerank_url': self.rerank_url, 'rerank_model': self.rerank_model}

    def public(self) -> dict:
        """What a manifest records: models and settings, never an endpoint's address."""
        return {'embedder': self.embed_model or None, 'embed_dims': int(self.embed_dims or 0) or None,
                'reranker': self.rerank_model or None, 'semantic': bool(self.embed_url),
                'settings': dict(sorted(self.settings.items()))}


# ------------------------------------------------------------------ the schedule

def units(session) -> list:
    """The session's turns as the host would send them: a user turn with the assistant turn after it, and a turn
    with no partner alone. Each unit is ``(user, assistant)``, either one None."""
    result, pending = [], None
    for turn in session.turns:
        if turn.role == 'user':
            if pending is not None:
                result.append((pending, None))
            pending = turn
        elif pending is not None:
            result.append((pending, turn))
            pending = None
        else:
            result.append((None, turn))
    if pending is not None:
        result.append((pending, None))
    return result


def schedule(conversation) -> list:
    """``[(session, start, end, units)]`` in time order: a session starts at its dataset time, or at the previous
    session's end when that is later; each unit takes one second, and ``end`` follows the last."""
    rows, end = [], None
    for session in conversation.sessions:
        parts = units(session)
        start = session.at.timestamp() if end is None else max(session.at.timestamp(), end)
        end = start + len(parts)
        rows.append((session, start, end, parts))
    return rows


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def question_time(questions) -> Optional[float]:
    """When the first of the conversation's questions is asked (never before any of its sessions: the dataset
    records refuse one that is)."""
    return min(question.asked_at.timestamp() for question in questions) if questions else None


# ------------------------------------------------------------------ the instance

@dataclass(frozen=True)
class Instance:
    root: Path
    home: Path
    owner: str

    @property
    def ledger(self) -> Path:
        return self.home / 'turn-idempotency.db'

    def key(self) -> str:
        return (self.home / 'api.key').read_text(encoding='utf-8').split()[0]


def _fresh(root: Path) -> Path:
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ReplayError(f'{root} is not empty: a replay starts in a new, empty directory')
    return root


def instance_config(owner: str, switches, endpoints: Endpoints, hermes: Path, context=None) -> dict:
    """The instance's ``colonymind.yaml``: the benchmark's owner, mind section and credential table
    (``native_memory_worker.instance_config``), the endpoints, and an empty host home."""
    from colonymind.qualification import native_memory_worker
    section = native_memory_worker.mind_section(switches)
    config = native_memory_worker.instance_config(owner, section, native_memory_worker.context_section(
        {**(switches if isinstance(switches, dict) else {}), **(context or {})}))
    config['hermes'] = {'home': str(hermes)}
    config['router'] = endpoints.router()
    if endpoints.settings:
        config['environment'] = {str(name): str(value) for name, value in endpoints.settings.items()}
    return config


def write_config(home: Path, config: dict) -> None:
    from colonymind.config import dump_config
    (home / 'colonymind.yaml').write_text(dump_config(config), encoding='utf-8')


async def prepare(root: Path, *, switches=None, endpoints: Optional[Endpoints] = None) -> Instance:
    """A new instance in ``root``: the owner contact (as ``colonymind init`` creates it), the instance key, the
    identity and the configuration. The clock must already be where the replay starts."""
    from colonymind.contacts.config import ContactsConfig
    from colonymind.contacts.store import SQLiteContactStore
    from colonymind.config import write_api_key
    from colonymind.setup import build_owner_contact
    root = Path(root).resolve()
    home, hermes = root / INSTANCE, root / HERMES
    home.mkdir(parents=True, mode=0o700)
    hermes.mkdir()
    (root / HOME).mkdir(mode=0o700)
    store = SQLiteContactStore(ContactsConfig(sqlite_path=str(home / 'colonymind-contacts.db')))
    await store.connect()
    try:
        owner = await build_owner_contact(store, 'Owner', [])
    finally:
        await store.close()
    write_api_key(secrets.token_urlsafe(32), home)
    (home / 'identity.yaml').write_text(json.dumps({'owner': {'name': 'Owner', 'contact_id': owner},
                                                    'agent': {'name': 'Agent'}}), encoding='utf-8')
    write_config(home, instance_config(owner, ALL_ON if switches is None else switches, endpoints or Endpoints(),
                                       hermes))
    return Instance(root=root, home=home, owner=owner)


def open_instance(root: Path, home: Optional[Path] = None) -> Instance:
    """An instance a replay made (or a copy of one)."""
    from colonymind.config import load_config
    root = Path(root).resolve()
    home = Path(home).resolve() if home is not None else root / INSTANCE
    owner = str(load_config(home, environ={}).get('owner.contact_id') or '')
    if not owner:
        raise ReplayError(f'{home} names no owner')
    return Instance(root=root, home=home, owner=owner)


@contextmanager
def environment(instance: Instance, overrides: Optional[Mapping[str, str]] = None):
    """The process environment the sidecar reads for ``instance``, as ``colonymind start`` derives it from the
    instance's configuration (``config.apply_environment``), and nothing else: every other ``COLONYMIND_`` or
    ``HERMES_`` variable is out of sight while it is served, and all of them are put back after. The user home is
    the replay's own (``HOME``), so nothing the sidecar keeps under ``~`` touches the operator's. The secrets
    settings the process caches (capture, the vault's key store) are read again from the instance inside, and the
    cached ones are put back after: an operator's never apply to a replayed store, nor a replay's to anything
    else in the process."""
    from colonymind.config import apply_environment, load_config
    from colonymind.secrets import config as secrets_config
    derived: Dict[str, str] = {}
    apply_environment(load_config(instance.home, environ={}), environ=derived)
    user_home = instance.root / HOME
    user_home.mkdir(mode=0o700, exist_ok=True)
    derived.update(COLONYMIND_SKIP_DOTENV='1', PYTHON_DOTENV_DISABLED='1', HOME=str(user_home),
                   COLONYMIND_EVENT_JOURNAL_DIR=str(instance.home / 'events'))
    for name, source in SECRET_VARIABLES.items():
        if os.environ.get(source):
            derived[name] = os.environ[source]
    derived.update({str(name): str(value) for name, value in (overrides or {}).items()})
    saved = {name: value for name, value in os.environ.items() if name.startswith(('COLONYMIND_', 'HERMES_'))
             or name in derived}
    cached_secrets = secrets_config._CURRENT
    for name in saved:
        os.environ.pop(name, None)
    os.environ.update(derived)
    secrets_config.set_settings(None)
    try:
        yield derived
    finally:
        for name in [name for name in os.environ if name.startswith(('COLONYMIND_', 'HERMES_')) or name in derived]:
            os.environ.pop(name, None)
        os.environ.update(saved)
        secrets_config.set_settings(cached_secrets)


def _mind():
    from colonymind.api.routers.mind import get_mind
    return get_mind()


def install_router(router) -> None:
    """The model router every consumer of the served sidecar reads: the host routes, the background worker (it
    reads the host's on every job) and the mind (its router, which it hands its faculties)."""
    from colonymind.api.routers import host
    host.set_llm_router(router)
    mind = _mind()
    if mind is not None:
        mind.router = router


def configured_router(llm_config: Mapping[str, Any]):
    """A router for the host model configuration ``llm_config`` (the object the sidecar persists as
    ``.colonymind-llm-config.json``), kept in memory: its credential is never written into the store."""
    from colonymind.router.router import LLMRouter
    router = LLMRouter(tiers={})
    router.configure(dict(llm_config))
    return router


@dataclass
class Served:
    instance: Instance
    app: Any
    client: Any
    environment: Dict[str, str]
    reranker: Any = None


@asynccontextmanager
async def serve(instance: Instance, *, router=None, llm_config=None, overrides=None, capture_reranker=False):
    """The instance's sidecar, served in process with its lifespan: the production app, its credential table,
    stores and background worker. The mind is served but its timer is not started. ``router`` (an object) or
    ``llm_config`` (a host model configuration) is the model; with neither, no model call can succeed.
    ``capture_reranker`` wraps the configured reranker in a ``ProviderCapture``. The lifespan installs its telemetry
    process-wide, stamped by the replay's clock; serving puts back what the process had, so nothing served later in
    the process reads the replay's record."""
    from colonymind.api.routers import host as host_routes
    telemetry = host_routes._telemetry
    try:
        async with _served(instance, router=router, llm_config=llm_config, overrides=overrides,
                           capture_reranker=capture_reranker) as served:
            yield served
    finally:
        host_routes.set_telemetry(telemetry)


@asynccontextmanager
async def _served(instance: Instance, *, router, llm_config, overrides, capture_reranker):
    import httpx
    from unittest.mock import patch
    with environment(instance, overrides) as derived:
        from colonymind.memory import blocks
        from colonymind.mind.tick import Mind
        from colonymind.server import create_app
        from colonymind.util import session_safety
        blocks.reset_settings()
        # The owner's last message time is kept under the user home, at a path fixed when the module loads.
        last_message = Path(derived['HOME']) / '.colonymind' / session_safety._LAST_MSG_PATH.name
        with patch.object(Mind, 'start', lambda self: None), \
                patch.object(session_safety, '_LAST_MSG_PATH', last_message):
            app = create_app()
            async with app.router.lifespan_context(app):
                if router is None and llm_config is not None:
                    router = configured_router(llm_config)
                if router is not None:
                    install_router(router)
                captured = None
                if capture_reranker:
                    from colonymind.api.routers import host
                    from colonymind.qualification.recall_capture import ProviderCapture
                    if host._reranker is not None:
                        captured = ProviderCapture(host._reranker)
                        host.set_reranker(captured)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url='http://replay', timeout=None,
                                             headers={'Authorization': 'Bearer ' + instance.key()}) as client:
                    try:
                        yield Served(instance=instance, app=app, client=client, environment=derived,
                                     reranker=captured)
                    finally:
                        if captured is not None:
                            from colonymind.api.routers import host
                            host.set_reranker(captured._provider)
        blocks.reset_settings()


async def drain(instance: Instance, *, seconds=DRAIN_SECONDS, idle=DRAIN_IDLE_SECONDS, poll=DRAIN_POLL_SECONDS):
    """Wait, never process, until the ledger's background queues owe nothing and run nothing, a queue sits idle
    (nothing running for ``idle`` seconds), or ``seconds`` pass: the paired worker's drain, on this event loop,
    which the served sidecar's worker shares. ``{status: drained|idle|budget|no_queue, waited_seconds, left}``."""
    from colonymind.qualification.paired_worker import background_backlog, unworked_queues
    skip = unworked_queues()
    started, idle_since = time.monotonic(), None
    while True:
        backlog, elapsed = background_backlog(instance.ledger), time.monotonic() - started
        if backlog is None:
            return {'status': 'no_queue', 'waited_seconds': 0.0, 'left': {}}
        backlog = {table: row for table, row in backlog.items() if table not in skip}
        owed, running = (sum(row[key] for row in backlog.values()) for key in ('owed', 'running'))
        idle_since = None if running else elapsed if idle_since is None else idle_since
        status = ('drained' if not owed and not running else 'idle' if idle_since is not None
                  and elapsed - idle_since >= idle else 'budget' if elapsed >= seconds else None)
        if status:
            return {'status': status, 'waited_seconds': round(elapsed, 3),
                    'left': {table: row for table, row in backlog.items() if any(row.values())},
                    **({'skipped': sorted(skip)} if skip else {})}
        await asyncio.sleep(poll)


# ------------------------------------------------------------------ the ledger, read back

def stored_source(instance: Instance, turn_id: str) -> Optional[dict]:
    """A stored source as the ledger keeps it: its messages (sealed and normalized), person, scope and time."""
    with closing(sqlite3.connect(instance.ledger.as_uri() + '?mode=ro', uri=True)) as conn:
        row = conn.execute('SELECT contact_id,session_id,scope,messages_json,occurred_at FROM turn_sources '
                           'WHERE turn_id=?', (turn_id,)).fetchone()
    if row is None:
        return None
    return {'contact_id': row[0], 'session_id': row[1], 'scope': row[2], 'messages': json.loads(row[3]),
            'occurred_at': row[4]}


def message_map(stored: Optional[dict], turns) -> list:
    """Each dataset turn of a unit and the hash of the stored message it became (by role, in order); None for a
    turn the source does not hold."""
    from colonymind.turns.idempotency import source_message_hash
    messages = list((stored or {}).get('messages') or [])
    result = []
    for turn in turns:
        index = next((position for position, message in enumerate(messages)
                      if isinstance(message, dict) and message.get('role') == turn.role), None)
        message = messages.pop(index) if index is not None else None
        result.append(source_message_hash(stored['session_id'], message) if message is not None else None)
    return result


def ledger_counts(instance: Instance) -> dict:
    counts = {}
    with closing(sqlite3.connect(instance.ledger.as_uri() + '?mode=ro', uri=True)) as conn:
        for name, sql in (('sources', 'SELECT count(*) FROM turn_sources'),
                          ('claims', 'SELECT count(*) FROM source_claims'),
                          ('claims_current', 'SELECT count(*) FROM source_claims WHERE superseded_by IS NULL '
                                             'AND retracted_by IS NULL')):
            try:
                counts[name] = int(conn.execute(sql).fetchone()[0])
            except sqlite3.Error:
                counts[name] = 0
        try:
            counts['claim_jobs'] = dict(conn.execute('SELECT status,count(*) FROM source_claim_jobs GROUP BY status'))
        except sqlite3.Error:
            counts['claim_jobs'] = {}
    return counts


def gold(manifest: Mapping[str, Any], question) -> list:
    """A question's evidence as ``scoring.reach`` gold: per evidence turn, the identifiers of the message it was
    stored as. The message hash names it; the turn id also does when its source holds that one message. A turn
    the store never held is a gold item nothing can cite."""
    turns = {(row['session'], row['index']): row for row in manifest['turns']}
    items = []
    for evidence in question.evidence:
        row = turns.get((evidence.session, evidence.turn))
        if row is None or not row.get('message_hash'):
            items.append({f'unstored:{evidence.session}:{evidence.turn}'})
            continue
        identifiers = {'hash:' + row['message_hash']}
        if row.get('alone'):
            identifiers.add('turn:' + row['turn_id'])
        items.append(identifiers)
    return items


def load_manifest(root: Path) -> dict:
    return json.loads((Path(root) / MANIFEST).read_text(encoding='utf-8'))


def _write_manifest(root: Path, manifest: dict) -> None:
    path = Path(root) / MANIFEST
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def _clock_row(event: str, **extra) -> dict:
    return {'event': event, 'clock': iso(time.time()), **extra}


def formation(manifest: Mapping[str, Any]) -> dict:
    """``{complete, problems}``: whether the store in ``manifest`` was formed as specified, and why not. A problem
    is a refused turn, a drain that ended on its budget (work still running when the harness moved on), or a night
    that did not answer, was skipped or had a call fail. A night that stopped at its own token budget is the
    product's night and not a problem (its ``exhausted`` and ``counts`` say so); neither is a drain that ended
    with a queue idle (the paired track's rule: work nothing runs is not waited for). The rule is this harness's."""
    problems = []
    refused = manifest.get('refused') or []
    if refused:
        problems.append(f'{len(refused)} turn(s) refused')
    for event in manifest.get('events') or []:
        if event.get('status') == 'budget':
            where = event.get('before_session')
            problems.append(f"{event['event']} {'before ' + str(where) if where else 'at the end'} ran out its "
                            'budget with work still running')
    night = manifest.get('night')
    if night is not None:
        if night.get('status') != 200:
            problems.append(f"night: the route answered {night.get('status')}")
        elif night.get('skipped'):
            problems.append(f"night: skipped ({night['skipped']})")
        if night.get('errors'):
            problems.append('night: ' + '; '.join(map(str, night['errors'])))
    return {'complete': not problems, 'problems': problems}


# ------------------------------------------------------------------ the formed store

def _body(instance: Instance, session_key: str, turn_id: str, occurred: float, user, assistant) -> dict:
    body = {'identity': dict(HOST), 'context': {
        'session_id': session_key, 'contact_id': instance.owner, 'turn_id': turn_id, 'channel_id': CHANNEL,
        'timezone': 'UTC', 'metadata': {'occurred_at': iso(occurred)}}}
    if user is not None:
        body['user_message'] = {'role': 'user', 'content': user.content}
    if assistant is not None:
        body['assistant_message'] = {'role': 'assistant', 'content': assistant.content}
    return body


async def replay(conversation, questions, root, *, router=None, llm_config=None, switches=None,
                 endpoints: Optional[Endpoints] = None, drain_seconds=DRAIN_SECONDS, drain_idle=DRAIN_IDLE_SECONDS,
                 drain_poll=DRAIN_POLL_SECONDS, dataset: Optional[Mapping[str, Any]] = None) -> dict:
    """Form ``conversation``'s store in ``root`` (new or empty), with one night before ``questions`` are asked;
    returns the manifest, also written to ``root/replay.json``."""
    root = _fresh(root)
    endpoints = endpoints or Endpoints()
    plan = schedule(conversation)
    if not plan:
        raise ReplayError(f'{conversation.id} has no sessions to replay')
    asked = question_time(questions)
    # The one move back: to the first session, before the instance or its key exists.
    paired_body.set_clock(plan[0][1], state=root)
    instance = await prepare(root, switches=switches, endpoints=endpoints)
    manifest = {'protocol': PROTOCOL, 'store': 'formed', 'conversation': conversation.id,
                'dataset': dict(dataset or {}), 'questions': [question.id for question in questions],
                'switches': ALL_ON if switches is None else switches, 'endpoints': endpoints.public(),
                'drain': {'seconds': drain_seconds, 'idle_seconds': drain_idle, 'poll_seconds': drain_poll},
                'events': [_clock_row('start')], 'sessions': [], 'turns': [], 'refused': []}
    async with serve(instance, router=router, llm_config=llm_config) as served:
        for position, (session, start, end, parts) in enumerate(plan):
            manifest['events'].append(_clock_row('drain', before_session=session.key, **await drain(
                instance, seconds=drain_seconds, idle=drain_idle, poll=drain_poll)))
            paired_body.clock_at_least(end, state=root)
            manifest['events'].append(_clock_row('session', session=session.key))
            manifest['sessions'].append({'key': session.key, 'at': session.at.isoformat(), 'start': iso(start),
                                         'end': iso(end), 'shift_seconds': start - session.at.timestamp(),
                                         'units': len(parts)})
            for k, (user, assistant) in enumerate(parts):
                turn_id, occurred = f'replay:{position}:{k}', start + k
                if occurred > time.time():
                    raise ReplayError('A replayed turn would be later than the clock')
                response = await served.client.post('/v1/host/turns/sync', json=_body(
                    instance, session.key, turn_id, occurred, user, assistant))
                result = response.json() if response.headers.get('content-type', '').startswith(
                    'application/json') else {}
                stored = stored_source(instance, turn_id)
                owned = stored is not None and stored['contact_id'] == instance.owner
                if response.status_code >= 400 or not result.get('accepted') or not owned:
                    manifest['refused'].append({'turn_id': turn_id, 'status': response.status_code,
                                                'skipped_reason': result.get('skipped_reason'),
                                                'stored_as_owner': owned})
                turns = [turn for turn in (user, assistant) if turn is not None]
                hashes = message_map(stored if owned else None, turns)
                for turn, digest in zip(turns, hashes):
                    manifest['turns'].append({'session': session.key, 'index': turn.index, 'role': turn.role,
                                              'turn_id': turn_id, 'message_hash': digest, 'alone': len(turns) == 1,
                                              'occurred_at': iso(occurred)})
        manifest['events'].append(_clock_row('drain', before_session=None, **await drain(
            instance, seconds=drain_seconds, idle=drain_idle, poll=drain_poll)))
    # The store without the night: the instance copied while no sidecar holds it.
    shutil.copytree(instance.home, root / BEFORE_NIGHT / INSTANCE, symlinks=True)
    manifest['before_night'] = {'counts': ledger_counts(instance), 'clock': iso(time.time())}
    night_at = paired_body.clock_at_least(asked if asked is not None else time.time(), state=root)
    async with serve(instance, router=router, llm_config=llm_config) as served:
        response = await served.client.post('/v1/mind/consolidate')
        night = response.json() if response.status_code == 200 else {}
        manifest['night'] = {'status': response.status_code, 'at': iso(night_at),
                             'shift_seconds': night_at - asked if asked is not None else None,
                             **{key: night.get(key) for key in ('done', 'skipped', 'counts', 'calls', 'local_date',
                                                                'errors', 'exhausted', 'rejected_lines')}}
    manifest['counts'] = ledger_counts(instance)
    manifest['events'].append(_clock_row('end'))
    manifest['formation'] = formation(manifest)
    _write_manifest(root, manifest)
    return manifest


# ------------------------------------------------------------------ the import-only store

def host_snapshot(conversation, path: Path, plan) -> list:
    """The history as a host records it, in the shape the history importer reads: one session row per session and
    one message row per turn, stamped with the schedule's times. Returns the importer's session bindings."""
    bindings = []
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, user_id TEXT, chat_id TEXT, '
                     'chat_type TEXT, model TEXT)')
        conn.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, '
                     'timestamp REAL)')
        for session, start, _end, parts in plan:
            conn.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?)',
                         (session.key, IMPORT_PLATFORM, 'owner', session.key, 'dm', None))
            for k, pair in enumerate(parts):
                for turn in pair:
                    if turn is not None:
                        conn.execute('INSERT INTO messages(session_id,role,content,timestamp) VALUES (?,?,?,?)',
                                     (session.key, turn.role, turn.content, start + k))
            bindings.append({'session_id': session.key, 'platform': IMPORT_PLATFORM, 'actor_id': 'owner',
                             'chat_id': session.key, 'chat_type': 'dm',
                             'review_evidence': 'a public dataset replayed as one owner\'s own conversations'})
    return bindings


async def import_only(conversation, questions, root, *, endpoints: Optional[Endpoints] = None,
                      drain_seconds=DRAIN_SECONDS, drain_idle=DRAIN_IDLE_SECONDS, drain_poll=DRAIN_POLL_SECONDS,
                      dataset: Optional[Mapping[str, Any]] = None) -> dict:
    """The import-only store of ``conversation`` in ``root`` (new or empty): no claims and no model call."""
    from colonymind.turns.hermes_history import import_history
    root = _fresh(root)
    endpoints = endpoints or Endpoints()
    plan = schedule(conversation)
    if not plan:
        raise ReplayError(f'{conversation.id} has no sessions to import')
    # An import happens once the history it imports is over: the clock goes back to the last session's end.
    paired_body.set_clock(plan[-1][2], state=root)
    instance = await prepare(root, switches=ALL_ON, endpoints=endpoints)
    (root / IMPORT).mkdir()
    snapshot = root / IMPORT / 'history.db'
    bindings = host_snapshot(conversation, snapshot, plan)
    for binding in bindings:
        binding['contact_id'] = instance.owner
    mapping = {'version': 1, 'namespace': NAMESPACE, 'reviewed': True, 'bindings': bindings}
    (root / IMPORT / 'mapping.json').write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    runs = []
    # The importer seals what capture detects under the instance's own settings and key, never the operator's.
    with environment(instance):
        while True:
            result = import_history(snapshot, mapping, state_dir=instance.home, apply=True, limit=10000)
            runs.append({key: result[key] for key in ('counts', 'cursor', 'remaining', 'complete', 'models_called')})
            if result['complete']:
                break
    manifest = {'protocol': PROTOCOL, 'store': 'import_only', 'conversation': conversation.id,
                'dataset': dict(dataset or {}), 'questions': [question.id for question in questions],
                'endpoints': endpoints.public(), 'import': runs, 'turns': [], 'events': [_clock_row('import')]}
    with closing(sqlite3.connect(snapshot.as_uri() + '?mode=ro', uri=True)) as conn:
        rows = {(session, role, content, timestamp): identifier for identifier, session, role, content, timestamp
                in conn.execute('SELECT id,session_id,role,content,timestamp FROM messages ORDER BY id')}
    for session, start, _end, parts in plan:
        for k, pair in enumerate(parts):
            for turn in pair:
                if turn is None:
                    continue
                identifier = rows[(session.key, turn.role, turn.content, start + k)]
                turn_id = f'hermes-history:{NAMESPACE}:{identifier}'
                stored = stored_source(instance, turn_id)
                owned = stored is not None and stored['contact_id'] == instance.owner
                digest, = message_map(stored if owned else None, [turn])
                manifest['turns'].append({'session': session.key, 'index': turn.index, 'role': turn.role,
                                          'turn_id': turn_id, 'message_hash': digest, 'alone': True,
                                          'occurred_at': iso(start + k)})
    if endpoints.embed_url:
        # The sources' vectors: the sidecar with claim extraction off calls no model, only the embedder.
        async with serve(instance, overrides={'COLONYMIND_SOURCE_CLAIMS': 'off'}):
            manifest['events'].append(_clock_row('vectors', **await drain(
                instance, seconds=drain_seconds, idle=drain_idle, poll=drain_poll)))
    manifest['counts'] = ledger_counts(instance)
    manifest['formation'] = formation(manifest)
    _write_manifest(root, manifest)
    return manifest


__all__ = ['ALL_ON', 'BEFORE_NIGHT', 'Endpoints', 'INSTANCE', 'MANIFEST', 'PROTOCOL', 'ReplayError', 'SETTINGS',
           'drain', 'formation', 'gold', 'import_only', 'load_manifest', 'open_instance', 'replay', 'schedule',
           'serve', 'units']
