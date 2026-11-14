"""Isolated native memory observation over real plugin/provider and host routes."""
from contextlib import ExitStack, contextmanager
import json
import hashlib
import os
from pathlib import Path
import secrets
import socket
import threading
import time
from unittest.mock import patch


MIND_FACULTIES = ('initiative', 'drives', 'deliberation', 'goals', 'people', 'affect', 'affect_rules', 'opinions',
                  'broadcast', 'semantic_recall', 'consolidation', 'self_narrative', 'lessons', 'skills', 'outreach',
                  'owner_gate', 'drift_check', 'senses', 'plasticity', 'working_memory', 'learning', 'wishes',
                  'wishes_named', 'hands', 'work_records')
WORKER_PROFILE = 'colonymind-act'
# The one flag ``all_on`` leaves off: the feeling rules replace the running feelings wherever feelings are read,
# whatever ``affect`` says (mind/affect.py), so "every faculty on" runs the feelings, and the rules are measured
# in their place (``plus_affect_rules``, the all-on-plus-affect_rules arm).
ALL_ON_EXCEPT = frozenset({'affect_rules'})
# The two uses of the embedder beyond recall, each the sidecar's own faculty (``config.MEANING_FACULTIES``): the
# night's contradiction candidates found by comparing claim keys' embeddings (``contradictions_by_meaning``, read
# from the served mind's section) and the vector lane of the prior records the claim worker offers the extractor
# (``priors_by_meaning``, exported as ``COLONYMIND_PRIORS_BY_MEANING``, ``arm_environment``). They ship on and a new
# colonymind.yaml does not name them, so a full arm writes their flags into its mind section itself (on, unless its
# ``minus_`` switch is on), and the sidecar reads each where it reads it on an install.
MEANING_FACULTIES = ('contradictions_by_meaning', 'priors_by_meaning')


def mind_section(switches, quiet_hours=None):
    """The ``mind`` section of the disposable ``colonymind.yaml`` for a profile's mind switches.

    Off (``None`` or ``False``) in the plain plugin arm. ``{'initiative': True}``
    (or ``True``) turns the mind on at autonomy ``standard`` with only the
    initiative faculty (mind-initiative-1); semantic recall, memory the plain
    plugin arm has too, stays on. ``{'full': True}`` sets every
    faculty flag and drive weight to its release-candidate value from the
    shipped defaults; ``all_on`` with it turns every faculty flag on (the
    measurement policy: every faculty on unless it has shown harm) except
    the feeling rules, which would replace the running feelings; a
    ``minus_<faculty>`` switch turns that faculty's flag
    off, a ``plus_<faculty>`` switch turns on one that ships off
    (``plus_skills``, ``plus_affect_rules``, ``plus_owner_gate``, ``plus_learning``, ``plus_wishes``,
    ``plus_drift_check``), and a
    ``minus_<drive>`` switch sets that drive's weight to 0 (evals section 3,
    the ``full-X`` and ``all-on-X`` arms). The flag is written whether or not the
    faculty's code has landed, so an ablation of a faculty nothing reads yet
    is a no-op contrast until its milestone; the two faculties the shipped
    defaults do not name (``MEANING_FACULTIES``) are written on unless
    ablated. Quiet hours and the daily digest
    are off in every mind arm because an episode's clock advance would
    otherwise hold or add owner notices that have nothing to do with the
    scenario; a family that declares quiet hours (``quiet_hours``, the owner
    outreach family) has that window written instead, the one its base arms
    read from the workspace.
    """
    if not switches:
        return {'enabled': False}
    if switches is True:
        switches = {'initiative': True}
    section = {'enabled': True, 'autonomy': 'standard', 'quiet_hours': quiet_hours or '', 'digest_hour': 24}
    if not switches.get('full'):
        section['faculties'] = {name: name in {'initiative', 'semantic_recall'} for name in MIND_FACULTIES}
        return section
    from copy import deepcopy
    from colonymind.config import DEFAULTS
    defaults = DEFAULTS['mind']
    faculties = {name: bool((switches.get('all_on') and name not in ALL_ON_EXCEPT)
                            or defaults['faculties'].get(name, False))
                 for name in MIND_FACULTIES}
    drives = deepcopy(defaults['drives'])
    for name in MIND_FACULTIES:
        if switches.get(f'minus_{name}'):
            faculties[name] = False
        if switches.get(f'plus_{name}'):
            faculties[name] = True
    for name in MEANING_FACULTIES:
        faculties[name] = not switches.get(f'minus_{name}')
    for name in drives:
        if switches.get(f'minus_{name}'):
            drives[name] = 0.0
    section.update(faculties=faculties, drives=drives, budgets=deepcopy(defaults['budgets']))
    return section


def context_section(switches):
    """Only status can be switched in a disposable instance; context fitting is mandatory."""
    from colonymind.config import DEFAULTS
    switches = switches if isinstance(switches, dict) else {}
    shipped = bool(DEFAULTS['context']['status'])
    wanted = True if switches.get('all_on') else shipped
    if switches.get('minus_status'):
        wanted = False
    return {'status': wanted} if wanted != shipped else None


def arm_environment(section, context):
    """The switches the served host reads from its environment, derived the way ``config.apply_environment``
    derives them on an install: the status block from ``context.status`` (``COLONYMIND_CONTEXT_STATUS``, read at
    every assemble), and the two switches of the claim projection, which runs outside the mind, in every plugin
    arm: the learning rules from ``mind.faculties.learning`` (``COLONYMIND_LEARNING``: without it an arm's lessons
    would follow the rules and its claims would not) and the vector lane of its prior records from
    ``mind.faculties.priors_by_meaning`` (``COLONYMIND_PRIORS_BY_MEANING``, on unless the section sets it off). Each
    is the arm's flag, and with the mind off the shipped default, as an install exports it. ``context`` is the arm's
    context section (``context_section``). An arm's overlay is applied after these."""
    from colonymind.config import DEFAULTS
    faculties = section.get('faculties') if isinstance(section.get('faculties'), dict) else {}
    status = (context or {}).get('status', DEFAULTS['context']['status'])
    # A section that names no learning flag (the mind off) leaves it at the shipped default, as an install does.
    learning = faculties.get('learning', DEFAULTS['mind']['faculties']['learning'])
    return {'COLONYMIND_CONTEXT_STATUS': 'on' if status else 'off',
            'COLONYMIND_LEARNING': '1' if learning is True else '0',
            'COLONYMIND_PRIORS_BY_MEANING': '0' if faculties.get('priors_by_meaning') is False else '1'}


# The disposable file must be one ``load_config`` accepts: its readers in the served host (the block fit, the
# secrets settings, the projection backlog) fall back to their defaults when it refuses the file, so an arm's
# context section would be ignored without a word. A mind arm's section keeps the daily digest off with an hour
# that never comes (24); the file's hours are 0..23, so the file names the last one. The served Mind is built
# from the section itself (``serve_mind``), and nothing in the benchmark reads the hour from the file.
FILE_DIGEST_HOUR = 23


def instance_config(person, section, context=None, *, plasticity=None, profile=None):
    """The disposable ``colonymind.yaml``: the owner, the arm's mind section, the benchmark's credential table
    and, where the arm changes it, its context section (``context_section``); a configuration the product
    accepts (``FILE_DIGEST_HOUR``)."""
    mind = dict(section)
    if type(mind.get('digest_hour')) is int and mind['digest_hour'] > FILE_DIGEST_HOUR:
        mind['digest_hour'] = FILE_DIGEST_HOUR
    config = {'owner': {'contact_id': person}, 'mind': mind, 'credentials': BENCHMARK_CREDENTIALS}
    if context is not None:
        config['context'] = dict(context)
    from copy import deepcopy
    from .paired_arms import apply_plasticity_arm, plasticity_arm
    if plasticity is not None:
        config['plasticity'] = deepcopy(plasticity)
    arm = plasticity_arm(profile or {}, (plasticity or {}).get('learners', {}))
    if arm is not None:
        config = apply_plasticity_arm(config, arm)
    if (profile or {}).get('wm_summary_only'):
        config['working_memory'] = {'summary_only': True}
    return config


def embedding_environment(inputs, section):
    """The embedding provider for one arm, as a binary choice recorded in the plan.

    The plan may carry one embedding endpoint (``paired plan --embedding-config``), written
    identically into every case's ``inputs['embedding']``. An arm uses it unless its mind
    section turns ``faculties.semantic_recall`` off, the rule ``config.apply_environment``
    applies on a real install; only the ``full-semantic_recall`` arm does, and the plain plugin
    arm has no faculties and uses it too. With no endpoint in the plan every arm keeps today's
    behaviour: the embedder off (``skip``). A named credential is read from this process's
    environment (the plan admits only names the native config already forwards). The served
    host opens the embedder from these variables (``paired_worker.semantic_recall``).
    """
    embedding = inputs.get('embedding')
    faculties = section.get('faculties') if isinstance(section.get('faculties'), dict) else {}
    semantic = isinstance(embedding, dict) and bool(embedding) and faculties.get('semantic_recall', True) is not False
    if not semantic:
        return {'COLONYMIND_EMBED_PROVIDER': 'skip'}
    values = {'COLONYMIND_EMBED_PROVIDER': 'openai_api', 'COLONYMIND_EMBED_BASE_URL': str(embedding['base_url']),
              'COLONYMIND_EMBED_MODEL': str(embedding['model']), 'COLONYMIND_EMBED_DIMS': str(int(embedding['dimensions']))}
    if embedding.get('api_key_env'):
        values['COLONYMIND_EMBED_API_KEY'] = os.environ[str(embedding['api_key_env'])]
    return values


ASSEMBLE_ROUTE = '/v1/host/context/assemble'
# The most messages one section's record is kept naming (my choice: a turn's packet shows tens of records, so only a
# pathological section reaches it; ``messages_truncated`` then says the list is incomplete).
SHOWN_MESSAGES = 512
_SOURCE_KEYS = ('source_id', 'source_turn_id', 'source')


def shown_messages(body):
    """``([[source_id, message_hash]], truncated)``: the messages a section body's records name by hash
    (``source_message_hash``, the ledger's hash of one message), each with the source the record says holds it
    (its ``source_id``, ``source_turn_id`` or ``source``, else the enclosing record's). Read from the structured
    records only (each ``- `` line's JSON), never from text: a citation names a whole source, and a source may hold
    many messages, so this is what says which of them the section showed. Bounded by ``SHOWN_MESSAGES``."""
    found = []

    def walk(value, source):
        if isinstance(value, dict):
            named = next((value[key] for key in _SOURCE_KEYS if isinstance(value.get(key), str) and value[key]), None)
            source = named.removeprefix('turn:') if named is not None else source
            digest = value.get('source_message_hash')
            if isinstance(digest, str) and digest:
                found.append((source, digest))
            for item in value.values():
                walk(item, source)
        elif isinstance(value, list):
            for item in value:
                walk(item, source)
    decoder = json.JSONDecoder()
    for line in str(body or '').split('\n'):
        stripped = line.strip()
        if not stripped.startswith('- '):
            continue
        try:
            record, _ = decoder.raw_decode(stripped[2:])
        except ValueError:
            continue
        walk(record, '')
    unique = [[source, digest] for source, digest in dict.fromkeys(found)]
    return unique[:SHOWN_MESSAGES], len(unique) > SHOWN_MESSAGES


def cited_sources(body):
    """``[{section, refs: [{source_id, source_version}], body_sha256, body_chars, messages}]``: the sources each
    section of an assembled context cited, identifiers only, with the SHA-256 and length of the section's body as
    returned, so the re-score can tell whether that body reached a model request (a provider drops an answer that
    comes too late, and may hold a section back), and the messages its records name (``shown_messages``;
    ``messages_truncated`` when the list was cut); None for a body that is not an assembled context."""
    try:
        sections = json.loads(body).get('sections')
    except (ValueError, AttributeError):
        return None
    if not isinstance(sections, list):
        return None
    cited = []
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get('citations'), list):
            continue
        refs = [{key: ref[key] for key in ('source_id', 'source_version') if isinstance(ref.get(key), str)}
                for ref in section['citations'] if isinstance(ref, dict) and isinstance(ref.get('source_id'), str)]
        if refs:
            entry = {'section': str(section.get('id')), 'refs': refs}
            if isinstance(section.get('body'), str):
                messages, truncated = shown_messages(section['body'])
                entry.update(body_sha256=hashlib.sha256(section['body'].encode('utf-8')).hexdigest(),
                             body_chars=len(section['body']), messages=messages)
                if truncated:
                    entry['messages_truncated'] = True
            cited.append(entry)
    return cited


def route_recorder(routes, diagnostic=None, cited=None):
    """The served host's route middleware: each route and status into ``routes`` (the episode's evidence) and a
    ``context_route`` event into the private trace, which for an assembled context also carries the sources each
    section cited (``cited_sources``), so whether a probe's evidence reached the model can be scored from what the
    host injected; each cited source id also goes into ``cited`` (a dict kept in the order last cited). The
    response reaches the caller byte for byte."""
    async def record_route(req, next_call):
        response = await next_call(req)
        routes.append({'path': req.url.path, 'status': response.status_code})
        if diagnostic is None and cited is None:
            return response
        event = {'method': req.method, 'path': req.url.path, 'status': response.status_code}
        if req.url.path == ASSEMBLE_ROUTE and response.status_code == 200:
            from starlette.responses import Response
            body = b''.join([chunk async for chunk in response.body_iterator])
            response = Response(content=body, status_code=response.status_code, headers=dict(response.headers),
                                background=response.background)
            sections = cited_sources(body)
            if sections is not None:
                event['source_refs'] = sections
                for section in sections if cited is not None else ():
                    for ref in section['refs']:
                        cited.pop(ref['source_id'], None)
                        cited[ref['source_id']] = True
        if diagnostic is not None:
            diagnostic.record('context_route', event)
        return response
    return record_route


def rerank_environment(inputs):
    """The reranker for one arm: the plan's one reranker (``paired plan --rerank-config``, written identically
    into every case's ``inputs['reranker']``) in the variables an install exports for ``router.rerank_url``, with
    its calibrated cutoff and the fingerprint that cutoff was calibrated under, or nothing without one. The
    embedding model and width are exported with it whether or not the arm embeds, as an install exports them
    whenever an embedding endpoint is configured: they are part of the fingerprint. The index generation is pinned
    to its unverified default, since every episode builds a new index. A named credential is read from this
    process's environment. The served host opens the reranker from these (``paired_worker.served_reranker``)."""
    rerank = inputs.get('reranker')
    if not isinstance(rerank, dict) or not rerank:
        return {}
    values = {'COLONYMIND_RECALL_RERANK': 'on', 'COLONYMIND_RERANKER_PROVIDER': 'openai_api',
              'COLONYMIND_RERANKER_BASE_URL': str(rerank['base_url']), 'COLONYMIND_RERANKER_MODEL': str(rerank['model']),
              'COLONYMIND_RERANKER_PROMPT_STYLE': str(rerank.get('prompt_style') or ''),
              'COLONYMIND_RERANKER_REVISION': str(rerank.get('revision') or 'unverified'),
              'COLONYMIND_RECALL_INDEX_GENERATION': 'unverified',
              'COLONYMIND_RECALL_RERANK_MIN_SCORE': str(rerank['cutoff']),
              'COLONYMIND_RECALL_RERANK_TIMEOUT_MS': str(int(rerank.get('timeout_ms') or 1200)),
              'COLONYMIND_RECALL_RERANK_CALIBRATION': str(rerank.get('fingerprint') or '')}
    embedding = inputs.get('embedding')
    if isinstance(embedding, dict) and embedding:
        values.update(COLONYMIND_EMBED_MODEL=str(embedding['model']),
                      COLONYMIND_EMBED_DIMS=str(int(embedding['dimensions'])))
    if rerank.get('api_key_env'):
        values['COLONYMIND_RERANKER_API_KEY'] = os.environ[str(rerank['api_key_env'])]
    return values


def mind_clock():
    """The body clock: ``time.time`` as the paired body shifts it, as an aware UTC datetime."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(time.time(), timezone.utc)


def benchmark_identity(person):
    """The disposable ``identity.yaml``: the owner, and the one handle a mind message can reach.

    The body resolves an owner message through ``owner.handles`` first, so
    without one no mind message goes out: the body finds no target and the
    sidecar leaves the message waiting unclaimed. The handle is the capture
    platform's home channel, where the heartbeat arm already delivers.
    """
    from colonymind.qualification.paired_body import OWNER, PLUGIN
    return {'owner': {'name': 'Owner', 'contact_id': person, 'handles': {PLUGIN: [OWNER]}},
            'agent': {'name': 'Agent'}}


# The benchmark's owner speaks on Hermes' CLI lane (``paired_worker.turn_message``): the arm's instance key
# is granted the owner there, as ``colonymind init --owner-platform cli`` grants it on an install.
BENCHMARK_CREDENTIALS = {'api-key': {'owner': ['cli'], 'contacts': 'all'}}


def benchmark_credentials():
    """The credential table an arm's sidecar serves (docs/CREDENTIALS.md)."""
    from colonymind.api.credentials import CredentialTable
    return CredentialTable.from_config(BENCHMARK_CREDENTIALS)


def mount_routes(app, *, mind):
    """The routes the arm serves, as the sidecar mounts them: the host routes, and with the mind
    its own routes and the people routes (an owner's ``colonymind_people`` reaches them)."""
    from colonymind.api.routers import branches, host
    from colonymind.mind.factory import mind_routers
    app.include_router(host.router)
    app.include_router(host.v2_router)
    app.include_router(branches.router)
    if mind:
        for router in mind_routers():   # the list the sidecar mounts (colonymind.mind.factory)
            app.include_router(router)


@contextmanager
def working_memory_store(state):
    """Keep branches beside host turns, using the arm's separately stored configuration."""
    from dataclasses import replace
    from colonymind.api.routers import branches
    from colonymind.config import load_config
    from colonymind.memory.branches import get_store
    config = replace(load_config(state / 'colonymind-instance'), home=state / 'memory-state')
    store = get_store(config)
    with patch.object(branches, '_store', lambda: store):
        yield store


@contextmanager
def plasticity_nights(mind, home, directory):
    """Run configured learners at the mind's nightly boundary on the episode's own body clock."""
    from colonymind.config import load_config
    from colonymind.plasticity.nightly import NightlyTask, Runtime
    runtime = Runtime.from_config(load_config(home), data_dir=directory, clock=mind_clock)
    try:
        if not runtime.enabled or not runtime.loaded:
            yield runtime
            return
        nightly = NightlyTask(runtime, tz=mind.tz, boundary_minute=mind.consolidation.boundary_minute())
        original_tick = mind.tick

        async def tick(*args, **kwargs):
            result = await original_tick(*args, **kwargs)
            # Complete the due night before the benchmark advances to its next body event.
            nightly.tick(mind.clock())
            return result

        with patch.object(mind, 'tick', tick):
            yield runtime
    finally:
        runtime.close()


@contextmanager
def serve_mind(app, state, person, section):
    """A real Mind over this arm's stores, on ``/v1/mind`` next to the host routes.

    The stores the host routes already own (the commitment store, contacts and
    the ledger) are shared; the intention, feedback and expectation stores live
    in the same ``memory-state`` directory, and the owner's appraisal records
    and reported outcomes are read from the arm's ledger, as production reads
    them. The Mind's own timer is never started: the body tick calls
    ``POST /v1/mind/tick`` through the plugin's ``tick()``, so intentions form
    in lockstep with the episode's clock.
    """
    from colonymind.api.routers import host
    from colonymind.api.routers import mind as mind_router
    from colonymind.feedback import TypeFeedbackStore
    from colonymind.initiatives.store import InitiativeStore
    from colonymind.mind.factory import build_mind
    from colonymind.self_model.expectations import ExpectationEngine, ExpectationStore
    from colonymind.turns import get_turn_idempotency_ledger
    directory = state / 'memory-state'
    directory.mkdir(parents=True, exist_ok=True)
    store = InitiativeStore(state_dir=directory)
    try:
        # The sidecar's own factory over the stores the host routes own (the commitment store, contacts,
        # the comms ledger, contact affect and the arm's router, the endpoint the plan pinned); only the
        # body clock and the backups are this process's own choice.
        mind = build_mind(host, config=section, store=store, state_dir=directory,
                          ledger=get_turn_idempotency_ledger(directory), owner_id=person,
                          feedback=TypeFeedbackStore(db_path=str(directory / 'colonymind-feedback.db')),
                          expectations=ExpectationEngine(ExpectationStore(str(directory / 'colonymind-expectations.db'))),
                          clock=mind_clock, backups=False)
        with plasticity_nights(mind, state / 'colonymind-instance', directory) as runtime:
            app.state.plasticity = runtime
            mind_router.set_mind(mind)
            yield mind
    finally:
        mind_router.set_mind(None)
        store.close()


def install_worker_profile(home):
    """The ``colonymind-act`` profile directory the kanban dispatcher spawns mind tasks on.

    The in-process worker runs the task with the episode's own recipe; the
    profile only has to exist for ``dispatch_once`` to consider the task
    spawnable, as ``colonymind init`` makes it exist on a real install.
    """
    directory = Path(home) / 'profiles' / WORKER_PROFILE
    for name in ('memories', 'sessions', 'skills', 'logs', 'workspace'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    config_path = directory / 'config.yaml'
    if not config_path.exists():
        config_path.write_text(json.dumps({'plugins': {'enabled': ['colonymind'], 'hook_callback_timeout': 0}}))
    return directory


@contextmanager
def prepare(request, state, arguments, config, *, setup_host=None, scopes=None, overlay=None, mind=None,
            context=None, cited=None):
    """Enable only this fixture's private profile and ledger, before agent construction.

    An arm profile overlay is applied after the forced flags below and before
    the plugin loads, so a frozen profile can flip any fixture default. With
    ``mind`` (the profile's mind switches, see ``mind_section``) the arm also
    serves ``/v1/mind`` over a real Mind. ``context`` (the profile's context
    switches) and ``all_on`` change the served host's per-turn context
    (``context_section``). ``cited`` (a dict) receives the source ids the
    served host's assemblies cite, in the order last cited (``route_recorder``).
    """
    inputs = request['inputs']
    person = inputs['contact_id']
    section = mind_section(mind, quiet_hours=inputs.get('quiet_hours'))
    context_value = context_section({**(mind if isinstance(mind, dict) else {}), **(context or {})})
    instance_value = instance_config(person, section, context_value, plasticity=inputs.get('plasticity'),
                                     profile=inputs.get('profile'))
    embedding = embedding_environment(inputs, section)
    os.environ.update(COLONYMIND_STATE_DIR=str(state / 'memory-state'),
        COLONYMIND_EVENT_JOURNAL_DIR=str(state / 'memory-state' / 'events'),
        COLONYMIND_SKIP_DOTENV='1', PYTHON_DOTENV_DISABLED='1',
        HERMES_BUNDLED_PLUGINS=str(state / 'empty-bundled'), HERMES_ENABLE_PROJECT_PLUGINS='0',
        HERMES_DISABLE_TELEMETRY='1', HERMES_DISABLE_LAZY_INSTALLS='1',
        COLONYMIND_GENERAL_PLUGIN_ACTIVE='1', COLONYMIND_MEMORY_TURN_WRITER='disabled',
        COLONYMIND_MEMORY_WORKER_TOOLS='0', COLONYMIND_MEMORY_DEFAULT_CONTEXT_AUTHORITY='none',
        COLONYMIND_OWNER_CONTACT_ID=person,
        **embedding, **rerank_environment(inputs), **arm_environment(section, context_value),
        # The body tick drives the adapter (tick() and flush()); its own thread stays parked
        # so no dispatch or send lands between two observed ticks.
        COLONYMIND_BODY_THREAD='0')
    os.environ.update(overlay or {})
    (state / 'empty-bundled').mkdir(exist_ok=True)
    from fastapi import FastAPI
    import uvicorn
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import host
    from colonymind.api.routers import mind as mind_router
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.vector import get_store
    from hermes_cli.plugins import get_plugin_manager
    from hermes_state import SessionDB
    import httpx

    secret = secrets.token_urlsafe(32)
    diagnostic = request.get('_diagnostic_recorder')
    if diagnostic is not None:
        diagnostic.add_secret(secret)
    # The disposable instance directory the adapter reads: the one key, the
    # owner identity and a stock colonymind.yaml (the arm's mind section and,
    # where the arm changes it, its context section). The same secret guards
    # the fixture API below.
    instance = state / 'colonymind-instance'
    instance.mkdir(mode=0o700, exist_ok=True)
    key_file = instance / 'api.key'
    key_file.write_text(secret + '\n')
    key_file.chmod(0o600)
    (instance / 'identity.yaml').write_text(json.dumps(benchmark_identity(person)))
    (instance / 'colonymind.yaml').write_text(json.dumps(instance_value))
    os.environ.update(COLONYMIND_HOME=str(instance), COLONYMIND_API_KEY=secret)
    from colonymind.memory import blocks
    blocks.reset_settings()   # the host's block fit reads this instance's context section, never a cached one
    if mind:
        install_worker_profile(state)
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=secret, credentials=benchmark_credentials())
    routes = []

    app.middleware('http')(route_recorder(routes, diagnostic, cited))

    mount_routes(app, mind=bool(mind))
    host_resources = ExitStack()
    listener = server = thread = None
    transport_observer = None
    try:
        host_resources.enter_context(working_memory_store(state))
        if setup_host is not None:
            host_resources.enter_context(setup_host(app, state, inputs, config))
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(64)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port,
            lifespan='on', access_log=False, log_level='error'))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        thread.start()
        started = time.monotonic()
        while not server.started:
            if not thread.is_alive() or time.monotonic() - started > 5:
                raise RuntimeError('Isolated memory API did not start')
            time.sleep(.01)
        url = f'http://127.0.0.1:{port}'
        if mind:
            # The host stores exist once the server's lifespan ran; the Mind shares them.
            host_resources.enter_context(serve_mind(app, state, person, section))
        already = [name for name in config.get('plugins', {}).get('enabled', []) if name != 'colonymind']
        # The keys `colonymind init` writes: the adapter finds the sidecar and
        # the key file here; callbacks run inline so no capture fire is dropped.
        config.update(plugins={'enabled': [*already, 'colonymind'], 'hook_callback_timeout': 0,
                               'colonymind': {**config.get('plugins', {}).get('colonymind', {}),
                                             'sidecar_url': url, 'key_file': str(key_file),
                                             'turn_outbox_path': str(state / 'colonymind-turn-outbox.sqlite3')}},
                      memory={'provider': 'colonymind-memory', 'config': {
                          'url': url, 'api_key': secret, 'contact_id': person}})
        (state / 'config.yaml').write_text(json.dumps(config))
        (state / 'SOUL.md').write_text('Use available evidence. Preserve uncertainty. Answer the current question.')
        manager = get_plugin_manager()
        # A native import may have discovered the initial provider-only profile.
        # Reload this isolated profile after installing its explicit fixture config.
        manager.discover_and_load(force=True)
        loaded = manager._plugins.get('colonymind')
        if loaded is None or not loaded.enabled:
            raise RuntimeError('ColonyMind adapter unavailable: ' + str(loaded.error if loaded else 'not discovered'))
        requests = []
        freshness = []
        original_send = httpx.Client.send
        provider_prefix = arguments['base_url'].rstrip('/') + '/'

        class ObservedStream(httpx.SyncByteStream):
            """Tee bytes unchanged; retain only model IDs, never generated text."""
            def __init__(self, stream, observation, sse):
                self.stream, self.observation, self.sse = stream, observation, sse
                self.buffer = b''

            def consume(self, raw):
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    return
                model = value.get('model') if isinstance(value, dict) else None
                if isinstance(model, str) and 0 < len(model) <= 512 and model not in self.observation['returned_models']:
                    self.observation['returned_models'].append(model)

            def __iter__(self):
                for chunk in self.stream:
                    self.buffer += chunk
                    if self.sse:
                        while b'\n' in self.buffer:
                            line, self.buffer = self.buffer.split(b'\n', 1)
                            if line.startswith(b'data:'):
                                self.consume(line[5:].strip())
                    if len(self.buffer) > 2097152:
                        self.observation['response_identity_truncated'] = True
                        self.buffer = b''
                    yield chunk
                if not self.sse:
                    self.consume(self.buffer)

            def close(self):
                self.stream.close()

        def observe_send(client, outgoing, *args, **kwargs):
            # Observe the serialized physical request after both the ordinary
            # middleware and native Relay filters. The earlier llm_request hook
            # sees an unfiltered input and is not evidence of model visibility.
            # This instrumentation is confined to the owned subprocess and does
            # not alter request content, responses, policy or endpoint selection.
            observation = None
            body = None
            if outgoing.method == 'POST' and str(outgoing.url).startswith(provider_prefix):
                body = json.loads(outgoing.content)
            if isinstance(body, dict) and any(name in body for name in ('messages', 'input', 'system', 'instructions')):
                content = {name: body[name] for name in ('messages', 'input', 'system', 'instructions')
                           if name in body}
                text = json.dumps(content, ensure_ascii=False)
                observation = {'boundary': 'httpx_serialized_provider_request',
                    'selected_binding': request['binding'] if body.get('model') == arguments['model'] else None,
                    'endpoint_sha256': hashlib.sha256(provider_prefix.encode()).hexdigest(),
                    'model': body.get('model'), 'text': text[:131072], 'truncated': len(text) > 131072,
                    'returned_models': [], 'response_identity_truncated': False}
                requests.append(observation)
            response = original_send(client, outgoing, *args, **kwargs)
            if observation is not None:
                observation['status'] = response.status_code
                sse = response.headers.get('content-type', '').startswith('text/event-stream')
                observer = ObservedStream(response.stream, observation, sse)
                if response.is_stream_consumed:
                    # OpenAI's nonstreaming transport may already have read it.
                    observer.consume(response.content)
                else:
                    response.stream = observer
            if outgoing.url.port == port and outgoing.url.path.endswith('/sources/erasures'):
                body = json.loads(response.read())
                freshness.append({name: body.get(name) for name in
                    ('complete', 'head', 'through', 'sources_current', 'annotation_checks_current')})
            return response

        transport_observer = patch.object(httpx.Client, 'send', observe_send)
        transport_observer.start()
        arguments.update(skip_memory=False, skip_background_review=True, enabled_toolsets=[],
            session_db=SessionDB(state / 'state.db'))

        def evidence(agent, response):
            from colonymind.memory.curated import current_candidates
            manager = agent._memory_manager
            provider = manager.get_provider('colonymind') if manager is not None else None
            ledger = get_turn_idempotency_ledger(state / 'memory-state')
            return {'consumer': 'native_colonymind_automatic_recollection',
                'memory_provider_loaded': provider is not None,
                'source_sessions': sorted({row['session_id'] for row in inputs['turns']}),
                'native_session_id': agent.session_id,
                'fresh_session': agent.session_id not in {row['session_id'] for row in inputs['turns']},
                'context_routes': list(routes), 'request_observations': list(requests),
                'freshness_responses': list(freshness),
                'erased_sources': {source: not current_candidates(ledger, [{'source_id': source}], contact_id=person)
                                  for source in inputs.get('forget_source_ids', [])},
                # What the served host actually opened, not what the plan asked for.
                'embedding_and_reranking': 'endpoint' if get_store() is not None else 'not_exercised',
                'reranking': 'endpoint' if host._reranker is not None else 'not_exercised',
                'transport_delivery': 'not_exercised'}

        yield evidence
    except Exception as exc:
        # Native logs stay inside the private attempt and are never public evidence.
        print('Native memory setup:', type(exc).__name__, str(exc)[:1000], flush=True)
        raise
    finally:
        if transport_observer is not None:
            transport_observer.stop()
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(5)
        if listener is not None:
            listener.close()
        host_resources.close()
        if thread is not None and thread.is_alive():
            raise RuntimeError('Isolated memory API did not stop')


if __name__ == '__main__':
    from colonymind.qualification.native_worker import main
    main(prepare)
