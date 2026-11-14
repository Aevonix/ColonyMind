"""Run one fidelity suite against the build under test, with repeats and an independent judge.

    python benchmarks/fidelity/run.py --suite borrowed_value --state-dir <new empty directory> --out <results.json>
    python benchmarks/fidelity/run.py --suite borrowed_value_context --state-dir <new empty directory> --out <results.json>
    python benchmarks/fidelity/run.py --suite language --state-dir <new empty directory> --out <results.json>

Endpoints (never a model a live instance uses): ``COLONYMIND_BENCH_CHAT_BASE_URL`` and ``COLONYMIND_BENCH_CHAT_MODEL``
(``COLONYMIND_BENCH_CHAT_API_KEY`` optional) are the model under test: extraction and review for the borrowed-value
suites (and, across messages, the extraction of each conversation's earlier messages), the night's writer and the plain
summary for the language suite. ``COLONYMIND_BENCH_JUDGE_BASE_URL`` and
``COLONYMIND_BENCH_JUDGE_MODEL`` (``COLONYMIND_BENCH_JUDGE_API_KEY`` optional) are the judge, called once per output
at temperature 0; a judge with the model under test's name is refused. ``COLONYMIND_BENCH_CHAT_TEMPERATURE`` pins the
model under test's temperature (unset: the provider's default). Results hold every attempt's outputs, grades and the
summary (``suites.summarize``); a trap is counted in every attempt it occurs in.

Run it with the interpreter of the build under test: it imports that interpreter's ``colonymind`` and falls back to
this checkout's sidecar only when none is installed.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def _suites():
    spec = importlib.util.spec_from_file_location('fidelity_suites', HERE / 'suites.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


suites = _suites()
#: The language suite's "now" (my choice: a fixed time, so every run dates the same statements alike).
LANGUAGE_NOW = '2026-05-10T12:00:00+00:00'
ROLES = ('chat', 'reasoning', 'planning', 'extraction', 'judging', 'coding')


def environment() -> dict:
    required = ('CHAT_BASE_URL', 'CHAT_MODEL', 'JUDGE_BASE_URL', 'JUDGE_MODEL')
    missing = ['COLONYMIND_BENCH_' + name for name in required if not os.environ.get('COLONYMIND_BENCH_' + name)]
    if missing:
        raise ValueError('Missing benchmark variables: ' + ', '.join(missing))
    names = (*required, 'CHAT_API_KEY', 'JUDGE_API_KEY', 'CHAT_TEMPERATURE')
    return {name: os.environ['COLONYMIND_BENCH_' + name] for name in names if 'COLONYMIND_BENCH_' + name in os.environ}


def check_endpoints(config: dict) -> dict:
    suites.independent({'model': config.get('CHAT_MODEL')}, {'model': config.get('JUDGE_MODEL')})
    return config


def router_config(base_url: str, model: str, api_key: str = '', *, temperature=None) -> dict:
    """Every role on one model (timeouts: a builder's choice for a benchmark)."""
    entry = {'model': model, 'baseUrl': base_url, 'apiKey': api_key, 'maxTokens': 4096, 'supportsTools': True,
             'supportsJsonSchema': True}
    if temperature is not None:
        entry['extraBody'] = {'temperature': float(temperature)}
    return {'provider': 'custom', 'protocol': 'openai-chat', 'modelPool': {'candidate': entry},
            'functionRoles': {role: {'candidates': ['candidate'], 'timeoutSeconds': 120, 'deadlineSeconds': 120}
                              for role in ROLES}}


async def borrowed(items, *, projection, router, judge, repeats=suites.REPEATS, contact_id=suites.OWNER,
                   timezone_name='UTC') -> dict:
    """Every item through the dry run ``repeats`` times, each attempt judged: ``{rows, summary}``."""
    rows = []
    for repeat in range(1, repeats + 1):
        for item in items:
            row = await suites.borrowed_attempt(item, projection=projection, router=router, judge=judge,
                                                contact_id=contact_id, timezone_name=timezone_name)
            rows.append({**row, 'repeat': repeat})
    return {'rows': rows, 'summary': suites.summarize(rows)}


async def borrowed_context(items, *, ledger, projection, router, judge, repeats=suites.REPEATS,
                           contact_id=suites.OWNER, timezone_name='UTC') -> dict:
    """Every conversation's earlier messages recorded and extracted once (``suites.context_setup``), then its last
    message through the dry run ``repeats`` times, each attempt judged: ``{rows, summary, setup}``."""
    setup = {}
    for item in items:
        setup[item['id']] = await suites.context_setup(item, ledger=ledger, router=router, contact_id=contact_id)
    rows = []
    for repeat in range(1, repeats + 1):
        for item in items:
            row = await suites.context_attempt(item, projection=projection, router=router, judge=judge,
                                               contact_id=contact_id, timezone_name=timezone_name)
            rows.append({**row, 'repeat': repeat})
    return {'rows': rows, 'summary': suites.summarize(rows), 'setup': setup}


async def language(items, *, router, judge, state: Path, repeats=suites.REPEATS, now=None) -> dict:
    """Every person's three questions ``repeats`` times, each attempt in a new state, each output judged."""
    now = now or datetime.fromisoformat(LANGUAGE_NOW)
    rows = []
    for repeat in range(1, repeats + 1):
        for item in items:
            attempt = await suites.language_attempt(item, router=router, judge=judge,
                                                    work=Path(state) / f"{item['id']}-{repeat}", now=now)
            rows.extend({**row, 'repeat': repeat} for row in attempt)
    return {'rows': rows, 'summary': suites.summarize(rows),
            'by_kind': {kind: suites.summarize([row for row in rows if row['kind'] == kind])
                        for kind in ('digest', 'episode', 'summary')}}


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--suite', choices=tuple(suites.SETS), required=True)
    parser.add_argument('--state-dir', type=Path, required=True, help='a new empty directory for this run')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=suites.REPEATS)
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 10:
        parser.error('repeats is 1 to 10')
    try:
        import colonymind  # noqa: F401  (the build under test: the interpreter's own)
    except ImportError:
        sys.path.insert(0, str(HERE.parents[1] / 'sidecar'))
    from colonymind.router.router import LLMRouter
    state = args.state_dir.resolve()
    if state.exists() and any(state.iterdir()):
        raise ValueError('Use a new empty disposable state directory')
    state.mkdir(parents=True, exist_ok=True)
    config = check_endpoints(environment())
    data = suites.load_set(args.suite)
    os.environ.update(COLONYMIND_STATE_DIR=str(state), COLONYMIND_OWNER_CONTACT_ID=suites.OWNER)
    model = LLMRouter(tiers={})
    model.configure(router_config(config['CHAT_BASE_URL'], config['CHAT_MODEL'], config.get('CHAT_API_KEY', ''),
                                  temperature=config.get('CHAT_TEMPERATURE')))
    judge_router = LLMRouter(tiers={})
    judge_router.configure(router_config(config['JUDGE_BASE_URL'], config['JUDGE_MODEL'],
                                         config.get('JUDGE_API_KEY', ''), temperature=0))
    judge = suites.Judge(judge_router)
    from colonymind.beliefs.source_claims import EXTRACTION_VERSION
    build = {'extraction_version': EXTRACTION_VERSION, 'model': config['CHAT_MODEL'], 'judge': config['JUDGE_MODEL'],
             'temperature': config.get('CHAT_TEMPERATURE'), 'repeats': args.repeats}
    if args.suite in {'borrowed_value', 'borrowed_value_context'}:
        from colonymind.beliefs.source_projection import SourceClaimProjection
        from colonymind.turns import TurnIdempotencyLedger
        ledger = TurnIdempotencyLedger(state / 'turn-idempotency.db')
        projection = SourceClaimProjection(ledger)
        if args.suite == 'borrowed_value':
            report = await borrowed(data['items'], projection=projection, router=model, judge=judge,
                                    repeats=args.repeats, contact_id=data['contact_id'],
                                    timezone_name=data['timezone'])
        else:
            report = await borrowed_context(data['items'], ledger=ledger, projection=projection, router=model,
                                            judge=judge, repeats=args.repeats, contact_id=data['contact_id'],
                                            timezone_name=data['timezone'])
    else:
        report = await language(data['items'], router=model, judge=judge, state=state / 'attempts',
                                repeats=args.repeats)
    report = {'suite': args.suite, 'set': data['version'], 'build': build, 'judge_calls': judge.calls, **report}
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'suite': args.suite, 'summary': {key: value for key, value in report['summary'].items()
                                                       if key != 'by_question'}}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
