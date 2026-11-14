"""Offline collection timing on a disposable clone of an operator's state snapshot.

Run with: python -m colonymind.context_card.measure STATE_DIR_COPY --owner-id CID
No server, worker, tick, model, card publication or network connection is started.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time
from types import SimpleNamespace


def _no_network(*args, **kwargs):
    raise RuntimeError("context_card_measurement_is_offline")


async def measure(state_dir: Path, home: Path, *, owner: str, name: str, day: str | None):
    from colonymind.config import load_config
    from colonymind.contacts.comms import CommsLog
    from colonymind.commitments.store import CommitmentStore
    from colonymind.initiatives.store import InitiativeStore
    from colonymind.intelligence.components.preference_learner import PreferenceLearner
    from colonymind.mind.tick import Mind
    from colonymind.self_model.perspective import SelfPerspective
    from colonymind.turns import get_turn_idempotency_ledger
    from .builder import build
    from .inputs import InputCollector
    from .refresh import local_day

    setup_start = time.perf_counter()
    with ExitStack() as cleanup:
        ledger = get_turn_idempotency_ledger(state_dir)
        initiatives = InitiativeStore(state_dir=state_dir)
        cleanup.callback(initiatives.close)
        commitments = CommitmentStore(state_dir / "colonymind-commitments.db")
        learner = PreferenceLearner(db_path=str(state_dir / "colonymind-preferences.db"),
            perspective=SelfPerspective(ledger, owner_id=owner))
        if learner._conn is not None:
            cleanup.callback(learner._conn.close)
        comms = CommsLog(str(state_dir / "colonymind-comms.db"), source_ledger=ledger)
        cleanup.callback(comms._conn.close)
        host = SimpleNamespace(_preference_learner=learner, _commitment_store=commitments,
                               _initiative_store=initiatives, _comms_log=comms)
        config = load_config(home, environ={})
        mind = Mind(config=config.get("mind"), store=initiatives, state_dir=state_dir,
                    owner_id=owner, commitments=commitments, ledger=ledger,
                    router=None, backups=False, identity=lambda: {})
        cleanup.callback(mind.concerns.close)
        collector = InputCollector(state_dir, host=host, mind_provider=lambda: mind,
            owner_provider=lambda: owner, name_provider=lambda: name, day_provider=lambda: day or local_day())
        setup_ms = (time.perf_counter() - setup_start) * 1000
        started = time.perf_counter()
        prepared = collector.prepare()
        prepared_at = time.perf_counter()
        # Poll a concurrent future so timing also works in offline sandboxes
        # whose socketpair cannot wake an asyncio loop from a worker thread.
        # Measure the ledger work inside the worker, excluding polling latency.
        def read_ledger():
            read_start = time.perf_counter()
            inputs = collector.collect_ledger(prepared)
            return inputs, (time.perf_counter() - read_start) * 1000
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(read_ledger)
            while not future.done():
                await asyncio.sleep(0.001)
            inputs, ledger_ms = future.result()
        collected_at = time.perf_counter()
        card = build(inputs)
        return {
            "setup_ms": round(setup_ms, 3),
            "shared_capture_ms": round((prepared_at - started) * 1000, 3),
            "ledger_read_ms": round(ledger_ms, 3),
            "collection_ms": round((collected_at - started) * 1000, 3),
            "build_ms": round((time.perf_counter() - collected_at) * 1000, 3),
            "tokens_conservative": card.token_count,
            "source_ref_count": len(inputs.source_refs),
            "highlight_count": len(inputs.highlights),
            "unavailable": list(inputs.unavailable),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_dir_copy", type=Path, help="A quiescent snapshot, never a live state directory")
    parser.add_argument("--home-copy", type=Path, help="Optional snapshot of identity.yaml/colonymind.yaml")
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--owner-name", default="the owner")
    parser.add_argument("--day", help="Optional local day YYYY-MM-DD, for historical snapshots")
    args = parser.parse_args(argv)
    if not args.state_dir_copy.is_dir():
        parser.error("state_dir_copy must be a directory")
    if args.home_copy is not None and not args.home_copy.is_dir():
        parser.error("home_copy must be a directory")
    # Prevent model libraries or accidental producer changes from reaching a
    # network. Socketpair/loop wakeups remain available for the ledger thread.
    socket.socket.connect = _no_network
    socket.socket.connect_ex = _no_network
    socket.socket.sendto = _no_network
    socket.create_connection = _no_network
    socket.getaddrinfo = _no_network
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      COLONYMIND_SKIP_DOTENV="1", PYTHON_DOTENV_DISABLED="1")
    with tempfile.TemporaryDirectory(prefix="context-card-measure-") as temporary:
        root = Path(temporary)
        state_dir, home = root / "state", root / "home"
        shutil.copytree(args.state_dir_copy, state_dir, ignore=shutil.ignore_patterns("*.sock", ".git"))
        home.mkdir()
        for leaf in ("identity.yaml", "colonymind.yaml"):
            source = (args.home_copy or args.state_dir_copy) / leaf
            if source.is_file():
                shutil.copy2(source, home / leaf)
        os.environ.update(COLONYMIND_STATE_DIR=str(state_dir), COLONYMIND_HOME=str(home),
                          COLONYMIND_OWNER_PERSON_ID=args.owner_id, COLONYMIND_OWNER_CONTACT_ID=args.owner_id)
        # All constructors and migrations touch only this disposable clone.
        result = asyncio.run(measure(state_dir, home, owner=args.owner_id,
                                    name=args.owner_name, day=args.day))
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
