"""Keep whole closures and prune discarded periods' training manifests.

The 14-closure limit is the design assumption in revisions. Digests of discarded
ref/teacher pairs stop pruned rows from being admitted again in a fresh period.
An interrupted manifest deletion is retried by sweeping manifests that no period
or known closure names, after durably recording the same exclusion witness.
"""
from __future__ import annotations

from . import _io
from .locking import StateLock


def discarded_rows(store):
    try:
        return set(_io.read_json(store.directory / "discarded-rows.json"))
    except FileNotFoundError:
        return set()


def compact(store):
    with StateLock(store.root):
        store.revisions._prune()
        excluded = discarded_rows(store)
        manifests = 0
        # A dropped manifest must not orphan a retained learned state that still
        # contains its rows. The revision store preserves identity-only withdrawal
        # markers when these closures and their learned states are deleted.
        for closure in store.revisions.known():
            if store.revisions.status(closure.id).state == "withdrawn":
                store.revisions.delete(closure.id)
                if not any(other.state == closure.state for other in store.revisions.retained()):
                    store.budgets.forget(store.name, closure.state)
        for period in store.periods.periods():
            if period.state != "discarded":
                continue
            for ident in period.manifests:
                try:
                    manifest = store.manifests.get(ident)
                except KeyError:
                    manifest = None
                if manifest is not None:
                    excluded.update(_io.digest([entry.ref, entry.teacher]) for entry in manifest.entries)
                # Publish the de-duplication witness before removing its manifest.
                _io.write_json(store.directory / "discarded-rows.json", sorted(excluded))
                store.periods.replace_manifest(period.id, ident, None)
        # A crash after detaching a manifest but before deleting its file leaves
        # no period pointer to revisit. Sweep all unreferenced files, preserving
        # references from open/closed periods and retained or pruned closures.
        referenced = {ident for period in store.periods.periods() for ident in period.manifests}
        referenced.update(ident for closure in store.revisions.known() for ident in closure.manifests)
        for ident in store.manifests.ids():
            if ident in referenced:
                continue
            manifest = store.manifests.get(ident)
            excluded.update(_io.digest([entry.ref, entry.teacher]) for entry in manifest.entries)
            # Also cover an interrupted admission that wrote its manifest before
            # attaching it to a period. Never delete the only durable witness.
            _io.write_json(store.directory / "discarded-rows.json", sorted(excluded))
            manifests += int(store.manifests.delete(ident))
        store.sweep()
        return {"closures": len(store.revisions.retained()), "manifests_pruned": manifests}
