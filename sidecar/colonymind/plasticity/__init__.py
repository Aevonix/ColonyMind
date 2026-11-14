"""Plasticity: learning in weights, governed like memory (docs/PLASTICITY.md).

Configured learners run at the existing nightly boundary and serve immediately after their gates pass. Every
input is a protocol with a synthetic implementation (``synthetic.py``), so the contract is tested on CPU. Modules
in the order a night's learning uses them:

- ``teachers.py`` and ``objectives.py``: the closed teacher list and the objective registry;
- ``manifests.py``: training rows by reference, the ``RowSource`` protocol and content-addressed manifests;
- ``admission.py``: what a period may learn from;
- ``periods.py``, ``builds.py``: the period ledger and the two ways to build a revision;
- ``budgets.py``: the declared budget per learner and the engine's recorded check;
- ``revisions.py``, ``lineage.py``: revisions as closures, their atomic promotion, retention and lineage;
- ``gates.py``: evaluation hygiene, pooled suites and the automatic gates;
- ``states.py``: candidate, serving, superseded and withdrawn, with automatic promotion and rollback;
- ``audience.py``: who a learned stack may serve;
- ``erasure.py``: unserve, delete, rebuild or discard, with no contaminated file left;
- ``store.py``: one learner's files together;
- ``rows.py``, ``nightly.py``: reference-only input binding and the configured runtime;
- ``comparator.py``: paired live outputs, outcome bounds and automatic rollback;
- ``locking.py``, ``compaction.py``: serialized changes and bounded retained closures.

No module holds GPU code: NumPy is the only numeric dependency, and a learner's own computation (its steps, its
budget measure, its predictions) is the engine's, behind the ``builds.Learner`` protocol.
"""
