"""Executable inventory of generic SQL consumers of the initiative namespace."""
import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sys


# Schema creation is exercised by reopening the store; every other entry is a
# real reader/writer invoked below. A new SQL consumer requires an oracle case.
CONSUMERS = {
    "init.py": {"_initiative_rows"},
    "initiatives/store.py": {"InitiativeStore." + name for name in (
        "_create_tables create_with_outcome get get_by_dedup_key get_active_by_dedup_base update list count "
        "retry create_intention transition "
        "_guarded_transition answer_request answer_request_write answer_requests_since intentions findings "
        "lesson_rows get_by_ask_code get_by_hermes_ref open_ask_codes count_transitions last_transition_at "
        "failures_since tokens_since prune_intentions log_history last_responses get_history "
        "get_agent_history delete_old "
        "record_engagement share resolve_share last_response_origin finished_under_base").split()},
    "instinct/outcomes.py": {"Sources.intention", "Sources.questions", "Sources.history", "_load_mind_wish_scope"},
    "mind/approval_history.py": {"_snapshot"},
    "mind/tick.py": {"Mind._owner_cadences"},
    "reports/instance.py": {"Instance.mind_rows", "Instance.open_rows", "Instance.handed_over", "Instance.history"},
    "reports/senses.py": {"collect"},
    "plasticity/learners/action_success.py": {"Rows._created", "Rows.remember", "report"},
    "plasticity/learners/reply_timing.py": {"Rows.snapshot", "report"},
    "plasticity/learners/salience.py": {"Rows.snapshot", "Rows.label", "report"},
}

DYNAMIC_CONSUMERS = {
    "secrets/migrate.py": {"_tables", "StoreRewriter.walk", "StoreRewriter._repoint_database"},
}


def discover(root):
    found = {}
    pattern = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE|REFERENCES)\s+(?:IF NOT EXISTS\s+)?"
                         r"(?:initiatives|generic_initiatives|assignment_history|ask_answer_requests|local_draft_acceptances|"
                         r"initiative_engagement|initiative_shares)\b", re.I)
    def walk(node, path, parents=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parents += (node.name,)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and pattern.search(node.value):
            found.setdefault(path, set()).add(".".join(parents))
        for child in ast.iter_child_nodes(node):
            walk(child, path, parents)
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("owner_goals/"):
            continue
        walk(ast.parse(path.read_text()), relative)
    return found


def run(stores, monkeypatch, snapshot):
    """Exercise every inventoried SQL consumer with a live Secret and an absent ID.

    The profile hook proves the manifest was actually invoked. Per-call owner
    snapshots detect writes even when a reader returns no projection. Explicit
    assertions on the low-level sources prevent higher-level filters masking a leak.
    """
    from colonymind import init
    from colonymind.initiatives.store import InitiativeStore, SettleGuard, TransitionRefused
    from colonymind.instinct.outcomes import Sources, _load_mind_wish_scope
    from colonymind.mind import approval_history, audit, outreach
    from colonymind.mind.tick import Mind
    from colonymind.reports import faculties, senses
    from colonymind.reports.instance import Instance
    from colonymind.secrets.migrate import Report, StoreRewriter, _tables
    from colonymind.plasticity.learners import action_success, reply_timing, salience
    from test_owner_goals import create, write, PROGRESS, owner_snapshot
    from test_plasticity_action_success import runtime as action_runtime
    from test_plasticity_reply_timing import runtime as reply_runtime
    from test_plasticity_salience_live import runtime as salience_runtime, NAME

    store, goals = stores
    root = create(goals, title="Zorp sentinel", description="Zorp sentinel")
    child = create(goals, parent_id=root["id"], title="Zorp sentinel")
    write(goals, "progress", PROGRESS | {"text": "Zorp sentinel"}, id=child["id"])
    proposal = write(goals, "propose", {"kind": "progress", "target_id": child["id"],
        "payload": {"at": PROGRESS["at"], "text": "Zorp sentinel"}})["proposal"]
    before = snapshot(goals)
    seen = set()
    code_root = Path(init.__file__).parent
    def profile(frame, event, arg):
        if event == "call":
            path = Path(frame.f_code.co_filename)
            if path.is_relative_to(code_root):
                seen.add((path.relative_to(code_root).as_posix(), frame.f_code.co_qualname))
    def check(value):
        assert "Zorp sentinel" not in repr(value)
        assert all(target not in repr(value) for target in (root["id"], child["id"], proposal["id"]))
        assert snapshot(goals) == before
        return value
    floor = datetime(2000, 1, 1, tzinfo=timezone.utc)
    ceiling = datetime(9999, 1, 1, tzinfo=timezone.utc)
    now = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        reopened = InitiativeStore(store._state_dir)
        reopened.close()
        # Approval history shares the generic namespace, including the full-text
        # notice lookups used by its existing terms reader. Neither an owner-goal
        # ID nor its dedupe key may expose or change owner-goal storage.
        recorded = approval_history._Recorded([], {})
        for target in (root["id"], child["id"], proposal["id"], "absent"):
            assert recorded.get(target) is None
            assert recorded.get_by_dedup_key(target) is None
            check(None)
        check(approval_history._snapshot(store, now))
        assert init._initiative_rows(store._db_path) == 0
        sources = Sources(store._state_dir, owner_id="owner")
        instance = Instance(store._state_dir)
        try:
            for target in (root["id"], child["id"], proposal["id"], "absent"):
                assert sources.intention(target) is None
                assert sources.history(target) == []
                assert store.get(target) is None
                assert store.get_by_dedup_key(target) is None
                assert store.get_active_by_dedup_base(target) is None
                assert not store.finished_under_base(target, other_than="absent")
                assert store.get_by_hermes_ref(target) is None
                assert store.get_by_ask_code(target) is None
                assert instance.handed_over([target]) == {}
                assert store.get_history(target) == []
                assert store.last_responses([target]) == {}
                assert store.last_response_origin(target) is None
                assert store.idea_states([target]) == [
                    {"id": target, "exists": False, "state": None, "status": None}]
                assert not store.record_engagement(target, {
                    "receipt_id": "hidden-" + target, "kind": "open", "visible_fraction": 1,
                    "surface": "ideas", "anchor": "fixture"})
                assert store.share(target, "Zorp sentinel", "Zorp sentinel") is None
                assert store.resolve_share(target) is None
                assert store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0
                assert store._db.execute("SELECT count(*) FROM initiative_shares").fetchone()[0] == 0
                for fn in (lambda: store.update(target, context={}), lambda: store.complete(target, "agent", "Zib"),
                           lambda: store.fail(target, "agent", "Zib"), lambda: store.retry(target, "agent"),
                           lambda: store.cancel(target, "agent"),
                           lambda: store.transition(target, "done", action="rated"),
                           lambda: store.log_history(target, "rated", "agent")):
                    assert fn() is None
                    check(None)
                try:
                    store.transition(target, "done", action="rated", guard=SettleGuard(expect=("recorded",)))
                except TransitionRefused as exc:
                    assert exc.current is None
                else:
                    raise AssertionError("hidden transition succeeded")
                assert store.answer_request("fixture", target) is None
                assert not store.record_answer_request(scope="fixture", key=target, ask_id=target,
                    request_hash="fixture", status_code=200, response={}, at=now)
                assert audit.why(store, target) is None
                check(None)
            check(sources.questions())
            check(_load_mind_wish_scope({"refs": {"wish": root["id"]}}, sources))
            check(instance.mind_rows(floor, ceiling))
            check(instance.open_rows(("recorded", "asked", "approved")))
            check(instance.history(("owner_goal_create", "owner_goal_progress"), floor, ceiling))
            check(senses.collect(instance, start=floor, end=ceiling))
            check(faculties.collect(store._state_dir, now=now))
        finally:
            instance.close()
        # Cover every selector/aggregate, including requests for owner kinds.
        assert store.ideas_page(1) == ([], None, False)
        for fn in (lambda: store.intentions(), lambda: store.intentions(kind=["owner_goal"]),
                   lambda: store.list(), lambda: store.count(), lambda: store.open_ask_codes(),
                   lambda: store.lesson_rows(),
                   lambda: store.findings(["done"], ["fixture"]), lambda: store.failures_since("internal", floor),
                   lambda: store.tokens_since(floor), lambda: store.count_transitions("owner_goal_create", floor),
                   lambda: store.last_transition_at("owner_goal_create"), lambda: store.get_agent_history("agent"),
                   lambda: store.answer_requests_since(floor),
                   lambda: store.prune_intentions(ceiling),
                   lambda: store.delete_old(["recorded"], ceiling), lambda: audit.log(store),
                   lambda: audit.stats(store, now=now)):
            check(fn())
        mind = Mind(config={"quiet_hours": ""}, store=store, state_dir=store._state_dir,
                    owner_id="owner", backups=False, clock=lambda: now)
        check(mind.dispatch())
        check(mind.goals.open())
        # Without a commitment anchor, hidden IDs must not supply a previous
        # attempt or a due time. Valid metadata takes every case through the SQL.
        cadence = {"person_id": "owner", "status": "pending",
                   "metadata": {"kind": "cadence", "recipient": "owner",
                                "cadence_minutes": 1440, "topic": "Zib"}}
        for target in (root["id"], child["id"], proposal["id"], "absent"):
            assert check(mind._owner_cadences([cadence | {"id": target}])) == []
        assert check(mind._owner_cadences([cadence | {"id": "absent", "made_at": now.isoformat()}])) == [
            outreach.Cadence("absent", "Zib", now + timedelta(days=1))]
        # Populate both generic creation paths so the oracle checks successful
        # generic writes, not just no-op writes against unknown IDs.
        generic = store.create(type="fixture", description="Zib")
        check(generic)
        check(store.ideas_page(1))
        check(store.idea_states([generic.id]))
        assert check(store.record_engagement(generic.id, {
            "receipt_id": "generic", "kind": "open", "visible_fraction": 1,
            "surface": "ideas", "anchor": "fixture"}))
        shared = check(store.share(generic.id, "Zib", "Zib"))
        assert check(store.resolve_share(shared["share_id"])) == shared
        store.log_history(generic.id, "llm_approved", "agent", origin={"surface": "ideas"})
        assert check(store.last_response_origin(generic.id)) == {"surface": "ideas"}
        generic, _ = store.create_intention(kind="task", type="fixture", title="Zib", drive="curiosity",
            cls="internal", decision="act", decision_reason="fixture", status="approved", dedup_key=None)
        check(generic)
        # All three learners and their report SQL run against the same seeded DB.
        for factory, name, module in ((action_runtime, "success", action_success),
                                     (reply_runtime, "timing", reply_timing),
                                     (salience_runtime, NAME, salience)):
            service = factory(store._state_dir, [now])
            try:
                assert name in service.loaded, service.rejected
                rows = service.loaded[name].driver.rows
                check(rows.snapshot())
                if module is action_success:
                    for target in (root["id"], "absent"):
                        rows.remember(target, "title", "Zib")
                        check(None)
                    rows.remember(generic.id, "fixture", 1)
                    check(None)
                if module is salience:
                    for target in (root["id"], "absent"):
                        check(rows.label({"id": target, "verdict": None, "created_at": now.isoformat()}))
                check(module.report(service.config, name, runtime=service))
            finally:
                service.close()
        check(store.transition(generic.id, "done", action="rated"))
        check(store.prune_intentions(ceiling))
        check(store.delete_old(["completed"], ceiling))
        # Recovery snapshots retain all tables. Their generic evidence/report
        # projections use the same namespace, while owner projections must supply
        # the explicit owner audience to OwnerGoalsStore.
        from colonymind.backup import _snapshot_databases
        archive = store._state_dir / "archive"
        check(_snapshot_databases(store._state_dir, archive))
        archived_sources = Sources(archive, owner_id="owner")
        archived_reports = Instance(archive)
        try:
            for target in (root["id"], child["id"], proposal["id"]):
                assert archived_sources.intention(target) is None
                assert archived_sources.history(target) == []
            check(archived_reports.mind_rows(floor, ceiling))
            check(archived_reports.history(("owner_goal_create", "owner_goal_propose"), floor, ceiling))
        finally:
            archived_reports.close()
        # Dynamic table walkers are consumers too. The transform is deliberately
        # destructive on generic text, so a missing exclusion changes the snapshot.
        rewriter = StoreRewriter(store._state_dir, report=Report())
        with goals.connection() as db:
            assert all(not table.startswith("owner_goal_") for table in _tables(db))
        def transform(text, row):
            check(text)
            return text.replace("Zorp sentinel", "Zib")
        rewriter.walk(transform)
        rewriter._repoint_database(store._db_path, [("Zorp sentinel", "Zib")])
        check(None)
    finally:
        sys.setprofile(previous)
    required = {(path, name) for path, names in {**CONSUMERS, **DYNAMIC_CONSUMERS}.items() for name in names}
    assert required <= seen, sorted(required - seen)
    assert owner_snapshot(goals) == before
