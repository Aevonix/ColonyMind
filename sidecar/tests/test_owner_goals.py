"""Owner-goals contract regression tests with synthetic payloads and trusted test doubles."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import sqlite3
import uuid

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.auth import anonymous_authority, key_authority
from colonymind.api.routers import owner_goals as routes
from colonymind.initiatives.store import InitiativeStore
from colonymind.owner_goals.artifacts import ArtifactHandle, ArtifactResolver, LibraryItem
from colonymind.owner_goals.authority import (ActionContext, Audience, VerifiedAction,
                                            get_action_verifier, verify_owner_goal_action)
from colonymind.owner_goals.store import OwnerGoalsStore, page
from colonymind.owner_goals.validation import (GoalError, MAX_INTEGER, RESPONSE_BYTES, dumps, loads,
                                             pagination, timestamp, validate)

ROOT = "/v1/mind/owner-goals"
OWNER = Audience("owner", frozenset({"owner"}), True)
CREATE = {"title": "Glorp", "description": "Blim blam.", "category": "Zorb", "source_id": "unkeyed:glorp"}
PROGRESS = {"at": "2026-01-02T03:04:05Z", "text": "Flim.", "source_id": "chat:flim"}


def permit(ctx, *, resolved_action, receipt, connection):
    purpose = {"propose": "proposal_only", "confirm": "confirmation"}.get(resolved_action["action"], "requested_write")
    return VerifiedAction(ctx.owner_id, ctx.principal_id,
                          receipt["evidence_id"] if receipt else "test-evidence:" + ctx.operation_id,
                          purpose, explicit_tier=True)


@pytest.fixture
def stores(tmp_path):
    intentions = InitiativeStore(tmp_path)
    goals = OwnerGoalsStore(intentions)
    yield intentions, goals
    intentions.close()


def ctx(action, body, *, id=None, key=None, principal="agent", owner="owner", path=None, method=None):
    method = method or {"create": "POST", "patch": "PATCH", "delete": "DELETE", "detach": "DELETE"}.get(action, "POST")
    if path is None:
        path = ROOT + ("/" + id if id else "")
        path += {"progress": "/progress", "propose": "/proposals", "confirm": "/confirm", "attach": "/artifacts"}.get(action, "")
    digest = hashlib.sha256(dumps([method, path, body]).encode()).hexdigest()
    return ActionContext(owner, principal, replace(OWNER, owner_id=owner, person_ids=frozenset({owner})),
                         method, path, key or str(uuid.uuid4()), body, digest)


def write(goals, action, body, *, id=None, verifier=permit, **kwargs):
    return goals.mutate(ctx(action, body, id=id), action, verifier, id=id, **kwargs)[1]


def create(goals, **edits):
    return write(goals, "create", CREATE | edits)["goal"]


SYNTHETIC_TOKEN = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"


def assert_no_plaintext_bytes(path, *canaries):
    for suffix in ("", "-wal"):
        candidate = path.with_name(path.name + suffix)
        if candidate.exists():
            raw = candidate.read_bytes()
            for canary in canaries:
                assert canary.encode() not in raw, candidate.name


def test_fresh_create_patch_delete_never_write_plaintext_cells_or_wal(stores):
    _, goals = stores
    changed = "hf_" + "ZyXwVuTsRqPoNmLkJiHgFeDcBa987654"
    with goals.connection(write=True) as db:
        assert db.execute("PRAGMA secure_delete").fetchone()[0] == 1
    root = create(goals, description=(SYNTHETIC_TOKEN + " ") * 100)
    assert_no_plaintext_bytes(goals.db_path, SYNTHETIC_TOKEN, changed)
    write(goals, "patch", {"expected_revision": 1, "description": changed}, id=root["id"])
    assert_no_plaintext_bytes(goals.db_path, SYNTHETIC_TOKEN, changed)
    write(goals, "delete", {"expected_revision": 2}, id=root["id"])
    assert_no_plaintext_bytes(goals.db_path, SYNTHETIC_TOKEN, changed)


@pytest.mark.parametrize("resume", [False, True])
def test_migration_scrubs_historical_cells_and_free_pages_before_marker(stores, monkeypatch, resume):
    from colonymind.owner_goals import sealing
    from colonymind.secrets.migrate import MARKER, migrate
    intentions, goals = stores
    connection = goals.connection
    @contextmanager
    def legacy_connection(**kwargs):
        with connection(**kwargs) as db:
            db.execute("PRAGMA secure_delete=OFF")
            yield db
    with monkeypatch.context() as patch:
        patch.setattr(goals, "connection", legacy_connection)
        patch.setattr(goals.codec, "seal", lambda value, owner: value)
        root = create(goals, description=(SYNTHETIC_TOKEN + " ") * 160)
        write(goals, "patch", {"expected_revision": 1, "description": "Zib."}, id=root["id"])
    # Include genuinely free pages from old owner content, not just live plaintext.
    with connection(write=True) as db:
        db.execute("PRAGMA secure_delete=OFF")
        db.execute("CREATE TABLE fixture_old_pages (content TEXT)")
        db.execute("INSERT INTO fixture_old_pages VALUES (?)", ((SYNTHETIC_TOKEN + " ") * 2000,))
        db.execute("DROP TABLE fixture_old_pages")
    intentions._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert SYNTHETIC_TOKEN.encode() in goals.db_path.read_bytes()
    before = logical_owner_snapshot(goals)
    vault = goals.codec.vault
    vault.set_marker("migration:v2-owner-goals", "fixture")
    if resume:
        with monkeypatch.context() as patch:
            connect = sqlite3.connect
            class Interrupted(sqlite3.Connection):
                def execute(self, sql, *args):
                    if sql == "VACUUM":
                        raise sqlite3.OperationalError("fixture scrub interruption")
                    return super().execute(sql, *args)
            patch.setattr(sqlite3, "connect", lambda *args, **kwargs: connect(*args, **(kwargs | {"factory": Interrupted})))
            with pytest.raises(sqlite3.OperationalError, match="scrub interruption"):
                migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
        assert vault.marker(MARKER) is None
        assert SYNTHETIC_TOKEN not in owner_dump(goals)
        assert SYNTHETIC_TOKEN.encode() in goals.db_path.read_bytes()
    verify = sealing.verify_physical_sealing
    checked = []
    def verify_before_marker(path, canaries):
        assert vault.marker(MARKER) is None
        assert SYNTHETIC_TOKEN.encode() in canaries
        with sqlite3.connect(path, timeout=0.01) as competing:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                competing.execute("BEGIN IMMEDIATE")
        verify(path, canaries)
        checked.append(path)
    monkeypatch.setattr(sealing, "verify_physical_sealing", verify_before_marker)
    migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert checked == [goals.db_path]
    assert vault.marker(MARKER)
    assert_no_plaintext_bytes(goals.db_path, SYNTHETIC_TOKEN)
    assert logical_owner_snapshot(goals) == before


def test_physical_verification_failure_prevents_completion(unsealed_owner_state, monkeypatch):
    from colonymind.owner_goals import sealing
    from colonymind.secrets.migrate import MARKER, migrate
    from colonymind.secrets.vault import VaultUnavailable
    intentions, goals, *_ = unsealed_owner_state
    vault = goals.codec.vault
    verify = sealing.verify_physical_sealing
    def surviving_bytes(path, canaries):
        # Simulate a scrub that left an obsolete cell outside the logical schema.
        with path.open("ab") as stream:
            stream.write(SYNTHETIC_TOKEN.encode())
        verify(path, canaries)
    with monkeypatch.context() as patch:
        patch.setattr(sealing, "verify_physical_sealing", surviving_bytes)
        with pytest.raises(VaultUnavailable, match="remains in database storage"):
            migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert vault.marker(MARKER) is None
    migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert vault.marker(MARKER)
    assert_no_plaintext_bytes(goals.db_path, SYNTHETIC_TOKEN)


def test_live_reader_defers_physical_completion_and_retry_succeeds(unsealed_owner_state, monkeypatch):
    from colonymind.secrets.migrate import MARKER, migrate
    from colonymind.secrets.vault import VaultUnavailable
    intentions, goals, *_ = unsealed_owner_state
    vault = goals.codec.vault
    reader = sqlite3.connect(goals.db_path)
    reader.execute("BEGIN")
    retained = reader.execute("SELECT context FROM owner_goal_records ORDER BY id").fetchall()
    try:
        with monkeypatch.context() as patch:
            connect = sqlite3.connect
            patch.setattr(sqlite3, "connect", lambda *args, **kwargs: connect(*args, **(kwargs | {"timeout": 0.01})))
            with pytest.raises(VaultUnavailable, match="active readers"):
                migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
        assert vault.marker(MARKER) is None
        assert reader.execute("SELECT context FROM owner_goal_records ORDER BY id").fetchall() == retained
    finally:
        reader.close()
    migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert vault.marker(MARKER)
    assert_no_plaintext_bytes(goals.db_path, SYNTHETIC_TOKEN)


@pytest.mark.parametrize("suffix", ["", "-wal"])
def test_physical_scan_checks_both_files_and_chunk_boundaries(tmp_path, suffix):
    from colonymind.owner_goals.sealing import plaintext_canaries, verify_physical_sealing
    from colonymind.secrets.vault import VaultUnavailable
    path = tmp_path / "fixture.db"
    path.write_bytes(b"")
    target = tmp_path / (path.name + suffix)
    target.write_bytes(b"\x00" * (1024 * 1024 - 5) + SYNTHETIC_TOKEN.encode())
    with pytest.raises(VaultUnavailable, match="remains in database storage"):
        verify_physical_sealing(path, plaintext_canaries(dumps({"text": SYNTHETIC_TOKEN})))


def logical_owner_snapshot(goals):
    return [[[goals.codec.plaintext(value) for value in row] for row in table]
            for table in owner_snapshot(goals)]


def owner_dump(goals):
    with goals.connection() as db:
        return "\n".join(db.iterdump())


@pytest.fixture
def unsealed_owner_state(stores, monkeypatch):
    """Synthetic pre-fix writes, including immutable copies and a tombstone."""
    intentions, goals = stores
    goals.resolver = ArtifactResolver(library=Library())
    with monkeypatch.context() as patch:
        patch.setattr(goals.codec, "seal", lambda value, owner: value)
        request = ctx("create", CREATE | {"description": SYNTHETIC_TOKEN})
        original = goals.mutate(request, "create", permit)
        root = original[1]["goal"]
        write(goals, "progress", PROGRESS | {"text": SYNTHETIC_TOKEN}, id=root["id"])
        proposal = write(goals, "propose", {"kind": "goal", "target_id": None,
            "payload": {"title": "Glorp", "description": SYNTHETIC_TOKEN, "category": "Zorb"}})["proposal"]
        write(goals, "patch", {"expected_revision": 2, "description": SYNTHETIC_TOKEN + " Zib"}, id=root["id"])
        write(goals, "attach", {"type": "library_item", "ref": "item", "label": SYNTHETIC_TOKEN}, id=root["id"])
        write(goals, "delete", {"expected_revision": 4}, id=root["id"])
    with goals.connection(write=True) as db:
        db.execute("INSERT INTO owner_goal_legacy_answers VALUES (?,?,?,?,?,?,?)",
                   ("fixture", "fixture", root["id"], "fixture", 200, dumps({"text": SYNTHETIC_TOKEN}), "fixture"))
        db.execute("UPDATE owner_goal_records SET decision=? WHERE id=?", (SYNTHETIC_TOKEN, root["id"]))
    return intentions, goals, request, original, proposal


@pytest.mark.parametrize("action", ["create", "patch", "progress", "propose", "confirm", "attach"])
def test_owner_content_is_sealed_on_every_fresh_write(stores, action):
    intentions, goals = stores
    goals.resolver = ArtifactResolver(library=Library())
    root = create(goals)
    if action == "create":
        result = create(goals, description=SYNTHETIC_TOKEN, source_id=SYNTHETIC_TOKEN)
    elif action == "patch":
        result = write(goals, action, {"expected_revision": 1, "description": SYNTHETIC_TOKEN}, id=root["id"])
    elif action == "progress":
        result = write(goals, action, PROGRESS | {"text": SYNTHETIC_TOKEN, "unit": SYNTHETIC_TOKEN}, id=root["id"])
    elif action in {"propose", "confirm"}:
        proposal = write(goals, "propose", {"kind": "goal", "target_id": None,
            "payload": {"title": "Glorp", "description": SYNTHETIC_TOKEN, "category": "Zorb"}})["proposal"]
        result = proposal if action == "propose" else write(goals, "confirm", {"expected_revision": 1}, id=proposal["id"])
    else:
        result = write(goals, action, {"type": "library_item", "ref": "item", "label": SYNTHETIC_TOKEN}, id=root["id"])
    assert SYNTHETIC_TOKEN not in owner_dump(goals)
    assert_no_plaintext_bytes(goals.db_path, SYNTHETIC_TOKEN)
    assert SYNTHETIC_TOKEN in dumps(logical_owner_snapshot(goals))
    # The approved reader and verifier retain the original content, not a changed hash preimage.
    assert SYNTHETIC_TOKEN in dumps(result)
    reopened = OwnerGoalsStore(intentions, resolver=goals.resolver)
    assert reopened.read("owner", OWNER) == goals.read("owner", OWNER)


def test_owner_sealing_repairs_old_completion_and_preserves_audit_receipts(unsealed_owner_state):
    from colonymind.secrets.migrate import MARKER, migrate, migrated
    intentions, goals, request, original, proposal = unsealed_owner_state
    before = logical_owner_snapshot(goals)
    vault = goals.codec.vault
    vault.set_marker("migration:v1", "fixture")
    assert not migrated(intentions._state_dir, vault=vault)
    report = migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert any(item["table"] == "owner_goal_records" for item in report["stores"])
    assert SYNTHETIC_TOKEN not in owner_dump(goals)
    for suffix in ("", "-wal"):
        path = goals.db_path.with_name(goals.db_path.name + suffix)
        if path.exists():
            assert SYNTHETIC_TOKEN.encode() not in path.read_bytes()
    assert logical_owner_snapshot(goals) == before
    assert vault.marker(MARKER)
    reopened = OwnerGoalsStore(intentions, resolver=goals.resolver)
    def verify(ctx, **kwargs):
        if kwargs["receipt"] is not None:
            retained = json.loads(kwargs["receipt"]["resolved_action"])
            assert retained == kwargs["resolved_action"]
            assert retained["body"] == validate("create", ctx.body)
            assert hashlib.sha256(dumps([ctx.method, ctx.path, ctx.body]).encode()).hexdigest() == kwargs["receipt"]["request_hash"]
        return permit(ctx, **kwargs)
    assert reopened.mutate(request, "create", verify) == original
    with reopened.connection() as db:
        row = db.execute("SELECT details FROM owner_goal_history WHERE initiative_id=? AND action='owner_goal_propose'",
                         (proposal["id"],)).fetchone()
        audit = reopened._decode(row[0])
    retained_proposal = reopened.read("owner", OWNER, id=proposal["id"], proposals=True)["proposal"]
    assert audit["payload_digest"] == hashlib.sha256(dumps(retained_proposal["payload"]).encode()).hexdigest()
    # Settlement after sealing reads the original proposal and writes sealed goal/receipt copies.
    write(reopened, "confirm", {"expected_revision": 1}, id=proposal["id"])
    assert SYNTHETIC_TOKEN not in owner_dump(goals)
    frozen = owner_snapshot(goals)
    assert migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)["stores"] == []
    assert owner_snapshot(goals) == frozen


@pytest.mark.parametrize("table", ["records", "history", "proposals", "artifacts", "receipts", "legacy_answers"])
@pytest.mark.parametrize("after", [False, True])
def test_interrupted_owner_sealing_resumes_without_changing_logical_rows(unsealed_owner_state, monkeypatch, table, after):
    from colonymind.secrets.migrate import MARKER, migrate
    intentions, goals, request, original, _ = unsealed_owner_state
    before = logical_owner_snapshot(goals)
    vault = goals.codec.vault
    connect = sqlite3.connect
    class Interrupted(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith(f'UPDATE "owner_goal_{table}"'):
                if after:
                    super().execute(sql, *args)
                raise sqlite3.OperationalError("fixture interruption")
            return super().execute(sql, *args)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", lambda *args, **kwargs: connect(*args, **(kwargs | {"factory": Interrupted})))
        with pytest.raises(sqlite3.OperationalError, match="fixture interruption"):
            migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert not vault.marker(MARKER)
    assert logical_owner_snapshot(goals) == before
    assert SYNTHETIC_TOKEN in owner_dump(goals)
    migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert SYNTHETIC_TOKEN not in owner_dump(goals)
    assert logical_owner_snapshot(goals) == before
    assert goals.mutate(request, "create", permit) == original
    assert vault.marker(MARKER)


def test_owner_sealing_dry_run_has_no_writes(unsealed_owner_state):
    from colonymind.secrets.migrate import MARKER, migrate
    intentions, goals, *_ = unsealed_owner_state
    before = owner_snapshot(goals)
    assert not goals.codec.vault.db_path.exists()
    report = migrate(intentions._state_dir, dry_run=True, owner_id="owner", write_report=False)
    assert report["stores"]
    assert SYNTHETIC_TOKEN not in dumps(report)
    assert owner_snapshot(goals) == before
    assert not goals.codec.vault.db_path.exists()
    assert goals.codec.vault.marker(MARKER) is None


@pytest.mark.parametrize("table,column", [("records", "context"), ("history", "details"), ("receipts", "result")])
def test_sealing_callback_cannot_edit_immutable_logical_content(unsealed_owner_state, table, column):
    intentions, goals, *_ = unsealed_owner_state
    goals.seal_existing(intentions._db_path, vault=goals.codec.vault)
    before = owner_snapshot(goals)
    changed = goals.codec.seal(dumps({"text": SYNTHETIC_TOKEN + " Zib"}), "owner")
    with pytest.raises(sqlite3.IntegrityError), goals.connection(write=True) as db:
        db.create_function("owner_goal_storage_rewrite", 2, goals.codec.equivalent)
        condition = " WHERE kind='owner_goal_progress'" if table == "records" else ""
        db.execute(f'UPDATE "owner_goal_{table}" SET "{column}"=?' + condition, (changed,))
    assert owner_snapshot(goals) == before


def test_owner_write_and_migration_fail_closed_when_vault_cannot_seal(unsealed_owner_state, monkeypatch):
    from colonymind.secrets.migrate import MARKER, migrate
    from colonymind.secrets.vault import VaultUnavailable
    intentions, goals, *_ = unsealed_owner_state
    before = owner_snapshot(goals)
    vault = goals.codec.vault
    def unavailable(*args, **kwargs):
        raise VaultUnavailable("fixture unavailable")
    monkeypatch.setattr(vault, "seal", unavailable)
    with pytest.raises(VaultUnavailable):
        create(goals, description=SYNTHETIC_TOKEN)
    assert owner_snapshot(goals) == before
    with pytest.raises(VaultUnavailable):
        migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert owner_snapshot(goals) == before
    assert vault.marker(MARKER) is None


def test_owner_sealing_requires_wal_retirement_before_completion(unsealed_owner_state, monkeypatch):
    from colonymind.secrets.migrate import MARKER, migrate
    from colonymind.secrets.vault import VaultUnavailable
    intentions, goals, *_ = unsealed_owner_state
    vault = goals.codec.vault
    connect = sqlite3.connect
    class BusyCheckpoint(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql == "PRAGMA wal_checkpoint(TRUNCATE)":
                return super().execute("SELECT 1,1,0")
            return super().execute(sql, *args)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", lambda *args, **kwargs: connect(*args, **(kwargs | {"factory": BusyCheckpoint})))
        with pytest.raises(VaultUnavailable, match="active readers"):
            migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert SYNTHETIC_TOKEN not in owner_dump(goals)
    assert vault.marker(MARKER) is None
    migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert vault.marker(MARKER)


def test_owner_completion_rechecks_for_remaining_plaintext(unsealed_owner_state, monkeypatch):
    from colonymind.owner_goals.sealing import Codec
    from colonymind.secrets.migrate import MARKER, migrate
    from colonymind.secrets.vault import VaultUnavailable
    intentions, goals, *_ = unsealed_owner_state
    vault = goals.codec.vault
    monkeypatch.setattr(Codec, "seal", lambda self, value, owner: value)
    with pytest.raises(VaultUnavailable, match="remains unsealed"):
        migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert SYNTHETIC_TOKEN in owner_dump(goals)
    assert vault.marker(MARKER) is None


def test_interrupted_owner_constraint_upgrade_rolls_back_and_resumes(unsealed_owner_state, monkeypatch):
    from colonymind.secrets.migrate import MARKER, migrate
    intentions, goals, *_ = unsealed_owner_state
    with goals.connection(write=True) as db:
        db.execute("DELETE FROM owner_goal_schema WHERE version=2")
        db.execute("DROP TRIGGER owner_goal_receipts_no_delete")
    with goals.connection() as db:
        before = list(map(tuple, db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name")))
    connect = sqlite3.connect
    class Interrupted(sqlite3.Connection):
        def execute(self, sql, *args):
            result = super().execute(sql, *args)
            if sql == "INSERT INTO owner_goal_schema VALUES (2)":
                raise sqlite3.OperationalError("fixture interruption")
            return result
    vault = goals.codec.vault
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", lambda *args, **kwargs: connect(*args, **(kwargs | {"factory": Interrupted})))
        with pytest.raises(sqlite3.OperationalError, match="fixture interruption"):
            migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert vault.marker(MARKER) is None
    with goals.connection() as db:
        assert not db.execute("SELECT 1 FROM owner_goal_schema WHERE version=2").fetchone()
        assert list(map(tuple, db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name"))) == before
    migrate(intentions._state_dir, vault=vault, owner_id="owner", write_report=False)
    assert SYNTHETIC_TOKEN not in owner_dump(goals)
    assert vault.marker(MARKER)


def test_owner_sealing_uses_default_rules_even_with_capture_disabled(stores, monkeypatch):
    from colonymind.secrets import config
    _, goals = stores
    monkeypatch.setattr(config, "_CURRENT", config.SecretsSettings(capture_enabled=False, rules=()))
    root = create(goals, description=SYNTHETIC_TOKEN)
    assert root["sensitivity"] == "Secret"
    assert SYNTHETIC_TOKEN not in owner_dump(goals)


def test_sealed_owner_dry_run_does_not_open_the_vault(stores, monkeypatch):
    _, goals = stores
    create(goals, description=SYNTHETIC_TOKEN)
    before = owner_snapshot(goals)
    def unexpected(*args):
        pytest.fail("a dry run opened the vault")
    monkeypatch.setattr(goals.codec.vault, "resolve", unexpected)
    assert goals.seal_existing(goals.db_path, vault=goals.codec.vault, dry_run=True) == {}
    assert owner_snapshot(goals) == before


def test_generic_sealing_and_lineage_never_read_owner_tables(unsealed_owner_state, monkeypatch):
    from colonymind.secrets import migrate as migration
    intentions, goals, *_ = unsealed_owner_state
    before = owner_snapshot(goals)
    connect = migration._connect
    reads = []
    def guarded(path):
        db = connect(path)
        def authorize(action, table, *args):
            if action == sqlite3.SQLITE_READ and table.startswith("owner_goal_"):
                reads.append(table)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        db.set_authorizer(authorize)
        return db
    monkeypatch.setattr(migration, "_connect", guarded)
    rewriter = migration.StoreRewriter(intentions._state_dir, report=migration.Report())
    rewriter.walk(lambda text, row: "fixture")
    rewriter._repoint_database(intentions._db_path, [(SYNTHETIC_TOKEN, "fixture")])
    assert reads == []
    assert owner_snapshot(goals) == before


@pytest.fixture
async def client(stores):
    app = FastAPI()
    app.include_router(routes.router)
    async def store_dependency():
        return stores[1]
    async def verifier_dependency():
        return permit
    app.dependency_overrides[routes.get_owner_goals_store] = store_dependency
    app.dependency_overrides[get_action_verifier] = verifier_dependency
    app.state.authority = key_authority()
    app.state.audience = OWNER

    @app.middleware("http")
    async def auth(request, call_next):
        request.state.colonymind_authority = app.state.authority
        if app.state.audience is not None:
            request.state.owner_goal_audience = app.state.audience
        return await call_next(request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as result:
        result.test_app = app
        yield result


async def send(client, method, path=ROOT, body=None, key=None):
    return await client.request(method, path, json=body,
                                headers={"Idempotency-Key": key or str(uuid.uuid4())})


def code(response, status, expected):
    assert response.status_code == status, response.text
    assert response.json() == {"detail": {"code": expected}}


READ_AUDIENCES = [OWNER, *[Audience("owner", frozenset({"guest"}), False, tier)
                          for tier in ("Public", "Everyday", "Personal", "Private", "Secret")],
                  Audience("owner", frozenset({"owner", "guest"}), True, "Private"),
                  Audience("owner", frozenset({"owner", "guest"}), True, "Secret"),
                  Audience("owner", frozenset({"owner"}), False, "Private"),
                  Audience("owner", frozenset(), True, "Secret")]


def break_content(goals, monkeypatch, id, failure, *, proposal=False):
    from colonymind.owner_goals.sealing import PREFIX
    from colonymind.secrets.vault import VaultUnavailable
    with goals.connection() as db:
        table, column = ("owner_goal_proposals", "data") if proposal else ("owner_goal_records", "context")
        value = db.execute(f"SELECT {column} FROM {table} WHERE id=?", (id,)).fetchone()[0]
    assert value.startswith(PREFIX)
    ref = value[len(PREFIX):]
    vault = goals.codec.vault
    if failure == "forgotten":
        assert vault.forget(ref)
    resolve = vault.resolve
    attempts = []
    def spy(candidate):
        attempts.append(candidate)
        if failure == "unavailable" and candidate == ref:
            raise VaultUnavailable("fixture unavailable")
        return resolve(candidate)
    monkeypatch.setattr(vault, "resolve", spy)
    return attempts


@pytest.mark.parametrize("audience", READ_AUDIENCES)
@pytest.mark.parametrize("tier", ["Public", "Everyday", "Personal", "Private", "Secret"])
@pytest.mark.parametrize("failure", ["forgotten", "unavailable"])
async def test_http_decrypt_failure_is_filtered_before_opening_content(client, stores, monkeypatch, audience, tier, failure):
    _, goals = stores
    root = create(goals, description=SYNTHETIC_TOKEN, sensitivity=tier)
    attempts = break_content(goals, monkeypatch, root["id"], failure)
    client.test_app.state.audience = audience
    absent = await client.get(ROOT + "/absent")
    code(absent, 404, "owner_goal_not_found")
    for suffix in ("", "/subgoals", "/progress", "/artifacts", "/activity"):
        response = await client.get(ROOT + "/" + root["id"] + suffix)
        if audience.allows(tier):
            code(response, 503, "owner_goal_content_unavailable")
        else:
            assert (response.status_code, response.json()) == (absent.status_code, absent.json())
    listing = await client.get(ROOT)
    if audience.allows(tier):
        code(listing, 503, "owner_goal_content_unavailable")
        assert attempts
    else:
        assert listing.status_code == 200 and listing.json()["goals"] == []
        assert attempts == []
        assert goals.recall("owner", audience)["goals"] == []


@pytest.mark.parametrize("audience", READ_AUDIENCES)
@pytest.mark.parametrize("kind", ["child", "progress"])
@pytest.mark.parametrize("failure", ["forgotten", "unavailable"])
async def test_hidden_nested_content_is_never_decrypted(client, stores, monkeypatch, audience, kind, failure):
    _, goals = stores
    root = create(goals, sensitivity="Public")
    if kind == "child":
        child = create(goals, parent_id=root["id"], description=SYNTHETIC_TOKEN, sensitivity="Secret")
        id = child["id"]
    else:
        write(goals, "progress", PROGRESS | {"text": SYNTHETIC_TOKEN, "sensitivity": "Secret"}, id=root["id"])
        with goals.connection() as db:
            id = db.execute("SELECT id FROM owner_goal_records WHERE kind='owner_goal_progress'").fetchone()[0]
    attempts = break_content(goals, monkeypatch, id, failure)
    client.test_app.state.audience = audience
    for path in (ROOT, ROOT + "/" + root["id"], ROOT + "/" + root["id"] + "/subgoals",
                 ROOT + "/" + root["id"] + "/progress", ROOT + "/" + root["id"] + "/activity"):
        response = await client.get(path)
        relevant = not path.endswith("/subgoals") if kind == "progress" else not path.endswith(("/progress", "/activity"))
        if audience.owner_only and relevant:
            code(response, 503, "owner_goal_content_unavailable")
        else:
            assert response.status_code == 200
            assert SYNTHETIC_TOKEN not in response.text
    if not audience.owner_only:
        assert attempts == []


@pytest.mark.parametrize("audience", READ_AUDIENCES)
@pytest.mark.parametrize("failure", ["forgotten", "unavailable"])
async def test_unopenable_proposals_follow_owner_only_filter(client, stores, monkeypatch, audience, failure):
    _, goals = stores
    proposal = write(goals, "propose", {"kind": "goal", "target_id": None,
        "payload": {"title": "Glorp", "description": SYNTHETIC_TOKEN,
                    "category": "Zorb", "sensitivity": "Public"}})["proposal"]
    attempts = break_content(goals, monkeypatch, proposal["id"], failure, proposal=True)
    client.test_app.state.audience = audience
    response = await client.get(ROOT + "/proposals/" + proposal["id"])
    listing = await client.get(ROOT + "/proposals")
    if audience.owner_only:
        code(response, 503, "owner_goal_content_unavailable")
        code(listing, 503, "owner_goal_content_unavailable")
    else:
        absent = await client.get(ROOT + "/proposals/absent")
        assert (response.status_code, response.json()) == (absent.status_code, absent.json())
        code(response, 404, "owner_goal_proposal_not_found")
        assert listing.status_code == 200 and listing.json()["proposals"] == []
        assert attempts == []


@pytest.mark.parametrize("tier", ["Public", "Secret"])
def test_inherited_visibility_does_not_decrypt_unavailable_parent(stores, monkeypatch, tier):
    _, goals = stores
    root = create(goals, description=SYNTHETIC_TOKEN, sensitivity=tier)
    child = create(goals, parent_id=root["id"])
    write(goals, "progress", PROGRESS, id=child["id"])
    attempts = break_content(goals, monkeypatch, root["id"], "forgotten")
    guest = Audience("owner", frozenset({"guest"}))
    if tier == "Public":
        assert goals.read("owner", guest, id=child["id"])["goal"]["sensitivity"] == "Public"
    else:
        with pytest.raises(GoalError, match="owner_goal_not_found"):
            goals.read("owner", guest, id=child["id"])
        assert goals.recall("owner", guest)["goals"] == []
    assert attempts == []


def test_legacy_encrypted_visibility_is_backfilled_without_opening_hidden_reads(stores, monkeypatch):
    from colonymind.owner_goals import schema
    _, goals = stores
    root = create(goals, description=SYNTHETIC_TOKEN, sensitivity="Public")
    with goals.connection(write=True) as db:
        db.execute("DROP TABLE owner_goal_visibility")
        db.execute("DELETE FROM owner_goal_schema WHERE version=3")
    schema.initialize(stores[0]._db)
    guest = Audience("owner", frozenset({"guest"}))
    with monkeypatch.context() as patch:
        patch.setattr(goals.codec, "plaintext", lambda value: pytest.fail("hidden legacy content was decrypted"))
        with pytest.raises(GoalError, match="owner_goal_not_found"):
            goals.read("owner", guest, id=root["id"])
        assert goals.recall("owner", guest)["goals"] == []
    goals.seal_existing(goals.db_path, vault=goals.codec.vault)
    assert goals.read("owner", guest, id=root["id"])["goal"]["description"] == SYNTHETIC_TOKEN


def test_kinds_depth_revision_preview_and_engine_isolation(stores):
    intentions, goals = stores
    root = create(goals)
    children = [create(goals, parent_id=root["id"]) for _ in range(3)]
    full = goals.read("owner", OWNER, id=root["id"])["goal"]
    assert full["revision"] == 4
    assert full["sensitivity"] == "Secret"
    assert full["status"] == "active" and full["completed_at"] is None
    assert full["subgoals"]["next_offset"] == 1
    assert len(full["subgoals"]["items"]) == 1
    assert children[0]["subgoals"] == {"items": [], "limit": 1, "offset": 0, "next_offset": None}
    with pytest.raises(GoalError, match="invalid_parent"):
        create(goals, parent_id=children[0]["id"])
    with pytest.raises(GoalError, match="invalid_parent"):
        create(goals, parent_id="absent")
    assert intentions.intentions() == [] and intentions.list() == []
    assert intentions.get(root["id"]) is None
    assert intentions.get(children[0]["id"]) is None
    with goals.connection() as db:
        row = db.execute("SELECT status,expires_at FROM owner_goal_records WHERE id=?", (root["id"],)).fetchone()
        assert tuple(row) == ("recorded", None)


def test_progress_append_only_inheritance_and_explicit_overrides(stores):
    intentions, goals = stores
    root = create(goals)
    inherited = create(goals, parent_id=root["id"])
    explicit = create(goals, parent_id=root["id"], sensitivity="Personal")
    write(goals, "progress", PROGRESS, id=inherited["id"])
    write(goals, "progress", PROGRESS | {"text": "Blorp.", "sensitivity": "Public"}, id=inherited["id"])
    with goals.connection() as db:
        before = list(db.execute("SELECT id,context FROM owner_goal_records WHERE kind='owner_goal_progress' ORDER BY id"))
    root = goals.read("owner", OWNER, id=root["id"])["goal"]
    write(goals, "patch", {"expected_revision": root["revision"], "sensitivity": "Everyday"}, id=root["id"])
    assert goals.read("owner", OWNER, id=inherited["id"])["goal"]["sensitivity"] == "Everyday"
    assert goals.read("owner", OWNER, id=explicit["id"])["goal"]["sensitivity"] == "Personal"
    entries = goals.read("owner", OWNER, id=inherited["id"], collection="progress")["items"]
    assert {p["sensitivity"] for p in entries} == {"Everyday", "Public"}
    with goals.connection() as db:
        after = list(db.execute("SELECT id,context FROM owner_goal_records WHERE kind='owner_goal_progress' ORDER BY id"))
    assert [tuple(r) for r in before] == [tuple(r) for r in after]
    assert all(intentions.get(entry["id"]) is None for entry in entries)
    assert all(p["value"] is None and p["unit"] is None for p in entries)
    assert len(goals.read("owner", OWNER, id=inherited["id"], collection="activity")["items"]) == 2


def test_receipts_replay_conflict_scope_and_restart(stores):
    intentions, goals = stores
    request = ctx("create", CREATE)
    original = goals.mutate(request, "create", permit)
    id = original[1]["goal"]["id"]
    write(goals, "delete", {"expected_revision": 1}, id=id)
    reopened = OwnerGoalsStore(intentions)
    assert reopened.mutate(request, "create", permit) == original
    for changed in (replace(request, body=CREATE | {"title": "Zim"}),
                    replace(request, path=ROOT + "/absent", method="PATCH")):
        changed = replace(changed, request_hash=hashlib.sha256(dumps([changed.method, changed.path, changed.body]).encode()).hexdigest())
        with pytest.raises(GoalError, match="idempotency_conflict"):
            reopened.mutate(changed, "create", permit)
    second = replace(request, principal_id="other-agent")
    assert reopened.mutate(second, "create", permit)[1]["goal"]["id"] != id
    with pytest.raises(GoalError, match="not_owner"):
        reopened.mutate(request, "create", lambda c, **kw: VerifiedAction(c.owner_id, c.principal_id, "wrong", "requested_write"))


def test_concurrent_same_operation_has_one_effect(stores):
    _, goals = stores
    request = ctx("create", CREATE)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: goals.mutate(request, "create", permit), range(8)))
    assert all(result == results[0] for result in results)
    assert len(goals.read("owner", OWNER)["goals"]) == 1


def test_failure_rolls_back_mutation_audit_and_receipt(stores, monkeypatch):
    _, goals = stores
    root = create(goals)
    with goals.connection() as db:
        before = db.execute("SELECT count(*) FROM owner_goal_history").fetchone()[0]
    monkeypatch.setattr(goals, "_project", lambda *args: (_ for _ in ()).throw(GoalError(503, "owner_goals_unavailable")))
    request = ctx("progress", PROGRESS, id=root["id"])
    with pytest.raises(GoalError, match="owner_goals_unavailable"):
        goals.mutate(request, "progress", permit, id=root["id"])
    with goals.connection() as db:
        assert db.execute("SELECT count(*) FROM owner_goal_records WHERE kind='owner_goal_progress'").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM owner_goal_history").fetchone()[0] == before
        assert db.execute("SELECT count(*) FROM owner_goal_receipts WHERE operation_id=?", (request.operation_id,)).fetchone()[0] == 0
        assert json.loads(db.execute("SELECT context FROM owner_goal_records WHERE id=?", (root["id"],)).fetchone()[0])["revision"] == 1


def test_tombstone_retains_progress_audit_proposals_receipts_and_status(stores):
    _, goals = stores
    root = create(goals)
    child = create(goals, parent_id=root["id"])
    write(goals, "progress", PROGRESS, id=child["id"])
    proposal = write(goals, "propose", {"kind": "progress", "target_id": child["id"], "payload": {k: v for k, v in PROGRESS.items() if k != "source_id"}})["proposal"]
    root = goals.read("owner", OWNER, id=root["id"])["goal"]
    write(goals, "delete", {"expected_revision": root["revision"]}, id=root["id"])
    assert goals.read("owner", OWNER)["goals"] == []
    for id in (root["id"], child["id"]):
        with pytest.raises(GoalError, match="owner_goal_not_found"):
            goals.read("owner", OWNER, id=id)
        with pytest.raises(GoalError, match="owner_goal_not_found"):
            write(goals, "progress", PROGRESS, id=id)
    assert goals.read("owner", OWNER, id=proposal["id"], proposals=True)["proposal"] == proposal
    with goals.connection() as db:
        assert db.execute("SELECT count(*) FROM owner_goal_records WHERE kind='owner_goal_progress'").fetchone()[0] == 1
        data = goals._row(db, "owner", root["id"], live=False)
        assert data["status"] == "active" and data["_deleted_by"] == "agent"
        assert db.execute("SELECT count(*) FROM owner_goal_history").fetchone()[0] >= 6
        assert db.execute("SELECT count(*) FROM owner_goal_receipts").fetchone()[0] == 5


@pytest.mark.parametrize("kind", ["goal", "progress"])
def test_proposal_exact_payload_target_and_settlement(stores, kind):
    _, goals = stores
    root = create(goals)
    payload = ({"at": PROGRESS["at"], "text": "Zibble."} if kind == "progress"
               else {"title": "Zibble", "description": "Fram.", "category": "Zorb", "parent_id": root["id"]})
    proposal = write(goals, "propose", {"kind": kind, "target_id": root["id"] if kind == "progress" else None,
                                        "payload": payload, "source_id": "chat:suggestion"})["proposal"]
    assert proposal["state"] == "proposed" and proposal["target_revision"] == 1
    assert proposal["payload"]["sensitivity"] == "Secret"
    assert proposal["confirmed_at"] is None and proposal["result_goal_id"] is None
    with pytest.raises(GoalError, match="revision_conflict"):
        write(goals, "confirm", {"expected_revision": 0}, id=proposal["id"])
    request = ctx("confirm", {"expected_revision": 1, "source_id": "chat:confirmation"}, id=proposal["id"])
    status, result = goals.mutate(request, "confirm", permit, id=proposal["id"])
    assert status == 200 and result["proposal"]["revision"] == 2
    assert result["proposal"]["state"] == "confirmed" and result["proposal"]["payload"] == proposal["payload"]
    assert result["proposal"]["result_goal_id"] == result["goal"]["id"]
    assert goals.mutate(request, "confirm", permit, id=proposal["id"]) == (status, result)
    with pytest.raises(GoalError, match="proposal_already_confirmed"):
        write(goals, "confirm", {"expected_revision": 1}, id=proposal["id"])
    accepted = result["goal"] if kind == "goal" else result["goal"]["progress"]["items"][0]
    assert accepted["title" if kind == "goal" else "text"] == "Zibble" + ("." if kind == "progress" else "")


@pytest.mark.parametrize("deleted", [False, True])
def test_confirmation_refuses_stale_or_deleted_target(stores, deleted):
    _, goals = stores
    root = create(goals)
    proposal = write(goals, "propose", {"kind": "progress", "target_id": root["id"], "payload": {"at": PROGRESS["at"], "text": "Zib."}})["proposal"]
    write(goals, "delete" if deleted else "patch", {"expected_revision": 1, **({} if deleted else {"title": "Frob"})}, id=root["id"])
    with pytest.raises(GoalError, match="owner_goal_not_found" if deleted else "revision_conflict"):
        write(goals, "confirm", {"expected_revision": 1}, id=proposal["id"])
    assert goals.read("owner", OWNER, id=proposal["id"], proposals=True)["proposal"]["state"] == "proposed"


def test_muted_completed_ancestor_and_deleted_suppress_outreach(stores):
    _, goals = stores
    root = create(goals)
    child = create(goals, parent_id=root["id"])
    assert goals.outreach_eligible("owner", child["id"])
    for edits in ({"muted": True}, {"muted": False, "status": "completed"}, {"status": "active"}):
        current = goals.read("owner", OWNER, id=root["id"])["goal"]
        result = write(goals, "patch", {"expected_revision": current["revision"], **edits}, id=root["id"])["goal"]
        assert goals.outreach_eligible("owner", child["id"]) == (not result["muted"] and result["status"] == "active")
        assert (result["completed_at"] is None) == (result["status"] == "active")
    write(goals, "delete", {"expected_revision": result["revision"]}, id=root["id"])
    assert not goals.outreach_eligible("owner", child["id"])


def test_recall_secret_excluded_from_guest_and_mixed_audiences(stores):
    _, goals = stores
    secret = create(goals)
    public = create(goals, sensitivity="Public")
    for audience in (Audience("owner", frozenset({"guest"})), Audience("owner", frozenset({"owner", "guest"}), True, "Private")):
        result = goals.recall("owner", audience)
        assert [g["id"] for g in result["goals"]] == [public["id"]]
        with pytest.raises(GoalError, match="owner_goal_not_found"):
            goals.read("owner", audience, id=secret["id"])
    assert len(goals.recall("owner", OWNER)["goals"]) == 2


@pytest.mark.parametrize("body", [CREATE, CREATE | {"by": "owner"}, CREATE | {"sender": "owner"}])
def test_default_verifier_never_authorizes(stores, body):
    _, goals = stores
    with pytest.raises(GoalError, match="owner_turn_required|invalid_request"):
        write(goals, "create", body, verifier=verify_owner_goal_action)
    assert goals.read("owner", OWNER)["goals"] == []


def test_verifier_purpose_identity_tier_and_exact_effects(stores):
    _, goals = stores
    for change in ({"purpose": "proposal_only"}, {"owner_id": "foreign"}, {"principal_id": "foreign"}, {"evidence_id": ""}):
        def bad(context, **kwargs):
            return replace(permit(context, **kwargs), **change)
        with pytest.raises(GoalError, match="not_owner"):
            write(goals, "create", CREATE, verifier=bad)
    def no_tier(context, **kwargs):
        return replace(permit(context, **kwargs), explicit_tier=False)
    with pytest.raises(GoalError, match="not_owner"):
        write(goals, "create", CREATE | {"sensitivity": "Public"}, verifier=no_tier)
    observed = []
    def capture(context, **kwargs):
        observed.append(kwargs["resolved_action"])
        assert kwargs["connection"].in_transaction
        return permit(context, **kwargs)
    create_body = dict(CREATE)
    write(goals, "create", create_body, verifier=capture)
    assert len(observed) == 2 and observed[0]["phase"] == "authorize"
    assert observed[1]["payload"]["sensitivity"] == "Secret"


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "limit=true", "offset=-1", "offset=+1", "offset=1.0",
                                   "offset=9007199254740992", "limit=1&limit=2", "x=1", "offset=١", "offset=%31", "limit="])
async def test_route_pagination_rejects_noncontract_queries(client, query):
    code((await client.get(ROOT + "?" + query)), 422, "invalid_request")


@pytest.mark.parametrize("edit", [{"title": " "}, {"title": "x" * 201}, {"category": ""}, {"muted": 1},
                                  {"sensitivity": "secret"}, {"parent_id": 1}, {"source_id": "a\x00b"},
                                  {"source_id": " "}, {"source_id": 4}, {"operation_id": "x"},
                                  {"title": "\ud800"}, {"description": "x\r"}])
def test_strict_create_validation(edit):
    with pytest.raises(GoalError, match="invalid_request"):
        validate("create", CREATE | edit)


@pytest.mark.parametrize("value", [True, "1", float("inf"), float("nan")])
def test_progress_numeric_validation(value):
    with pytest.raises(GoalError, match="invalid_request"):
        validate("progress", PROGRESS | {"value": value})


@pytest.mark.parametrize("stamp", ["2026-01-02t03:04:05Z", "2026-01-02T03:04:60Z", "2026-02-29T03:04:05Z",
                                   "2026-01-02T24:04:05Z", "2026-01-02T03:04:05+24:00", "2026-01-02T03:04:05+00:60",
                                   "20260102T03:04:05Z", "2026-01-02 03:04:05Z", "0000-01-02T03:04:05Z",
                                   "2026-01-02T03:04:05.Z", "2026-01-02T03:04:05+00:00:01", 0])
def test_timestamp_invalid(stamp):
    with pytest.raises(GoalError, match="invalid_request"):
        timestamp(stamp)


@pytest.mark.parametrize("stamp", ["0001-01-01T00:00:00+23:59", "9999-12-31T23:59:59-23:59",
                                   "2024-02-29T01:02:03.123456789Z", "2026-01-02T03:04:05-04:30"])
def test_timestamp_valid(stamp):
    assert timestamp(stamp)


async def test_route_shapes_static_precedence_source_and_real_errors(client):
    response = (await send(client, "POST", body=CREATE))
    assert response.status_code == 201, response.text
    goal = response.json()["goal"]
    assert set(goal) == {"id", "revision", "title", "description", "category", "status", "parent_id", "muted", "sensitivity",
                         "created_at", "updated_at", "completed_at", "subgoals", "progress", "artifacts", "activity"}
    assert (await client.get(ROOT)).json()["goals"] == [goal]
    assert (await client.get(ROOT + "/proposals")).json() == {"proposals": [], "limit": 50, "offset": 0, "next_offset": None}
    for collection in ("subgoals", "progress", "artifacts", "activity"):
        assert (await client.get(ROOT + "/" + goal["id"] + "/" + collection)).json() == {"items": [], "limit": 50, "offset": 0, "next_offset": None}
    assert (await send(client, "POST", ROOT + "/" + goal["id"] + "/progress", PROGRESS)).status_code == 201
    code((await send(client, "PATCH", ROOT + "/" + goal["id"], {"expected_revision": True, "title": "Zib"})), 422, "invalid_request")
    code((await send(client, "PATCH", ROOT + "/" + goal["id"], {"expected_revision": 1, "title": "Zib"})), 409, "revision_conflict")
    code((await client.get(ROOT + "/absent")), 404, "owner_goal_not_found")
    code((await client.get(ROOT + "/proposals/absent")), 404, "owner_goal_proposal_not_found")
    code((await client.get(ROOT + "/" + goal["id"] + "?limit=1")), 422, "invalid_request")
    code((await send(client, "POST", ROOT + "?offset=0", CREATE)), 422, "invalid_request")


async def test_route_default_deny_missing_idempotency_duplicate_json_and_unknown_fields(client):
    client.test_app.dependency_overrides.pop(get_action_verifier)
    code((await send(client, "POST", body=CREATE)), 422, "owner_turn_required")
    code((await send(client, "POST", body={k: v for k, v in CREATE.items() if k != "source_id"})), 422, "owner_turn_required")
    code((await client.post(ROOT, json=CREATE)), 422, "invalid_request")
    code((await send(client, "POST", body=CREATE, key="not-uuid")), 422, "invalid_request")
    raw = '{"title":"Zib","title":"Zob","description":"","category":"Zorb"}'
    code((await client.post(ROOT, content=raw, headers={"Idempotency-Key": str(uuid.uuid4())})), 422, "invalid_request")
    for body in (CREATE | {"sender": "owner"}, CREATE | {"by": "owner"}, CREATE | {"operation_id": str(uuid.uuid4())}):
        code((await send(client, "POST", body=body)), 422, "invalid_request")
    client.test_app.state.authority = anonymous_authority()
    code((await client.get(ROOT)), 403, "not_owner")
    code((await send(client, "POST", body=CREATE)), 403, "not_owner")


async def test_route_replay_survives_deletion_and_changed_path_conflicts(client):
    key = str(uuid.uuid4())
    response = (await send(client, "POST", body=CREATE, key=key))
    id = response.json()["goal"]["id"]
    assert (await send(client, "DELETE", ROOT + "/" + id, {"expected_revision": 1})).status_code == 200
    assert (await send(client, "POST", body=CREATE, key=key)).json() == response.json()
    code((await send(client, "POST", body=CREATE | {"title": "Zob"}, key=key)), 409, "idempotency_conflict")
    code((await send(client, "POST", ROOT + "/absent/progress", PROGRESS, key=key)), 409, "idempotency_conflict")


async def test_route_proposal_confirm_and_tier_read_controls(client):
    proposal = (await send(client, "POST", ROOT + "/proposals", {"kind": "goal", "target_id": None,
                    "payload": {k: v for k, v in CREATE.items() if k != "source_id"}, "source_id": "chat:proposal"})).json()["proposal"]
    assert (await client.get(ROOT + "/proposals/" + proposal["id"])).json()["proposal"] == proposal
    response = (await send(client, "POST", ROOT + "/proposals/" + proposal["id"] + "/confirm", {"expected_revision": 1, "source_id": "chat:confirm"}))
    assert response.status_code == 200, response.text
    assert response.json()["proposal"]["state"] == "confirmed"
    client.test_app.state.audience = Audience("owner", frozenset({"owner", "guest"}), True, "Private")
    assert (await client.get(ROOT)).json()["goals"] == []
    assert (await client.get(ROOT + "/proposals")).json()["proposals"] == []
    code((await client.get(ROOT + "/proposals/" + proposal["id"])), 404, "owner_goal_proposal_not_found")


async def test_no_progress_edit_or_delete_route(client):
    goal = (await send(client, "POST", body=CREATE)).json()["goal"]
    id = goal["id"]
    progress = (await send(client, "POST", ROOT + "/" + id + "/progress", PROGRESS)).json()["goal"]["progress"]["items"][0]
    for method in ("PATCH", "DELETE"):
        assert (await send(client, method, ROOT + "/" + id + "/progress/" + progress["id"], {})).status_code == 404


class Library:
    def __init__(self):
        self.item = LibraryItem("namespace", "item", "object-v1", "acl-v1", "Public", "owner", frozenset())

    def get_item(self, ref):
        return replace(self.item, id=ref) if self.item is not None else None


class Cron:
    def __init__(self):
        self.eligible = True
        self.available = True
        self.rows = []
        self.pages = []

    def resolve_owner_artifact(self, owner_id, ref, audience):
        if not self.available:
            raise OSError()
        if not self.eligible:
            raise GoalError(404, "artifact_not_found")
        return ArtifactHandle(owner_id, "cron_job", ref, "profile", "object-v1", "delivery-v1",
                              "Secret", "Glorp", True, True)

    def list_owner_job_runs(self, owner, handle, audience, limit, offset):
        self.pages.append(offset)
        if not self.available:
            raise OSError()
        return page(iter(self.rows[offset:]), limit, offset)


def test_library_authority_ownership_acl_tier_and_versions(stores):
    _, goals = stores
    library = Library()
    goals.resolver = ArtifactResolver(library=library)
    handle = goals.resolver.resolve_owner_artifact("owner", "library_item", "item", OWNER)
    assert handle.namespace_id == "namespace" and handle.evidence_version == "acl-v1"
    assert handle.sensitivity == "Public"
    root = create(goals)
    result = write(goals, "attach", {"type": "library_item", "ref": "item", "label": "Zib", "source_id": "chat:attach"}, id=root["id"])["goal"]
    artifact = result["artifacts"]["items"][0]
    assert artifact["cron"] is None and artifact["sensitivity"] == "Secret"
    assert artifact["label"] == "Zib"
    library.item = replace(library.item, owner_id="foreign", owner_acl=frozenset({"owner"}), sensitivity="Private", acl_version="acl-v2")
    assert goals.resolver.resolve_owner_artifact("owner", "library_item", "item", OWNER).evidence_version == "acl-v2"
    library.item = replace(library.item, owner_acl=frozenset())
    with pytest.raises(GoalError, match="artifact_not_found"):
        goals.resolver.resolve_owner_artifact("owner", "library_item", "item", OWNER)
    with pytest.raises(GoalError, match="owner_goals_unavailable"):
        goals.read("owner", OWNER, id=root["id"])
    # Removing a relation does not need unavailable output or current object ownership.
    result = write(goals, "detach", {"expected_revision": result["revision"]}, id=root["id"], type="library_item", ref="item")["goal"]
    assert result["artifacts"]["items"] == []
    assert library.item.owner_id == "foreign"


def test_missing_and_incomplete_library_authority_fail_closed(stores):
    _, goals = stores
    with pytest.raises(GoalError, match="owner_goals_unavailable"):
        goals.resolver.resolve_owner_artifact("owner", "library_item", "item", OWNER)
    library = Library()
    resolver = ArtifactResolver(library=library)
    for change in ({"owner_id": None}, {"acl_version": ""}, {"namespace_id": ""}, {"sensitivity": "other"}):
        library.item = replace(Library().item, **change)
        with pytest.raises(GoalError, match="owner_goals_unavailable"):
            resolver.resolve_owner_artifact("owner", "library_item", "item", OWNER)
    library.item = None
    with pytest.raises(GoalError, match="artifact_not_found"):
        resolver.resolve_owner_artifact("owner", "library_item", "item", OWNER)


def test_library_audience_applies_linked_acl_and_tier():
    library = Library()
    resolver = ArtifactResolver(library=library)
    mixed = Audience("owner", frozenset({"owner", "guest"}), True, "Private")
    library.item = replace(library.item, sensitivity="Secret")
    with pytest.raises(GoalError, match="artifact_not_found"):
        resolver.resolve_owner_artifact("owner", "library_item", "item", mixed)
    library.item = replace(library.item, sensitivity="Public", readers=frozenset({"owner"}))
    with pytest.raises(GoalError, match="artifact_not_found"):
        resolver.resolve_owner_artifact("owner", "library_item", "item", mixed)


def run(id, at="2026-01-02T03:04:05Z", ref="job"):
    return {"id": id, "source_id": "cron:profile:" + ref + ":" + id, "kind": "cron_run", "at": at,
            "title": "Glorp", "summary": "Blim.", "artifact_ref": ref, "sensitivity": "Secret"}


def test_cron_merge_revalidation_detach_and_tombstone_never_change_job(stores):
    _, goals = stores
    cron = Cron()
    cron.rows = [run("run_a", "2026-01-04T00:00:00Z"), run("run_b", "2026-01-01T00:00:00Z")]
    goals.resolver = ArtifactResolver(cron=cron)
    root = create(goals)
    attached = write(goals, "attach", {"type": "cron_job", "ref": "job", "label": "Glorp"}, id=root["id"])["goal"]
    assert attached["artifacts"]["items"][0]["cron"] == {"enabled": True, "recurring": True}
    assert attached["activity"]["next_offset"] == 1
    result = write(goals, "progress", PROGRESS, id=root["id"])["goal"]
    activity = goals.read("owner", OWNER, id=root["id"], collection="activity", limit=1, offset=1)
    assert activity["items"][0]["kind"] == "progress" and activity["next_offset"] == 2
    assert goals.read("owner", OWNER, id=root["id"], collection="activity", offset=2)["items"][0]["id"] == "run_b"
    cron.available = False
    with pytest.raises(GoalError, match="owner_goals_unavailable"):
        goals.read("owner", OWNER, id=root["id"])
    write(goals, "delete", {"expected_revision": result["revision"]}, id=root["id"])
    assert len(cron.rows) == 2 and cron.eligible


def test_failed_artifact_projection_never_commits_attachment(stores):
    _, goals = stores
    cron = Cron()
    cron.rows = [{"bad": "output"}]
    goals.resolver = ArtifactResolver(cron=cron)
    root = create(goals)
    with pytest.raises(GoalError, match="owner_goals_unavailable"):
        write(goals, "attach", {"type": "cron_job", "ref": "job", "label": "Glorp"}, id=root["id"])
    with goals.connection() as db:
        assert db.execute("SELECT count(*) FROM owner_goal_artifacts").fetchone()[0] == 0
    assert goals.read("owner", OWNER, id=root["id"])["goal"]["revision"] == 1


async def test_route_artifacts_resolver_detach_and_both_verifier_phases(client, stores):
    _, goals = stores
    goals.resolver = ArtifactResolver(library=Library())
    called = []
    def spy(context, **kwargs):
        called.append((context.body, kwargs["resolved_action"]))
        return permit(context, **kwargs)
    async def verifier_dependency():
        return spy
    client.test_app.dependency_overrides[get_action_verifier] = verifier_dependency
    id = (await send(client, "POST", body=CREATE)).json()["goal"]["id"]
    result = (await send(client, "POST", ROOT + "/" + id + "/artifacts",
                  {"type": "library_item", "ref": "item", "label": "Glorp", "source_id": "chat:attach"}))
    assert result.status_code == 200
    assert len(called) == 4 and called[-1][1]["artifact_binding"]["owner_id"] == "owner"
    result = (await send(client, "DELETE", ROOT + "/" + id + "/artifacts/library_item/item",
                  {"expected_revision": result.json()["goal"]["revision"], "source_id": "chat:detach"}))
    assert result.status_code == 200 and result.json()["goal"]["artifacts"]["items"] == []
    assert len(called) == 6


def test_pagination_past_ten_thousand_safe_edges_and_deterministic_ties(stores):
    _, goals = stores
    root = create(goals)
    with goals.connection(write=True) as db:
        for i in range(10005):
            data = {"id": f"progress_{i:05}", "parent_id": root["id"], "at": PROGRESS["at"],
                    "text": "Zib.", "value": None, "unit": None, "sensitivity": "Secret",
                    "_tier_override": False, "_sort_at": timestamp(PROGRESS["at"])}
            goals._insert(db, "owner", "agent", "owner_goal_progress", data, "chat:fixture")
    for collection in ("progress", "activity"):
        result = goals.read("owner", OWNER, id=root["id"], collection=collection, offset=10000, limit=2)
        assert [row["id"] for row in result["items"]] == ["progress_10000", "progress_10001"]
        assert result["next_offset"] == 10002
        end = goals.read("owner", OWNER, id=root["id"], collection=collection, offset=10004, limit=2)
        assert len(end["items"]) == 1 and end["next_offset"] is None
        assert goals.read("owner", OWNER, id=root["id"], collection=collection, offset=MAX_INTEGER)["items"] == []
    assert pagination([("offset", str(MAX_INTEGER)), ("limit", "001")]) == {"limit": 1, "offset": MAX_INTEGER}


def test_goal_and_proposal_collections_have_no_five_hundred_cap(stores):
    _, goals = stores
    root = create(goals)
    template = {k: v for k, v in root.items() if k not in {"subgoals", "progress", "artifacts", "activity"}}
    with goals.connection(write=True) as db:
        for i in range(10002):
            data = template | {"id": f"goal_{i:05}", "_tier_override": False}
            goals._insert(db, "owner", "agent", "owner_goal", data, "chat:fixture")
            proposal = {"id": f"proposal_{i:05}", "sensitivity": "Secret", "created_at": root["created_at"]}
            db.execute("INSERT INTO owner_goal_proposals VALUES (?,?,?,?,?,?)", (proposal["id"], "owner", dumps(proposal), 0, None, root["created_at"]))
    assert len(goals.read("owner", OWNER, offset=500, limit=100)["goals"]) == 100
    for proposals in (True, False):
        result = goals.read("owner", OWNER, offset=10000, limit=1, proposals=proposals)
        assert len(result["proposals" if proposals else "goals"]) == 1
        assert result["next_offset"] == 10001


def test_original_runs_paginate_beyond_ten_thousand_and_merge_offset(stores):
    _, goals = stores
    cron = Cron()
    cron.rows = [run(f"run_{i:05}") for i in range(10003)]
    goals.resolver = ArtifactResolver(cron=cron)
    root = create(goals)
    write(goals, "attach", {"type": "cron_job", "ref": "job", "label": "Glorp"}, id=root["id"])
    result = goals.read("owner", OWNER, id=root["id"], collection="activity", limit=2, offset=10000)
    assert [r["id"] for r in result["items"]] == ["run_10000", "run_10001"]
    assert result["next_offset"] == 10002 and 10000 in cron.pages


def test_byte_budget_reduces_page_without_empty_nonterminal_page():
    values = [{"blob": "🫧" * 30000, "id": str(i)} for i in range(20)]
    result = page(iter(values), 20, 0)
    assert 0 < len(result["items"]) < 20 and result["next_offset"] == len(result["items"])
    assert len(dumps(result).encode()) <= RESPONSE_BYTES
    with pytest.raises(GoalError, match="owner_goals_unavailable"):
        page(iter([{"blob": "x" * RESPONSE_BYTES}]), 1, 0)


def test_progress_timestamp_order_uses_exact_instant_and_fraction(stores):
    _, goals = stores
    root = create(goals)
    for at in ("2026-01-02T03:04:05.0000000001Z", "2026-01-02T04:04:05+01:00", "2026-01-02T03:04:05.0000000002Z"):
        write(goals, "progress", PROGRESS | {"at": at}, id=root["id"])
    rows = goals.read("owner", OWNER, id=root["id"], collection="progress")["items"]
    assert [r["at"] for r in rows] == ["2026-01-02T03:04:05.0000000002Z", "2026-01-02T03:04:05.0000000001Z", "2026-01-02T04:04:05+01:00"]


def test_inherited_tier_change_invalidates_child_confirmation_and_filters_proposal(stores):
    _, goals = stores
    root = create(goals, sensitivity="Public")
    child = create(goals, parent_id=root["id"])
    proposal = write(goals, "propose", {"kind": "progress", "target_id": child["id"],
                                        "payload": {"at": PROGRESS["at"], "text": "Zib."}})["proposal"]
    audience = Audience("owner", frozenset({"guest"}))
    assert goals.read("owner", audience, proposals=True)["proposals"] == []
    root = goals.read("owner", OWNER, id=root["id"])["goal"]
    write(goals, "patch", {"expected_revision": root["revision"], "sensitivity": "Secret"}, id=root["id"])
    assert goals.read("owner", audience, proposals=True)["proposals"] == []
    with pytest.raises(GoalError, match="revision_conflict"):
        write(goals, "confirm", {"expected_revision": 1}, id=proposal["id"])
    retained = goals.read("owner", OWNER, id=proposal["id"], proposals=True)["proposal"]
    assert retained["payload"] == proposal["payload"] and retained["sensitivity"] == "Secret"


def test_lower_tier_goal_filters_linked_secret_metadata_before_offset(stores):
    _, goals = stores
    library = Library()
    library.item = replace(library.item, sensitivity="Secret")
    goals.resolver = ArtifactResolver(library=library)
    root = create(goals, sensitivity="Public")
    write(goals, "attach", {"type": "library_item", "ref": "item", "label": "Zib"}, id=root["id"])
    audience = Audience("owner", frozenset({"guest"}))
    result = goals.read("owner", audience, id=root["id"])["goal"]
    assert result["artifacts"]["items"] == [] and result["activity"]["items"] == []
    assert result["sensitivity"] == "Public"


def test_progress_and_audit_cannot_be_modified_or_pruned(stores):
    from datetime import datetime, timezone
    import sqlite3
    intentions, goals = stores
    root = create(goals)
    write(goals, "progress", PROGRESS, id=root["id"])
    with goals.connection() as db:
        progress_id = db.execute("SELECT id FROM owner_goal_records WHERE kind='owner_goal_progress'").fetchone()[0]
    for query, parameters in (("UPDATE owner_goal_records SET context='{}' WHERE id=?", (progress_id,)),
                               ("DELETE FROM owner_goal_records WHERE id=?", (root["id"],)),
                               ("DELETE FROM owner_goal_history WHERE initiative_id=?", (root["id"],))):
        with pytest.raises(sqlite3.IntegrityError), goals.connection(write=True) as db:
            db.execute(query, parameters)
    assert intentions.delete_old(["recorded"], datetime(9999, 1, 1, tzinfo=timezone.utc)) == 0
    assert intentions.prune_intentions(datetime(9999, 1, 1, tzinfo=timezone.utc)) == {}
    assert intentions.count() == 0
    assert len(goals.read("owner", OWNER, id=root["id"], collection="progress")["items"]) == 1


@pytest.mark.asyncio
async def test_generic_routes_and_audit_cannot_bypass_owner_reader(stores, monkeypatch):
    from colonymind.api.routers import host
    from colonymind.mind import audit
    _, goals = stores
    root = create(goals)
    monkeypatch.setattr(host, "_initiative_store", stores[0])
    for call in (lambda: host.get_initiative(root["id"]), lambda: host.cancel_initiative(root["id"]),
                 lambda: host.respond_to_initiative(root["id"], action="approved")):
        with pytest.raises(Exception) as error:
            await call()
        assert error.value.status_code == 404
    assert audit.why(stores[0], root["id"]) is None
    assert audit.log(stores[0], kind=["owner_goal"]) == []
    assert goals.read("owner", OWNER, id=root["id"])["goal"]["revision"] == 1


def test_changed_invalid_body_with_existing_key_still_conflicts(stores):
    _, goals = stores
    request = ctx("create", CREATE)
    goals.mutate(request, "create", permit)
    changed = ctx("create", {"unknown": "Zib"}, key=request.operation_id)
    with pytest.raises(GoalError, match="idempotency_conflict"):
        goals.mutate(changed, "create", permit)


def test_duplicate_keys_nonfinite_body_and_size_are_rejected():
    for raw in (b'{"a": {"x":1,"x":2}}', b'{"a":NaN}', b'{"a":Infinity}', b'[]', b'{}x',
                b'{"a":"\\ud800"}', b'{"a":' + b'1' * 17000 + b'}'):
        with pytest.raises(GoalError, match="invalid_request"):
            loads(raw)


def test_owner_and_audience_scope_do_not_come_from_body(stores):
    _, goals = stores
    root = create(goals)
    foreign = Audience("foreign", frozenset({"foreign"}), True)
    assert goals.read("foreign", foreign)["goals"] == []
    with pytest.raises(GoalError, match="owner_goal_not_found"):
        goals.read("foreign", foreign, id=root["id"])
    with pytest.raises(GoalError, match="invalid_parent"):
        body = CREATE | {"parent_id": root["id"]}
        goals.mutate(ctx("create", body, owner="foreign"), "create", permit)


async def test_default_verifier_refuses_forged_capability(client):
    client.test_app.dependency_overrides.pop(get_action_verifier)
    response = (await client.post(ROOT, json=CREATE,
                           headers={"Idempotency-Key": str(uuid.uuid4()), "Owner-Goal-Action": "A" * 43}))
    code(response, 403, "not_owner")


async def test_read_requires_server_bound_audience_even_with_owner_credential(client):
    client.test_app.state.audience = None
    code((await client.get(ROOT)), 503, "owner_goals_unavailable")


async def test_every_mutation_route_denies_without_owner_gate(client, stores):
    stores[1].resolver = ArtifactResolver(library=Library())
    id = (await send(client, "POST", body=CREATE)).json()["goal"]["id"]
    (await send(client, "POST", ROOT + "/" + id + "/artifacts", {"type": "library_item", "ref": "item", "label": "Zib"}))
    proposal_body = {"kind": "progress", "target_id": id, "payload": {"at": PROGRESS["at"], "text": "Zib."},
                     "source_id": "chat:proposal"}
    proposal = (await send(client, "POST", ROOT + "/proposals", proposal_body)).json()["proposal"]
    client.test_app.dependency_overrides.pop(get_action_verifier)
    cases = [("POST", ROOT, CREATE),
             ("PATCH", ROOT + "/" + id, {"expected_revision": 2, "title": "Zib", "source_id": "chat:patch"}),
             ("POST", ROOT + "/" + id + "/progress", PROGRESS),
             ("POST", ROOT + "/" + id + "/artifacts", {"type": "library_item", "ref": "item", "label": "Zib", "source_id": "chat:attach"}),
             ("DELETE", ROOT + "/" + id + "/artifacts/library_item/item", {"expected_revision": 2, "source_id": "chat:detach"}),
             ("DELETE", ROOT + "/" + id, {"expected_revision": 2, "source_id": "chat:delete"}),
             ("POST", ROOT + "/proposals", proposal_body),
             ("POST", ROOT + "/proposals/" + proposal["id"] + "/confirm", {"expected_revision": 1, "source_id": "chat:confirm"})]
    for method, path, body in cases:
        code((await send(client, method, path, body)), 422, "owner_turn_required")
    goal = (await client.get(ROOT + "/" + id)).json()["goal"]
    assert goal["revision"] == 2 and goal["progress"]["items"] == []
    assert (await client.get(ROOT + "/proposals/" + proposal["id"])).json()["proposal"]["state"] == "proposed"


# These manifests include IDs in bodies and query parameters as well as paths.
# Route coverage below makes a newly added ID handler require an oracle case.
GENERIC_ID_CASES = [
    ("GET", "/v1/host/initiatives/ideas", None),
    ("GET", "/v1/host/initiatives/ideas/state?ids={id}", None),
    ("POST", "/v1/host/initiatives/ideas/associate", {"cards": [
        {"unit_id": "{id}", "sources": [{"key": "commitment:{id}", "revision": "0" * 64}]}]}),
    ("GET", "/v1/host/initiatives/shares/{id}", None),
    ("GET", "/v1/host/initiatives/media/{id}", None),
    ("GET", "/v1/host/initiatives/{id}", None),
    ("POST", "/v1/host/initiatives/{id}/execute", {"mode": "full"}),
    ("POST", "/v1/host/initiatives/{id}/share", None),
    ("POST", "/v1/host/initiatives/{id}/feedback", {"direction": "up", "surface": "ideas"}),
    ("POST", "/v1/host/initiatives/{id}/feedback", {"direction": "down", "surface": "ideas"}),
    ("POST", "/v1/host/initiatives/{id}/engagement", {
        "receipt_id": "fixture", "kind": "open", "visible_fraction": 1,
        "surface": "ideas", "anchor": "fixture"}),
    ("POST", "/v1/host/initiatives/{id}/complete", {"agent_id": "agent"}),
    ("POST", "/v1/host/initiatives/{id}/fail", {"agent_id": "agent", "error_message": "Zib"}),
    ("PATCH", "/v1/host/initiatives/{id}/priority", {"priority": 1}),
    ("POST", "/v1/host/initiatives/{id}/retry", None),
    ("POST", "/v1/host/initiatives/{id}/context/refresh", None),
    ("DELETE", "/v1/host/initiatives/{id}", None),
    ("POST", "/v1/host/initiatives/{id}/respond", {"action": "approved"}),
    ("POST", "/v1/host/initiatives/{id}/respond", {"mode": "full", "origin": {"surface": "feed"}}),
    ("POST", "/v1/host/initiatives/{id}/respond", {
        "mode": "selected_items", "item_ids": ["fixture"], "origin": {"surface": "ideas"}}),
    ("POST", "/v1/mind/dispatch/{id}/bound", {"hermes_ref": "fixture"}),
    ("POST", "/v1/mind/outcome", {"id": "{id}", "status": "done", "outcome": "done"}),
    ("POST", "/v1/mind/outbox/{id}/sending", {}),
    ("POST", "/v1/mind/outbox/{id}/sent", {"result": "sent"}),
    ("POST", "/v1/mind/rate", {"id": "{id}", "verdict": "useful"}),
    ("GET", "/v1/mind/why/{id}", None),
    ("GET", "/v1/mind/log/{id}", None),
    ("GET", "/v1/mind/permit-asks/{id}", None),
    ("GET", "/v1/mind/vault-captures/{id}", None),
    ("POST", "/v1/mind/tool-origins", {}),
    ("POST", "/v1/mind/tool-origins/claim", {}),
    ("POST", "/v1/mind/owner-actions", {}),
    ("POST", "/v1/mind/permit-asks/{id}/answer", {"platform": "test", "contact_id": "guest", "choice": "yes"}),
    ("POST", "/v1/mind/asks/{id}/show", None),
    ("POST", "/v1/mind/asks/{id}/shown", {"platform": "test", "chat_id": "chat", "message_id": "message", "asks": ["{id}"]}),
    ("POST", "/v1/mind/asks/{id}/yes", {}),
    ("POST", "/v1/mind/asks/{id}/no", {}),
    ("POST", "/v1/mind/answers", {"platform": "test", "contact_id": "guest", "text": "Zib", "ledger": [{"kind": "outbox", "outbox_id": "{id}"}]}),
    ("POST", "/v1/mind/guard", {"tool": "fixture", "run": "guest", "task_id": "{id}"}),
    ("POST", "/v1/mind/lineage", {"child": "fixture-{id}", "parent": "{id}"}),
    ("POST", "/v1/mind/cron-origin", {"job_id": "fixture-{id}", "task_id": "{id}"}),
    ("GET", "/v1/mind/cron-origin?job_id={id}", None),
    ("POST", "/v1/mind/skills/used", {"skill": "fixture", "task_id": "{id}"}),
    ("POST", "/v1/mind/lessons/{id}/retire", {"reason": "Zib"}),
    ("POST", "/v1/mind/grants/{id}/revoke", {}),
    ("POST", "/v1/mind/allows/{id}/revoke", {}),
    ("POST", "/v1/mind/learning/pending/{id}/reject", {"reason": "Zib"}),
]
OWNER_ID_CASES = [
    ("GET", ROOT, None),
    ("GET", ROOT + "/{id}", None),
    ("GET", ROOT + "/{id}/subgoals", None),
    ("GET", ROOT + "/{id}/progress", None),
    ("GET", ROOT + "/{id}/artifacts", None),
    ("GET", ROOT + "/{id}/activity", None),
    ("GET", ROOT + "/proposals", None),
    ("GET", ROOT + "/proposals/{id}", None),
    ("POST", ROOT, CREATE | {"parent_id": "{id}"}),
    ("PATCH", ROOT + "/{id}", {"expected_revision": 0, "title": "Zib"}),
    ("PATCH", ROOT + "/{id}", {"expected_revision": True, "title": "Zib"}),
    ("DELETE", ROOT + "/{id}", {"expected_revision": 0}),
    ("POST", ROOT + "/{id}/progress", PROGRESS),
    ("POST", ROOT + "/{id}/artifacts", {"type": "library_item", "ref": "item", "label": "Zib"}),
    ("DELETE", ROOT + "/{id}/artifacts/library_item/item", {"expected_revision": 0}),
    ("POST", ROOT + "/proposals", {"kind": "progress", "target_id": "{id}", "payload": {"at": PROGRESS["at"], "text": "Zib"}}),
    ("POST", ROOT + "/proposals", {"kind": "goal", "target_id": None, "payload": {k: v for k, v in (CREATE | {"parent_id": "{id}"}).items() if k != "source_id"}}),
    ("POST", ROOT + "/proposals/{id}/confirm", {"expected_revision": 0}),
]


def substitute(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new).replace(old.upper(), new)
    if isinstance(value, dict):
        return {key: substitute(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, old, new) for item in value]
    return value


def owner_snapshot(goals):
    with goals.connection() as db:
        return [list(map(tuple, db.execute(query))) for query in (
            "SELECT * FROM owner_goal_records WHERE kind IN ('owner_goal','owner_subgoal','owner_goal_progress') ORDER BY id",
            "SELECT * FROM owner_goal_history ORDER BY id",
            "SELECT * FROM owner_goal_proposals ORDER BY id",
            "SELECT * FROM owner_goal_artifacts ORDER BY goal_id,type,ref",
            "SELECT * FROM owner_goal_receipts ORDER BY principal_id,owner_id,operation_id",
            "SELECT * FROM owner_goal_legacy_answers ORDER BY scope,key",
            "SELECT * FROM owner_goal_legacy_acceptances ORDER BY acceptance_key")]


def test_generic_sql_inventory_and_every_consumer(stores, monkeypatch):
    from pathlib import Path
    import colonymind
    from owner_goal_consumers import CONSUMERS, discover, run
    assert discover(Path(colonymind.__file__).parent) == CONSUMERS
    run(stores, monkeypatch, owner_snapshot)


@pytest.mark.parametrize("path", ["settle"])
def test_secret_absent_timing_on_large_store(stores, path, record_property):
    from statistics import median
    from time import perf_counter_ns
    from colonymind.initiatives.store import SettleGuard, TransitionRefused
    store, goals = stores
    root = create(goals)
    store._db.executemany("INSERT INTO initiatives(id,type,description,kind,status) VALUES (?,?,?,'task','pending')",
                         [(f"fixture-{n}", "fixture", "Zib") for n in range(10000)])
    store._db.commit()
    before = owner_snapshot(goals)
    def invoke(target):
        with pytest.raises(TransitionRefused) as exc:
            store.transition(target, "done", action="rated", guard=SettleGuard(expect=("recorded",)))
        assert exc.value.current is None
    samples = [[], []]
    for iteration in range(141):
        # Alternate order to avoid systematic cache/warm-up bias.
        for index in (iteration % 2, 1 - iteration % 2):
            started = perf_counter_ns()
            invoke((root["id"], "absent")[index])
            if iteration >= 20:
                samples[index].append(perf_counter_ns() - started)
    medians = [median(values) / 1e6 for values in samples]
    record_property(path + "_medians_ms", medians)
    assert abs(medians[0] - medians[1]) < .2, medians
    # Timing equality alone could hide two equally expensive scans. Check work
    # on both paths as well: no more than 1,000 VM instructions on 10,000 rows.
    for target in (root["id"], "absent"):
        steps = []
        store._db.set_progress_handler(lambda: steps.append(1) or 0, 100)
        try:
            invoke(target)
        finally:
            store._db.set_progress_handler(None, 0)
        assert len(steps) < 10
    assert owner_snapshot(goals) == before


@pytest.mark.parametrize("kind", ["owner_goal", "owner_subgoal", "owner_goal_progress"])
def test_direct_generic_sql_cannot_insert_or_promote_owner_kinds(stores, kind):
    import sqlite3
    store, goals = stores
    root = create(goals)
    generic = store.create(type="fixture", description="Zib")
    before = owner_snapshot(goals)
    for query, params in (
        ("INSERT INTO initiatives(id,type,description,kind) VALUES ('fixture','fixture','Zib',?)", (kind,)),
        ("UPDATE initiatives SET kind=? WHERE id=?", (kind, generic.id)),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="Owner kinds"):
            store._db.execute(query, params)
        store._db.rollback()
    assert store.get(root["id"]) is None
    assert store.get(generic.id).kind is None
    assert owner_snapshot(goals) == before


@pytest.mark.parametrize("interrupt", [False, True])
def test_one_time_migration_preserves_records_history_and_all_joins(stores, tmp_path, monkeypatch, interrupt):
    import sqlite3
    from colonymind.owner_goals import schema
    _, goals = stores
    root = create(goals)
    child = create(goals, parent_id=root["id"])
    write(goals, "progress", PROGRESS, id=child["id"])
    proposal = write(goals, "propose", {"kind": "progress", "target_id": child["id"],
        "payload": {k: v for k, v in PROGRESS.items() if k != "source_id"}})["proposal"]
    # Include a tombstone, engine metadata and an ordinary history action on an
    # owner row. Ownership of history depends on the record, not the action name.
    with goals.connection(write=True) as db:
        db.execute("UPDATE owner_goal_records SET hermes_ref='fixture' WHERE id=?", (root["id"],))
        db.execute("INSERT INTO owner_goal_history(initiative_id,agent_id,action,details) VALUES (?, 'agent','rated','{}')", (root["id"],))
    current = goals.read("owner", OWNER, id=root["id"])["goal"]
    write(goals, "delete", {"expected_revision": current["revision"]}, id=root["id"])
    expected = owner_snapshot(goals)
    legacy_dir = tmp_path / "legacy"
    with monkeypatch.context() as patch:
        patch.setattr(schema, "initialize", lambda db: None)
        legacy = InitiativeStore(legacy_dir)
    generic = legacy.create(type="fixture", description="Zib")
    with goals.connection() as db:
        for source, target in (("owner_goal_records", "initiatives"), ("owner_goal_history", "assignment_history")):
            columns = [row[1] for row in db.execute(f"PRAGMA table_info({source})")]
            rows = [tuple(row) for row in db.execute(f"SELECT * FROM {source}")]
            legacy._db.executemany(f"INSERT INTO {target} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", rows)
        for table in ("owner_goal_proposals", "owner_goal_artifacts", "owner_goal_receipts"):
            legacy._db.execute(db.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0])
            rows = [tuple(row) for row in db.execute(f"SELECT * FROM {table}")]
            if rows:
                legacy._db.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in rows[0])})", rows)
    legacy._db.execute("INSERT INTO owner_goal_artifacts VALUES (?, 'library_item','fixture','Zib','fixture','{}',NULL)", (root["id"],))
    legacy._db.execute("INSERT INTO ask_answer_requests VALUES ('fixture','owner',?,'fixture',200,'{}','fixture')", (root["id"],))
    legacy._db.execute("INSERT INTO ask_answer_requests VALUES ('fixture','generic',?,'fixture',200,'{}','fixture')", (generic.id,))
    legacy._db.execute("CREATE TABLE local_draft_acceptances (acceptance_key TEXT PRIMARY KEY,initiative_id TEXT NOT NULL,handoff_hash TEXT NOT NULL DEFAULT '')")
    legacy._db.executemany("INSERT INTO local_draft_acceptances VALUES (?,?,'')", [("owner", root["id"]), ("generic", generic.id)])
    # Exercise replacement of the round-one retention triggers.
    legacy._db.execute("""CREATE TRIGGER owner_goal_records_retained BEFORE DELETE ON initiatives
        WHEN OLD.kind IN ('owner_goal','owner_subgoal','owner_goal_progress')
        BEGIN SELECT RAISE(ABORT,'retained'); END""")
    legacy._db.commit()
    if interrupt:
        class Interrupted(sqlite3.Connection):
            def execute(self, query, *args):
                if query.startswith("DELETE FROM initiatives WHERE"):
                    raise sqlite3.OperationalError("fixture interruption")
                return super().execute(query, *args)
        db = sqlite3.connect(legacy._db_path, factory=Interrupted)
        try:
            with pytest.raises(sqlite3.OperationalError, match="fixture interruption"):
                schema.initialize(db)
            assert db.execute("SELECT count(*) FROM initiatives").fetchone()[0] == 4
            assert db.execute("SELECT count(*) FROM assignment_history").fetchone()[0] == len(expected[1])
            assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='owner_goal_records'").fetchone()
        finally:
            db.close()
    reopened = InitiativeStore(legacy_dir)
    migrated = OwnerGoalsStore(reopened)
    actual = owner_snapshot(migrated)
    assert actual[:3] == expected[:3] and actual[4] == expected[4]
    assert len(actual[3]) == 1
    with migrated.connection() as db:
        assert db.execute("SELECT id FROM initiatives").fetchall()[0][0] == generic.id
        assert db.execute("SELECT count(*) FROM initiatives").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM assignment_history").fetchone()[0] == 0
        assert db.execute("SELECT ask_id FROM ask_answer_requests").fetchone()[0] == generic.id
        assert db.execute("SELECT initiative_id FROM local_draft_acceptances").fetchone()[0] == generic.id
        assert db.execute("SELECT ask_id FROM owner_goal_legacy_answers").fetchone()[0] == root["id"]
        assert db.execute("SELECT initiative_id FROM owner_goal_legacy_acceptances").fetchone()[0] == root["id"]
    assert migrated.read("owner", OWNER, id=proposal["id"], proposals=True)["proposal"] == proposal
    assert reopened.get(generic.id) is not None
    again = InitiativeStore(legacy_dir)
    assert owner_snapshot(OwnerGoalsStore(again)) == actual
    again.close()
    reopened.close()
    legacy.close()


@pytest.fixture
async def oracle(client, stores, tmp_path, monkeypatch):
    from colonymind.api.routers import host, mind as mind_routes
    from colonymind.mind.tick import Mind
    mind = Mind(config={"quiet_hours": ""}, store=stores[0], state_dir=tmp_path,
                owner_id="owner", backups=False)
    monkeypatch.setattr(host, "_initiative_store", stores[0])
    monkeypatch.setattr(mind_routes, "_mind", mind)
    client.test_app.include_router(host.router)
    client.test_app.include_router(mind_routes.router)
    return client


@pytest.mark.parametrize("caller", ["anonymous", "guest", "mixed", "unbound", "unverified", "forged"])
async def test_secret_existence_oracle_across_every_id_route(oracle, stores, caller):
    client, goals = oracle, stores[1]
    root = create(goals)
    proposal = write(goals, "propose", {"kind": "progress", "target_id": root["id"],
        "payload": {k: v for k, v in PROGRESS.items() if k != "source_id"}})["proposal"]
    assert goals.read("owner", OWNER, id=proposal["id"], proposals=True)["proposal"] == proposal
    before = owner_snapshot(goals)
    client.test_app.state.authority = anonymous_authority() if caller == "anonymous" else key_authority()
    client.test_app.state.audience = {
        "guest": Audience("owner", frozenset({"guest"}), maximum_tier="Private"),
        "mixed": Audience("owner", frozenset({"owner", "guest"}), True, "Private"),
        "unbound": None,
    }.get(caller, OWNER)
    if caller in {"unverified", "forged"}:
        client.test_app.dependency_overrides.pop(get_action_verifier)
    headers = {"Owner-Goal-Action": "A" * 43} if caller == "forged" else {}
    for method, path, body in [*GENERIC_ID_CASES, *OWNER_ID_CASES]:
        # These two cases have a verified owner read audience, but no write proof.
        # Non-owner reads are covered by the other four caller contexts.
        if caller in {"unverified", "forged"} and method == "GET" and path.startswith(ROOT):
            continue
        responses = []
        target = proposal["id"] if "/proposals/{id}" in path else root["id"]
        for id in (target, "absent"):
            response = await client.request(method, substitute(path, "{id}", id),
                                            json=substitute(body, "{id}", id),
                                            headers={**headers, "Idempotency-Key": str(uuid.uuid4())})
            assert response.status_code < 500 or response.status_code in {501, 503}, response.text
            responses.append((response.status_code, substitute(response.json(), id, "{id}")))
        assert responses[0] == responses[1], (caller, method, path, responses)
    assert owner_snapshot(goals) == before


def test_oracle_manifest_covers_id_handlers():
    import re
    from colonymind.api.routers import host, mind
    def canonical(path):
        return re.sub(r"\{[^}]+\}", "{id}", path.split("?")[0])
    cases = {(method, canonical(path)) for method, path, _ in [*GENERIC_ID_CASES, *OWNER_ID_CASES]}
    for router in (host.router, mind.router, routes.router):
        for route in router.routes:
            if router is host.router and "/initiatives/" not in route.path:
                continue
            if router is mind.router and not route.param_convertors:
                continue
            # Observation sources are a separate namespace, not record IDs.
            if "{source}" in route.path:
                continue
            path = canonical(route.path)
            path = path.replace("/{id}/{id}", "/{id}/yes") if "{answer}" in route.path else path
            path = path.replace("/{id}/artifacts/{id}/{id}", "/{id}/artifacts/library_item/item")
            path = path.replace("/pending/{id}/{id}", "/pending/{id}/reject")
            for method in route.methods:
                assert (method, path) in cases, (method, route.path)


@pytest.mark.parametrize("kind", ["owner_goal", "owner_subgoal", "owner_goal_progress"])
async def test_idea_paths_treat_every_owner_kind_as_missing(oracle, stores, kind):
    store, goals = stores
    root = create(goals)
    child = create(goals, parent_id=root["id"])
    write(goals, "progress", PROGRESS, id=child["id"])
    with goals.connection(write=True) as db:
        # Owner completion must not set the catalog's built-idea indicator either.
        db.execute("UPDATE owner_goal_records SET status='completed' WHERE kind != 'owner_goal_progress'")
        target = db.execute("SELECT id FROM owner_goal_records WHERE kind=?", (kind,)).fetchone()[0]
    before = owner_snapshot(goals)
    assert store.ideas_page(1) == ([], None, False)
    catalog = await oracle.get("/v1/host/initiatives/ideas?limit=1")
    assert catalog.status_code == 200
    assert catalog.json() == {"sections": [], "viewer_state": {"has_built_idea": False},
                              "pagination": {"next_cursor": None, "has_next_page": False}}
    for id in (target, "absent"):
        assert store.idea_states([id]) == [{"id": id, "exists": False, "state": None, "status": None}]
        assert store.last_response_origin(id) is None
        receipt = {"receipt_id": "fixture", "kind": "open", "visible_fraction": 1,
                   "surface": "ideas", "anchor": "fixture"}
        assert store.record_engagement(id, receipt) is False
        assert store.share(id, "Zorp sentinel", "Zorp sentinel") is None
    # Every Ideas/FEED19 route is checked with a verified owner audience too.
    for method, path, body in GENERIC_ID_CASES:
        if not path.startswith("/v1/host/initiatives/"):
            continue
        responses = []
        for id in (target, "absent"):
            response = await oracle.request(method, substitute(path, "{id}", id),
                                            json=substitute(body, "{id}", id))
            if "/ideas/state" in path:
                assert response.status_code == 200
                assert response.json() == {"states": [
                    {"id": id, "exists": False, "state": None, "status": None}]}
            elif "/ideas/associate" in path:
                assert response.status_code == 200
                assert response.json() == {"cards": [{"unit_id": id}],
                    "capabilities": {"feed_idea_association": True}, "ready": True}
            elif path != "/v1/host/initiatives/ideas":
                assert response.status_code == 404, response.text
            responses.append((response.status_code, substitute(response.json(), id, "{id}")))
        assert responses[0] == responses[1], (method, path, responses)
    assert store._db.execute("SELECT count(*) FROM initiative_shares").fetchone()[0] == 0
    assert store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0
    # A stale snapshot/history reference must not make a hidden source readable.
    store._db.execute("INSERT INTO initiative_shares VALUES ('fixture', ?, 'Zorp sentinel', 'Zorp sentinel')", (target,))
    store._db.execute("""INSERT INTO assignment_history(initiative_id,agent_id,action,origin)
        VALUES (?, 'agent', 'llm_approved', '{"surface":"ideas"}')""", (target,))
    store._db.commit()
    assert store.last_response_origin(target) is None
    assert store.resolve_share("fixture") is None
    hidden = await oracle.get("/v1/host/initiatives/shares/fixture")
    absent = await oracle.get("/v1/host/initiatives/shares/absent")
    assert hidden.status_code == absent.status_code == 404
    assert hidden.json() == absent.json()
    assert owner_snapshot(goals) == before


@pytest.mark.parametrize("kind", ["owner_goal", "owner_subgoal", "owner_goal_progress"])
def test_new_idea_consumers_use_generic_view_on_unmigrated_rows(tmp_path, monkeypatch, kind):
    from colonymind.owner_goals import schema
    # A connection opened before owner-table migration still relies on the
    # generic view's kind filter, rather than physical table separation.
    with monkeypatch.context() as patch:
        patch.setattr(schema, "initialize", lambda db: None)
        store = InitiativeStore(tmp_path)
    try:
        store._db.execute("""INSERT INTO initiatives(id,type,description,kind,status)
            VALUES ('hidden','fixture','Zorp sentinel',?,'completed')""", (kind,))
        store._db.execute("""INSERT INTO assignment_history(initiative_id,agent_id,action,origin)
            VALUES ('hidden','agent','llm_approved','{"surface":"ideas"}')""")
        store._db.commit()
        assert store.ideas_page(1) == ([], None, False)
        assert store.idea_states(["hidden"]) == [
            {"id": "hidden", "exists": False, "state": None, "status": None}]
        assert store.last_response_origin("hidden") is None
        assert store.share("hidden", "Zorp sentinel", "Zorp sentinel") is None
        assert not store.record_engagement("hidden", {
            "receipt_id": "fixture", "kind": "open", "visible_fraction": 1,
            "surface": "ideas", "anchor": "fixture"})
        assert store._db.execute("SELECT count(*) FROM initiative_shares").fetchone()[0] == 0
        assert store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0
        store._db.execute("INSERT INTO initiative_shares VALUES ('fixture','hidden','Zorp sentinel','Zorp sentinel')")
        store._db.commit()
        assert store.resolve_share("fixture") is None
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["update", "complete", "fail", "retry", "cancel",
                                       "transition", "settle", "history", "answer_receipt"])
@pytest.mark.parametrize("kind", ["owner_goal", "owner_subgoal", "owner_goal_progress"])
def test_generic_store_mutations_cannot_reach_any_owner_kind(stores, operation, kind):
    from datetime import datetime, timezone
    from colonymind.initiatives.store import SettleGuard, TransitionRefused
    store, goals = stores
    root = create(goals)
    child = create(goals, parent_id=root["id"])
    write(goals, "progress", PROGRESS, id=child["id"])
    with goals.connection() as db:
        id = db.execute("SELECT id FROM owner_goal_records WHERE kind=?", (kind,)).fetchone()[0]
    before = owner_snapshot(goals)
    callbacks = []
    calls = {
        "update": lambda id: store.update(id, verdict="useful", kind="task", context={}),
        "complete": lambda id: store.complete(id, "agent", "Zib"),
        "fail": lambda id: store.fail(id, "agent", "Zib", retry=True),
        "retry": lambda id: store.retry(id, "agent"),
        "cancel": lambda id: store.cancel(id, "agent"),
        "transition": lambda id: store.transition(id, "done", action="rated", verdict="useful"),
        "settle": lambda id: store.transition(id, "done", action="rated", guard=SettleGuard(
            expect=("recorded",), writes=lambda *args: callbacks.append(args) or [])),
        "history": lambda id: store.log_history(id, "rated", "agent", details={"verdict": "useful"}),
        "answer_receipt": lambda id: store.record_answer_request(scope="fixture", key=id, ask_id=id,
            request_hash="fixture", status_code=404, response={}, at=datetime.now(timezone.utc)),
    }
    results = []
    for target in (id, "absent"):
        try:
            results.append(calls[operation](target))
        except TransitionRefused as exc:
            assert exc.current is None
            results.append("refused")
    assert results[0] == results[1]
    assert callbacks == []
    assert owner_snapshot(goals) == before
    assert store.get(id) is None and store.get_history(id) == [] and store.last_responses([id]) == {}


@pytest.mark.parametrize("kind", ["owner_goal", "owner_subgoal", "owner_goal_progress"])
def test_generic_creation_and_kind_promotion_are_closed(stores, kind):
    store, goals = stores
    args = dict(kind=kind, type="fixture", title="Zib", drive="fixture", cls="internal",
                decision="defer", decision_reason="fixture", status="recorded", dedup_key=None)
    with pytest.raises(ValueError, match="Owner kinds"):
        store.create_intention(**args)
    with pytest.raises(ValueError, match="Owner kinds"):
        store.create(type="fixture", description="Zib", kind=kind)
    generic = store.create(type="fixture", description="Zib")
    with pytest.raises(ValueError, match="Owner kinds"):
        store.update(generic.id, kind=kind)
    with pytest.raises(ValueError, match="Owner kinds"):
        store.update("absent", kind=kind)
    assert store.get(generic.id).kind is None
    assert goals.read("owner", OWNER)["goals"] == []


@pytest.mark.parametrize("audience", [Audience("owner", frozenset({"guest"})),
                                     Audience("owner", frozenset({"owner", "guest"}), True, "Private"),
                                     Audience("owner", frozenset({"owner"}))])
@pytest.mark.parametrize("tier", ["Public", "Everyday", "Personal", "Private", "Secret"])
def test_proposals_are_owner_only_at_every_tier(stores, audience, tier):
    _, goals = stores
    proposal = write(goals, "propose", {"kind": "goal", "target_id": None, "payload": {k: v for k, v in
        (CREATE | {"sensitivity": tier}).items() if k != "source_id"}, "source_id": "chat:fixture"})["proposal"]
    assert goals.read("owner", audience, proposals=True)["proposals"] == []
    for id in (proposal["id"], "absent"):
        with pytest.raises(GoalError, match="owner_goal_proposal_not_found") as exc:
            goals.read("owner", audience, proposals=True, id=id)
        assert exc.value.status == 404
    assert goals.read("owner", OWNER, proposals=True, id=proposal["id"])["proposal"] == proposal


def test_authorization_precedes_every_lookup_and_resolved_effects_are_reverified(stores, monkeypatch):
    _, goals = stores
    root = create(goals)
    before = owner_snapshot(goals)
    def no_lookup(*args, **kwargs):
        pytest.fail("lookup preceded authorization")
    def deny(context, **kwargs):
        assert kwargs["resolved_action"]["phase"] == "authorize"
        assert kwargs["receipt"] is None
        assert kwargs["connection"].in_transaction
        raise GoalError(403, "not_owner")
    with monkeypatch.context() as patch:
        for name in ("_goal", "_row", "_revision", "_proposal", "_resolve"):
            patch.setattr(goals, name, no_lookup)
        for target in (root["id"], "absent"):
            with pytest.raises(GoalError, match="not_owner"):
                write(goals, "patch", {"expected_revision": 0, "title": "Zib"}, id=target, verifier=deny)
    def deny_effects(context, **kwargs):
        if kwargs["resolved_action"].get("phase") != "authorize":
            raise GoalError(403, "not_owner")
        return permit(context, **kwargs)
    with pytest.raises(GoalError, match="not_owner"):
        write(goals, "patch", {"expected_revision": 1, "title": "Zib"}, id=root["id"], verifier=deny_effects)
    def no_tier(context, **kwargs):
        return replace(permit(context, **kwargs), explicit_tier=False)
    with monkeypatch.context() as patch:
        patch.setattr(goals, "_resolve", no_lookup)
        for target in (root["id"], "absent"):
            with pytest.raises(GoalError, match="not_owner"):
                write(goals, "patch", {"expected_revision": 0, "sensitivity": "Public"}, id=target, verifier=no_tier)
    assert owner_snapshot(goals) == before


@pytest.mark.parametrize("status", ["asked", "failed", "acknowledged"])
def test_generic_selectors_and_aggregates_exclude_owner_rows_even_with_engine_fields(stores, status):
    from datetime import datetime, timezone
    store, goals = stores
    root = create(goals)
    floor, ceiling = datetime(2000, 1, 1, tzinfo=timezone.utc), datetime(9999, 1, 1, tzinfo=timezone.utc)
    # Model a legacy row with engine metadata; exclusion must depend on kind.
    with goals.connection(write=True) as db:
        db.execute("""UPDATE owner_goal_records SET status=?,dedup_key='fixture',dedup_base='fixture',
            hermes_ref='fixture',ask_code='FIXTURE',lesson_ids='["fixture"]',cost_tokens=100,
            cls='internal',outcome='failed',failed_at=?,assigned_at=?,acknowledged_at=?,
            expires_at=?,timeout_seconds=1,assigned_agent_id='agent' WHERE id=?""",
                   (status, floor.isoformat(), floor.isoformat(), floor.isoformat(), floor.isoformat(), root["id"]))
    assert store.get_by_dedup_key("fixture") is None
    assert store.get_active_by_dedup_base("fixture") is None
    assert store.get_by_hermes_ref("fixture") is None
    assert store.get_by_ask_code("FIXTURE") is None
    assert store.open_ask_codes() == []
    assert store.intentions(kind=["owner_goal"]) == []
    assert store.intentions() == [] and store.list() == [] and store.count() == 0
    assert store.lesson_rows() == [] and store.failures_since("internal", floor) == []
    assert store.tokens_since(floor) == 0
    assert store.count_transitions("owner_goal_create", floor) == 0
    assert store.last_transition_at("owner_goal_create") is None
    assert store.get_agent_history("agent") == []
    before = owner_snapshot(goals)
    assert store.prune_intentions(ceiling) == {}
    assert store.delete_old([status], ceiling) == 0
    assert owner_snapshot(goals) == before


def test_generic_serializers_reject_owner_rows_like_missing_rows(stores):
    from colonymind.api.routers import host, mind
    from colonymind.initiatives.models import StoredInitiative
    from colonymind.mind import audit
    _, goals = stores
    root = create(goals)
    child = create(goals, parent_id=root["id"])
    write(goals, "progress", PROGRESS, id=child["id"])
    with goals.connection() as db:
        rows = [StoredInitiative.from_row(dict(row)) for row in db.execute("SELECT * FROM owner_goal_records")]
    for serializer in (audit.entry, mind._entry, host._initiative_to_response, host._map_initiative_to_schema):
        with pytest.raises(Exception) as absent:
            serializer(None)
        for row in rows:
            with pytest.raises(type(absent.value)) as hidden:
                serializer(row)
            assert str(hidden.value) == str(absent.value)


@pytest.mark.parametrize("audience", [None, Audience("owner", frozenset({"guest"})),
                                     Audience("owner", frozenset({"owner", "guest"}), True, "Private")])
async def test_public_proposal_http_reads_require_owner_only_audience(client, stores, audience):
    proposal = write(stores[1], "propose", {"kind": "goal", "target_id": None, "source_id": "chat:fixture",
        "payload": {k: v for k, v in (CREATE | {"sensitivity": "Public"}).items() if k != "source_id"}})["proposal"]
    client.test_app.state.audience = audience
    responses = [await client.get(ROOT + "/proposals/" + id) for id in (proposal["id"], "absent")]
    assert responses[0].status_code == responses[1].status_code == (503 if audience is None else 404)
    assert responses[0].json() == responses[1].json()
    response = await client.get(ROOT + "/proposals")
    if audience is None:
        code(response, 503, "owner_goals_unavailable")
    else:
        assert response.json() == {"proposals": [], "limit": 50, "offset": 0, "next_offset": None}


async def test_generic_collection_reads_never_include_owner_records(oracle, stores):
    root = create(stores[1])
    before = owner_snapshot(stores[1])
    for path in ("/v1/host/initiatives", "/v1/host/initiatives/ideas", "/v1/mind/goals", "/v1/mind/dispatch", "/v1/mind/outbox",
                 "/v1/mind/log", "/v1/mind/log?kind=owner_goal", "/v1/mind/stats", "/v1/mind/asks"):
        response = await oracle.get(path)
        assert response.status_code == 200, response.text
        assert root["id"] not in response.text and CREATE["description"] not in response.text
    response = await oracle.post("/v1/mind/tick")
    assert response.status_code == 200, response.text
    assert owner_snapshot(stores[1]) == before
