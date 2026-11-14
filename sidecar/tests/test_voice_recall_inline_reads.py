"""Voice recall keeps its reads on its own worker and its one SQLite connection."""
import asyncio
import threading

from colonymind.memory import producers, turn_budget
from colonymind.memory.turn_budget import TurnBudget
from colonymind.memory.voice_recall import read_voice_packet
from colonymind.turns import TurnIdempotencyLedger
from test_canonical_memory_search import memory_app  # noqa: F401
from test_turn_source_evidence import source_app  # noqa: F401


def thread_of_read():
    return asyncio.run(producers.read(threading.get_ident))


def test_unbudgeted_reads_run_on_the_calling_worker_only_inside_inline_reads():
    me = threading.get_ident()
    assert thread_of_read() != me
    with producers.inline_reads():
        assert thread_of_read() == me
    assert thread_of_read() != me


def test_budgeted_reads_still_use_the_pool_inside_inline_reads():
    async def read():
        with turn_budget.turn(TurnBudget(wait_s=7.5)):
            return await producers.read(threading.get_ident)

    with producers.inline_reads():
        assert asyncio.run(read()) != threading.get_ident()


def test_a_voice_packet_opens_one_connection(monkeypatch, tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    opened = []
    connect = TurnIdempotencyLedger._connect
    monkeypatch.setattr(TurnIdempotencyLedger, "_connect", lambda self: opened.append(1) or connect(self))
    read_voice_packet(ledger, query="what did we talk about this week", contact_id="person", session_id="",
                      k=5, max_tokens=600, timezone_name="America/New_York", cancelled=threading.Event())
    assert len(opened) == 1
    assert not producers._INLINE.get()

from colonymind.memory import curated


def test_voice_fences_borrow_one_connection_from_both_read_interfaces(monkeypatch, tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    for source, text in [('kept', 'Fixture harbor schedule'), ('withdrawn', 'Unrelated fixture')]:
        ledger.record_source(source, contact_id='person', session_id='session',
                             messages=[{'role': 'user', 'content': text}], derive_claims=False)
    curated.mutate(ledger, 'person', 'inline', 'remember', request_id='inline', content='Fixture harbor note')
    curated.forget_sources(ledger, 'person', source_ids=['withdrawn'])
    opened = []
    connect, read_connect = TurnIdempotencyLedger._connect, TurnIdempotencyLedger._read_connect

    def native(self):
        opened.append('native')
        return connect(self)

    def readonly(self):
        opened.append('readonly')
        return read_connect(self)

    monkeypatch.setattr(TurnIdempotencyLedger, '_connect', native)
    monkeypatch.setattr(TurnIdempotencyLedger, '_read_connect', readonly)
    packet = read_voice_packet(ledger, query='harbor', contact_id='person', session_id='session',
                              k=5, max_tokens=600, timezone_name='UTC', cancelled=threading.Event())
    assert packet is not None and 'harbor' in packet.content
    assert opened == ['native']
    assert not producers._INLINE.get()


async def test_inline_reads_stay_off_the_event_loop_and_see_an_erasure_mid_packet(memory_app, monkeypatch):
    from httpx import ASGITransport, AsyncClient
    from onekey import KEY
    from colonymind.memory import search
    app, ledger = memory_app
    loop_thread, seen, erase_now = threading.get_ident(), [], [False]
    prepare = search._prepare_memory

    def watched(*args, **kwargs):
        seen.append((threading.get_ident(), producers.active()))
        if erase_now[0]:
            erase_now[0] = False
            curated.forget_sources(ledger, 'person', source_ids=['report'])
        return prepare(*args, **kwargs)

    monkeypatch.setattr(search, "_prepare_memory", watched)
    body = {"identity": {"host_id": "fixture"}, "person_id": "person", "session_id": "later",
            "query": "hydrofoil", "k": 5, "max_tokens": 600}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        first = await client.post("/v1/host/memory/voice-recall", json={**body, "utterance_id": "u1", "request_id": "r1"})
        assert "Friday at nine" in first.json()["content"]
        erase_now[0] = True  # the erasure lands after the lanes read, before the packet is prepared
        second = await client.post("/v1/host/memory/voice-recall", json={**body, "utterance_id": "u2", "request_id": "r2"})
        assert second.status_code == 200 and "Friday" not in second.json()["content"]
    assert seen and all(thread != loop_thread and not pooled for thread, pooled in seen)
