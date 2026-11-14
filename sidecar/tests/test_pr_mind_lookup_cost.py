"""The summary lookup's work follows the session's own newest summaries and nothing else the contact has.

A session's summaries share one id prefix. The lookup finds them through the primary key, newest date in the
id first, and ``_sessions`` reads it up to the first summary that is not forgotten. The lookups of one
``_sessions`` call are traced and run again under a counter of SQLite's virtual-machine steps, once read
that way and once to their end, on a store without planner statistics and after ``ANALYZE``. What the call
itself takes from SQLite is counted on its own connection: a call that read on would not show in the rerun.
"""
from contextlib import closing
from datetime import timedelta

import pytest

from colonymind.memory import curated
from test_mind_consolidate import CONTACT
from test_pr_mind_episodes import fx, session

STATISTICS = pytest.mark.parametrize('statistics', [False, True], ids=['no-statistics', 'after-analyze'])
LOOKAHEAD = 20      # virtual-machine steps allowed for the one row the driver reads past the row it hands over


def lookups(fx, monkeypatch):
    """The summary lookups of one ``_sessions`` call, as SQLite ran them."""
    statements = []
    connect = fx.ledger._connect

    def traced(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    c = fx.mind.consolidation
    with monkeypatch.context() as patch:
        patch.setattr(fx.ledger, '_connect', traced)
        c._sessions(fx.now, c.local_date(fx.now), fx.now - timedelta(days=1))
    return [sql for sql in statements if 'episode_summary' in sql]


def work(fx, statements, to_end=False):
    """Rows read, virtual-machine steps taken and the plans when SQLite runs the statements again.

    Each is read as ``_sessions`` reads it, up to the first summary that is not forgotten, or ``to_end``.
    """
    steps = [0]

    def step():
        steps[0] += 1
        return 0

    read = []
    with closing(fx.ledger._connect()) as conn:
        plans = {' | '.join(row[3] for row in conn.execute('EXPLAIN QUERY PLAN ' + sql)) for sql in statements}
        conn.set_progress_handler(step, 1)
        for sql in statements:
            rows = 0
            for row in conn.execute(sql):
                rows += 1
                if not to_end and not curated.is_forgotten(fx.ledger, row['turn_id']):
                    break
            read.append(rows)
    return sorted(read), steps[0], plans


def taken_by_the_call(fx, monkeypatch):
    """The summary rows that one ``_sessions`` call takes from SQLite, counted on the call's own connection.

    Every row of a lookup that the call fetches is counted, whether the call then uses the row or not.
    """
    taken = []
    connect = fx.ledger._connect

    def counted(*args, **kwargs):
        conn = connect(*args, **kwargs)
        factory = conn.row_factory

        def row(cursor, values):
            made = factory(cursor, values)
            if [column[0] for column in cursor.description] == ['turn_id', 'last_at']:
                taken.append(made['turn_id'])
            return made

        conn.row_factory = row
        return conn

    c = fx.mind.consolidation
    with monkeypatch.context() as patch:
        patch.setattr(fx.ledger, '_connect', counted)
        c._sessions(fx.now, c.local_date(fx.now), fx.now - timedelta(days=1))
    return taken


def measure(fx, monkeypatch, statistics):
    """The work of one ``_sessions`` call's lookups: read to the first standing summary, and read to the end."""
    with closing(fx.ledger._connect()) as conn:
        if statistics:
            conn.execute('ANALYZE')
            conn.commit()
        analysed = conn.execute("SELECT count(*) FROM sqlite_master WHERE name='sqlite_stat1'").fetchone()[0]
    assert bool(analysed) == statistics
    statements = lookups(fx, monkeypatch)
    return work(fx, statements), work(fx, statements, to_end=True)


def small(fx):
    """A session whose summary of tonight stands and a session that has none."""
    c, now = fx.mind.consolidation, fx.now
    fx.now = now - timedelta(hours=2)
    assert c.autobiography.record(f'episode:kept:{c.local_date(now)}', 'episode_summary', 'An earlier answer.',
                                  contact_id=CONTACT, session='kept')
    fx.now = now
    session(fx, 'kept', now - timedelta(hours=1))
    session(fx, 'fresh', now - timedelta(hours=1))


def more_turns(fx):
    for index in range(300):
        fx.turn(f'history-{index}', CONTACT, f'history-{index // 20}', f'Old message {index}.',
                at=fx.now - timedelta(days=30, minutes=-index))


def summaries_of_other_sessions(fx):
    c = fx.mind.consolidation
    for index in range(300):
        assert c.autobiography.record(f'episode:other-{index}:{c.local_date(fx.now)}', 'episode_summary',
                                      f'Another session was answered ({index}).', contact_id=CONTACT,
                                      session=f'other-{index}')


def mind_rows_that_are_no_summaries(fx):
    c = fx.mind.consolidation
    for index in range(300):
        assert c.autobiography.record(f'note-{index}', 'noted', f'A note that is no summary ({index}).',
                                      contact_id=CONTACT)


def older_summaries_of_the_session(fx, days):
    """One summary of ``kept`` for each of the given days before tonight, stored on its day."""
    c, now = fx.mind.consolidation, fx.now
    for back in days:
        fx.now = now - timedelta(days=back)
        assert c.autobiography.record(f'episode:kept:{c.local_date(fx.now)}', 'episode_summary',
                                      f'The answer {back} days before.', contact_id=CONTACT, session='kept')
    fx.now = now


@STATISTICS
@pytest.mark.parametrize('grow', [more_turns, summaries_of_other_sessions, mind_rows_that_are_no_summaries])
def test_summary_lookup_work_does_not_grow_with_the_contacts_other_history(fx, monkeypatch, grow, statistics):
    small(fx)
    before = measure(fx, monkeypatch, statistics)
    grow(fx)
    after = measure(fx, monkeypatch, statistics)
    print(f'small store: (rows read, VM steps, plans) to the first standing summary {before[0]}, to the end {before[1]}')
    print(f'with 300 {grow.__name__}: to the first standing summary {after[0]}, to the end {after[1]}')
    assert before[0][0] == before[1][0] == [0, 1]
    # No tolerance: the same rows read, the same steps and the same plan, however the statement is read.
    assert after == before


@STATISTICS
def test_summary_lookup_work_does_not_grow_with_the_sessions_older_summaries(fx, monkeypatch, statistics):
    small(fx)
    first, whole = measure(fx, monkeypatch, statistics)
    older_summaries_of_the_session(fx, [1])
    one, _ = measure(fx, monkeypatch, statistics)
    taken = taken_by_the_call(fx, monkeypatch)
    older_summaries_of_the_session(fx, range(2, 301))
    many, whole_after = measure(fx, monkeypatch, statistics)
    taken_after = taken_by_the_call(fx, monkeypatch)
    print(f'small store: (rows read, VM steps, plans) to the first standing summary {first}, to the end {whole}')
    print(f'with 1 older summary of the session: to the first standing summary {one}')
    print(f'with 300 older summaries of the session: to the first standing summary {many}, to the end {whole_after}')
    print(f'summary rows the call itself takes: with 1 older summary {taken}, with 300 {len(taken_after)} row(s)')
    assert first[0] == whole[0] == [0, 1]
    # The newest summary stands, so the read ends there. The driver has by then stepped the statement once
    # more, to the next row: one older summary is read ahead, never a second. The tolerance is that one row.
    assert one[0] == [0, 1] and one[2] == first[2] and 0 <= one[1] - first[1] <= LOOKAHEAD
    assert many == one
    # Read to its end the statement does return them all: the bound is the read that stops, not the statement.
    assert whole_after[0] == [0, 301]
    # The call itself stops there too: on its own connection it takes tonight's summary and no older one.
    tonight = f'mind:episode:kept:{fx.mind.consolidation.local_date(fx.now)}:episode_summary'
    assert taken == [tonight] and taken_after == [tonight]


@STATISTICS
def test_summary_lookup_runs_on_the_primary_key_and_sorts_nothing(fx, monkeypatch, statistics):
    small(fx)
    plans = measure(fx, monkeypatch, statistics)[0][2]
    summaries_of_other_sessions(fx)
    grown = measure(fx, monkeypatch, statistics)[0][2]
    print(f'EXPLAIN QUERY PLAN of the lookup, small store: {sorted(plans)}')
    print(f'EXPLAIN QUERY PLAN of the lookup, 300 summaries of other sessions: {sorted(grown)}')
    for plan in plans | grown:
        assert 'sqlite_autoindex_turn_sources_1' in plan and 'B-TREE' not in plan
