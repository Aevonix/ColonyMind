"""All keyword projections preserve ranking and scope across the read window."""
from contextlib import closing
import json

import pytest

from colonymind.turns import claim_keys, idempotency, keywords


LANES = {
    "literal": ("turn_source_search", "f.turn_id, f.role, f.content",
                "CROSS JOIN turn_sources AS s ON s.turn_id=f.turn_id"),
    "stem": (keywords.TABLE, "m.turn_id, m.role, m.chunk, f.content AS indexed",
             f"CROSS JOIN {keywords.CHUNKS} AS m ON m.id=f.rowid CROSS JOIN turn_sources AS s ON s.turn_id=m.turn_id"),
    "claim": (claim_keys.TABLE, "k.claim_id, k.turn_id, k.message_hash, f.content AS indexed",
              f"CROSS JOIN {claim_keys.ROWS} AS k ON k.id=f.rowid CROSS JOIN turn_sources AS s ON s.turn_id=k.turn_id"),
}
SOURCE_FIELDS = ("contact_id", "session_id", "scope", "occurred_at", "ingested_at")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    ledger = idempotency.TurnIdempotencyLedger(tmp_path_factory.mktemp("scoped") / "turns.db")
    texts = ("harbour ferry", "harbour harbour moving ferry", "moving boxes", "quiet orchard")
    with closing(ledger._connect()) as conn, conn:
        for index in range(640):
            turn = f"turn-{index}"
            text = texts[index % len(texts)] + " filler" * (index % 5)
            messages = [{"role": "user", "content": text}, {"role": "assistant", "content": text}]
            conn.execute("INSERT INTO turn_sources(turn_id,content_sha256,contact_id,session_id,scope,"
                         "messages_json,occurred_at,ingested_at) VALUES (?,?,?,?,?,?,?,?)",
                         (turn, f"{index:064x}", ("owner", "sam", "lee")[index % 3],
                          f"session-{index % 4}", "person" if index % 5 == 0 else "session",
                          json.dumps(messages), None if index % 2 else "2026-01-01T00:00:00Z",
                          "2026-01-02T00:00:00Z"))
            ledger._index_messages(conn, turn, messages)
            row = conn.execute(f"INSERT INTO {claim_keys.ROWS}(claim_id,turn_id,message_hash) VALUES (?,?,?)",
                               (f"claim-{index}", turn, f"message-{index}"))
            conn.execute(f"INSERT INTO {claim_keys.TABLE}(rowid,content) VALUES (?,?)", (row.lastrowid, text))
            if index % 17 == 0:
                conn.execute("INSERT INTO source_attribution_invalidations VALUES (?,?,?)",
                             (turn, "corrected", f"operation-{index}"))
        # Old/stale projections can have no canonical source or no location row.
        conn.execute("INSERT INTO turn_source_search(turn_id,role,content) VALUES ('gone','user','harbour ferry')")
        conn.execute(f"INSERT INTO {keywords.TABLE}(rowid,content) VALUES (99999,'harbour ferry')")
        conn.execute(f"INSERT INTO {claim_keys.TABLE}(rowid,content) VALUES (99999,'harbour ferry')")
    return ledger


def expected_rows(conn, lane, expression, contact, session):
    """Rank the FTS table alone; look up and authorize each source in Python.

    No joined or windowed query is used to construct the oracle.
    """
    table = LANES[lane][0]
    sources = {row["turn_id"]: dict(row) for row in conn.execute("SELECT * FROM turn_sources")}
    invalid = {row[0] for row in conn.execute("SELECT source_id FROM source_attribution_invalidations")}
    locations = {}
    if lane != "literal":
        mapping = keywords.CHUNKS if lane == "stem" else claim_keys.ROWS
        locations = {row["id"]: dict(row) for row in conn.execute(f"SELECT * FROM {mapping}")}
    matches = list(conn.execute(f"SELECT rowid, *, bm25({table}) AS score FROM {table} WHERE {table} MATCH ?",
                                (expression,)))
    matches.sort(key=lambda row: (row["score"], row["rowid"]))
    result = []
    for match in matches:
        location = match if lane == "literal" else locations.get(match["rowid"])
        if location is None:
            continue
        source = sources.get(location["turn_id"])
        if (source is None or source["turn_id"] in invalid or source["contact_id"] != contact
                or not (source["scope"] == "person" or source["session_id"] == session)):
            continue
        fields = {"literal": ("turn_id", "role", "content"), "stem": ("turn_id", "role", "chunk"),
                  "claim": ("claim_id", "turn_id", "message_hash")}[lane]
        value = {key: location[key] for key in fields}
        if lane != "literal":
            value["indexed"] = match["content"]
        value.update({key: source[key] for key in SOURCE_FIELDS})
        value["rowid"] = match["rowid"]
        result.append(value)
    return result, len(matches)


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("expression", ['"harbour"', '"harbour" OR "moving"', '"harbour ferry"',
                                         '"moving" NOT "boxes"', '"absent"'])
@pytest.mark.parametrize("contact,session", [("owner", "session-0"), ("sam", "session-1"),
                                               ("lee", "session-2"), ("nobody", "session-3")])
def test_scoped_matches_equal_independent_rows(store, lane, expression, contact, session):
    table, select, joins = LANES[lane]
    select += ", " + ", ".join("s." + field for field in SOURCE_FIELDS)
    with closing(store._connect()) as conn:
        expected, count = expected_rows(conn, lane, expression, contact, session)
        # Include no fast window, both sides of its boundary, and the complete ranked set.
        for window in dict.fromkeys((0, 1, 7, 256, max(0, count - 1), count, count + 1, None)):
            actual = list(idempotency._scoped_matches(conn, table=table, select=select, joins=joins,
                          expression=expression, contact_id=contact, session_id=session, window=window))
            assert [dict(row) for row in actual] == expected, (lane, expression, contact, session, window)
