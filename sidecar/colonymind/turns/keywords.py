"""The keyword lanes' terms and the stemmed index (docs/RECALL-HYBRID.md, Keyword lanes).

Two full-text tables index the same chunks of every canonical source (``idempotency._lexical_chunks``):

- ``turn_source_search`` holds each chunk as written (``unicode61``);
- ``turn_source_stems`` holds each chunk with English stemming (``porter unicode61``), after every run of CJK
  ideographs, kana or hangul in it is rewritten as overlapping two-character pieces (:func:`cjk_bigrams`). There
  "moved" finds "moving", and "东京" finds "我昨天去了东京出差", which ``unicode61`` keeps as one token. The stemming is
  English only: other inflected languages gain nothing from it.

A stemmed row names its chunk by place, never by text: ``turn_source_stem_chunks`` gives each row its source, its
speaker and the chunk's index among that source's chunks, and a hit is read back from the canonical source (the
indexed text is a rewrite, not the chunk). The rows follow their source in SQL, whoever changes it: triggers on
``turn_sources`` delete a source's rows when the source is deleted or its messages change, and queue every new or
changed source in ``turn_source_stem_pending``. The ledger indexes a source when it writes it, and indexes what is
still queued when it opens: a store from before the table (the upgrade build), a build cut short, sources an older
version wrote.

Both lanes search by the same choice of terms (:func:`query_terms`): a question's identifiers first, then its words,
at most 12. When a question has more, the rest are its rarest words by the number of indexed chunks that hold them
(``fts5vocab`` over the lane's own index, each word read through that index's tokenizer), kept in the question's
order, so a long message searches its most telling words rather than its opening. A word no chunk holds can find
nothing, so it only fills a place no held word wants. The counts are over the whole store (they choose terms; the
search itself keeps the person's scope). When they cannot be read, the terms are the first ones, as before.
"""
from __future__ import annotations

from contextlib import closing
import re
import sqlite3
from typing import Iterable, Sequence

from colonymind.util.temporal import strip_arrival_stamps

#: The most terms a keyword query searches (the count is unchanged).
MAX_TERMS = 12
#: The most identifiers kept first ("p-72", "4.2m": searched as the phrase of their pieces).
MAX_IDENTIFIERS = 4
#: Question words that find nothing telling.
STOP = frozenset({"the", "and", "that", "this", "what", "when", "where", "how", "you", "your", "was", "were", "are",
                  "for", "with", "remember", "about"})
#: The most characters of a question that are searched.
MAX_QUERY_CHARS = 4096

# CJK ideographs (with the iteration and closing marks), kana and hangul: scripts written without spaces between words.
_CJK = ("々-〇ぁ-ゖゝ-ゟァ-ヺー-ヿㇰ-ㇿ㐀-䶿一-鿿"
        "豈-﫿ｦ-ﾟᄀ-ᇿㄱ-ㆎꥠ-ꥼ가-힣ힰ-ퟻ"
        "\U00020000-\U0002fa1f\U00030000-\U000323af")
_CJK_RUN = re.compile(f"[{_CJK}]+")
_IDENTIFIER = re.compile(r"[^\W_]+(?:[-./][^\W_]+)+")
_WORD = re.compile(r"\w+")
# Statistics are read in groups of this many terms (older SQLite allows 999 parameters).
_LOOKUP_GROUP = 500

TABLE = "turn_source_stems"
CHUNKS = "turn_source_stem_chunks"
PENDING = "turn_source_stem_pending"


def cjk_bigrams(text: str) -> str:
    """``text`` with every run of CJK ideographs, kana or hangul written as its overlapping two-character pieces
    (a run of one stays one), set apart by spaces: "东京出差" becomes " 东京 京出 出差 ". Other text is unchanged."""
    return _CJK_RUN.sub(_pieces, text)


def _pieces(match: re.Match) -> str:
    run = match.group()
    return " " + " ".join([run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]) + " "


def query_terms(conn: sqlite3.Connection, query: str, *, stemmed: bool = False) -> list[str]:
    """The terms a keyword lane searches for ``query``: its identifiers, then its words (longer than two characters,
    not a stop word; for the stemmed lane, the CJK pieces too), at most :data:`MAX_TERMS`. Past that, the words kept
    are the rarest the index holds (:func:`document_counts`), then the first of those it does not hold, in the
    question's order; position order when the counts cannot be read."""
    # The arrival stamp is when the question came, in every stamped message alike: not search terms.
    text = strip_arrival_stamps(str(query or "")[:MAX_QUERY_CHARS])
    if stemmed:
        text = cjk_bigrams(text)
    # An identifier ("p-72", "B-12", "4.2m") is searched as the phrase of its pieces, first: split into one- and
    # two-letter words it never took part. One whose pieces are all words the search keeps anyway ("answer.json")
    # needs no phrase of its own.
    identifiers = list(dict.fromkeys(
        piece.lower() for piece in _IDENTIFIER.findall(text)
        if any(len(part) <= 2 for part in _WORD.findall(piece))))[:MAX_IDENTIFIERS]
    words = [word.lower() for word in _WORD.findall(text)
             if (len(word) > 2 or stemmed and _CJK_RUN.fullmatch(word)) and word.lower() not in STOP]
    terms = list(dict.fromkeys([*identifiers, *words]))
    if len(terms) <= MAX_TERMS:
        return terms
    rest = terms[len(identifiers):]
    try:
        counts = document_counts(conn, rest, stemmed=stemmed)
    except sqlite3.Error:
        return terms[:MAX_TERMS]
    # A word no chunk holds matches nothing: it goes after every word some chunk holds, never before.
    rarest = sorted(range(len(rest)), key=lambda index: (rest[index] not in counts, counts.get(rest[index], 0), index))
    return identifiers + [rest[index] for index in sorted(rarest[:MAX_TERMS - len(identifiers)])]


def document_counts(conn: sqlite3.Connection, terms: Sequence[str], *, stemmed: bool = False) -> dict[str, int]:
    """How many chunks of the lane's own index hold each of ``terms`` (a term none holds is left out). Each term is
    read through that index's tokenizer, as a question's words are when searched: case and Latin diacritics folded
    and, for the stemmed lane, stemmed, so "moved" is counted as the chunks that hold any form of "move". A term the
    tokenizer splits counts as its rarest piece."""
    from colonymind.memory import producers
    if producers.active():
        path = next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")
        # TEMP fts5 tables need a separate connection; all other producer reads stay query-only.
        with closing(producers.connection(path, query_only=False)) as reader:
            reader.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ATTACH
                                  else sqlite3.SQLITE_OK)
            return _document_counts(reader, terms, stemmed=stemmed)
    return _document_counts(conn, terms, stemmed=stemmed)


def _document_counts(conn: sqlite3.Connection, terms: Sequence[str], *, stemmed: bool) -> dict[str, int]:
    table, tokenizer, name = ((TABLE, "porter unicode61", "stemmed") if stemmed
                              else ("turn_source_search", "unicode61", "written"))
    query, tokens, index = (f"temp.keyword_query_{name}", f"temp.keyword_query_{name}_tokens",
                            f"temp.keyword_index_{name}")
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {query} USING fts5(term, tokenize='{tokenizer}')")
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {tokens} "
                 f"USING fts5vocab(temp, 'keyword_query_{name}', 'instance')")
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {index} USING fts5vocab(main, '{table}', 'row')")
    unique = list(dict.fromkeys(terms))
    conn.execute(f"DELETE FROM {query}")
    conn.executemany(f"INSERT INTO {query}(rowid, term) VALUES (?, ?)", enumerate(unique, 1))
    pieces: dict[int, set[str]] = {}
    for row, token in conn.execute(f"SELECT doc, term FROM {tokens}"):
        pieces.setdefault(int(row), set()).add(token)
    wanted = sorted({token for found in pieces.values() for token in found})
    documents: dict[str, int] = {}
    for start in range(0, len(wanted), _LOOKUP_GROUP):
        group = wanted[start:start + _LOOKUP_GROUP]
        documents.update(conn.execute(
            f"SELECT term, doc FROM {index} WHERE term IN ({','.join('?' * len(group))})", group).fetchall())
    counts: dict[str, int] = {}
    for row, found in pieces.items():
        count = min(int(documents.get(token, 0)) for token in found)
        if count:
            counts[unique[row - 1]] = count
    return counts


def match_expression(terms: Sequence[str]) -> str | None:
    """A full-text query for any of ``terms``, each as a phrase; None when there is none."""
    if not terms:
        return None
    return " OR ".join('"' + term.replace('"', '') + '"' for term in terms)


# --- The stemmed index ----------------------------------------------------------------------------------------------

_FORGET = (f"DELETE FROM {TABLE} WHERE rowid IN (SELECT id FROM {CHUNKS} WHERE turn_id=old.turn_id); "
           f"DELETE FROM {CHUNKS} WHERE turn_id=old.turn_id; "
           f"DELETE FROM {PENDING} WHERE turn_id=old.turn_id;")
_QUEUE = f"INSERT OR IGNORE INTO {PENDING}(turn_id) VALUES (new.turn_id);"
_SCHEMA = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5(content, tokenize='porter unicode61')",
    f"CREATE TABLE IF NOT EXISTS {CHUNKS} (id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT NOT NULL, "
    "role TEXT NOT NULL, chunk INTEGER NOT NULL)",
    f"CREATE INDEX IF NOT EXISTS {CHUNKS}_source ON {CHUNKS}(turn_id)",
    f"CREATE TABLE IF NOT EXISTS {PENDING} (turn_id TEXT PRIMARY KEY)",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_queue AFTER INSERT ON turn_sources BEGIN {_QUEUE} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_change AFTER UPDATE OF turn_id, messages_json ON turn_sources "
    f"BEGIN {_FORGET} {_QUEUE} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_forget AFTER DELETE ON turn_sources BEGIN {_FORGET} END",
)
_OBJECTS = {TABLE, CHUNKS, PENDING, f"{CHUNKS}_source", f"{TABLE}_queue", f"{TABLE}_change", f"{TABLE}_forget"}


def ready(conn: sqlite3.Connection) -> bool:
    """Whether the stemmed index and everything that keeps it are in the store."""
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE name IN (%s)" % ",".join("?" * len(_OBJECTS)), sorted(_OBJECTS))}
    return names == _OBJECTS


def initialize(conn: sqlite3.Connection) -> bool:
    """Create the stemmed index, its chunk map, its queue and the triggers that keep them with ``turn_sources``, in
    the caller's transaction. True when the index was created now: then every existing source is queued."""
    created = not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone()
    for statement in _SCHEMA:
        conn.execute(statement)
    if created:
        conn.execute(f"DELETE FROM {CHUNKS}")
        conn.execute(f"INSERT OR IGNORE INTO {PENDING}(turn_id) SELECT turn_id FROM turn_sources")
    return created


def index(conn: sqlite3.Connection, turn_id: str, chunks: Iterable[tuple[str, str]]) -> None:
    """Index one source's chunks, ``(role, text)`` in the source's chunk order, in place of any rows it had, and take
    it off the queue."""
    forget(conn, turn_id)
    for place, (role, chunk) in enumerate(chunks):
        row = conn.execute(f"INSERT INTO {CHUNKS}(turn_id, role, chunk) VALUES (?, ?, ?)", (turn_id, role, place))
        conn.execute(f"INSERT INTO {TABLE}(rowid, content) VALUES (?, ?)", (row.lastrowid, cjk_bigrams(chunk)))
    conn.execute(f"DELETE FROM {PENDING} WHERE turn_id=?", (turn_id,))


def forget(conn: sqlite3.Connection, turn_id: str) -> None:
    """Delete one source's stemmed rows."""
    conn.execute(f"DELETE FROM {TABLE} WHERE rowid IN (SELECT id FROM {CHUNKS} WHERE turn_id=?)", (turn_id,))
    conn.execute(f"DELETE FROM {CHUNKS} WHERE turn_id=?", (turn_id,))


def queue_all(conn: sqlite3.Connection) -> None:
    """Empty the stemmed index and queue every source (a rebuild from the canonical sources)."""
    conn.execute(f"DELETE FROM {TABLE}")
    conn.execute(f"DELETE FROM {CHUNKS}")
    conn.execute(f"INSERT OR IGNORE INTO {PENDING}(turn_id) SELECT turn_id FROM turn_sources")


__all__ = ["MAX_TERMS", "MAX_IDENTIFIERS", "STOP", "cjk_bigrams", "query_terms", "document_counts", "match_expression",
           "ready", "initialize", "index", "forget", "queue_all"]
