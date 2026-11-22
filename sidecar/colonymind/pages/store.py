"""The lines of the agent's pages: one page per person and one on the agent itself.

A page is a list of short lines of a few kinds. Every line names the source turns it rests on; a line with
no source is not written. The lines live in the source ledger's database (``page_lines``), so there is one
memory and source erasure reaches them: a line stops being read when a turn it cites is erased or forgotten
after the line was written, and the nightly pass then removes it.

A new kind of line is one entry in ``KINDS`` and in the tuple of the pages that carry it.
"""
from __future__ import annotations

import json
from contextlib import closing
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from colonymind.util.temporal import ledger_stamp

#: The subject of the agent's own page. Every other subject is a contact id.
SELF = "self"
#: kind -> heading, in page order.
KINDS: Dict[str, str] = {
    "boundary": "Your boundaries with them",
    "about": "About",
    "like": "Likes",
    "dislike": "Dislikes",
}
PERSON_KINDS = ("boundary", "about")
SELF_KINDS = ("about", "like", "dislike")
NIGHT, AGENT = "night", "agent"
LINE_CHARS = 280
SOURCES_PER_LINE = 8
_INSERT = ("INSERT INTO page_lines(subject, kind, text, sources_json, writer, erasure_mark, written_at) "
           "VALUES (?,?,?,?,?,?,?)")


def initialize(conn) -> None:
    conn.execute('''CREATE TABLE IF NOT EXISTS page_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
        sources_json TEXT NOT NULL, writer TEXT NOT NULL, erasure_mark INTEGER NOT NULL,
        written_at TEXT NOT NULL)''')
    conn.execute('CREATE INDEX IF NOT EXISTS page_lines_subject ON page_lines(subject, kind, id)')
    # When the nightly pass last wrote a page again, on the ledger's clock (a turn's ``ingested_at``).
    conn.execute('''CREATE TABLE IF NOT EXISTS page_rewrites (
        subject TEXT PRIMARY KEY, rewritten_at TEXT NOT NULL)''')


def kinds_of(subject: str) -> Sequence[str]:
    """The kinds of line the page of ``subject`` carries."""
    return SELF_KINDS if subject == SELF else PERSON_KINDS


def clean(text: Any) -> str:
    return " ".join(str(text or "").split())[:LINE_CHARS]


def _sources(sources: Optional[Iterable[Any]]) -> List[str]:
    cited = dict.fromkeys(str(item or "").strip() for item in sources or ())
    return [turn for turn in cited if turn][:SOURCES_PER_LINE]


def _head(conn) -> int:
    return int(conn.execute("SELECT coalesce(max(sequence), 0) FROM source_erasures").fetchone()[0])


def begin(ledger) -> tuple[int, str]:
    """The newest source erasure and the ledger's clock now, read before a rewrite reads its sources: an
    erasure or a turn after this moment is seen as newer than the rewrite (``rewrite``)."""
    with closing(ledger._connect()) as conn:
        return _head(conn), conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0]


def _stands(ledger, conn, row, head: int) -> bool:
    """No cited turn was erased after the line was written, and none is forgotten."""
    sources = json.loads(row["sources_json"])
    if head > row["erasure_mark"] and conn.execute(
            "SELECT 1 FROM source_erasures e LEFT JOIN source_erasure_revisions r USING(sequence) "
            f"WHERE e.sequence>? AND coalesce(r.source_turn_id, e.turn_id) IN ({','.join('?' * len(sources))})",
            (row["erasure_mark"], *sources)).fetchone():
        return False
    from colonymind.memory import curated
    return not any(curated.is_forgotten(ledger, turn, conn=conn) for turn in sources)


def _read(ledger, conn, subject: Optional[str], kinds: Optional[Sequence[str]], *, standing: bool = True):
    sql, args = "SELECT * FROM page_lines", []
    if subject is not None:
        sql, args = sql + " WHERE subject=?", [subject]
    head = _head(conn)
    rows = [{"id": row["id"], "subject": row["subject"], "kind": row["kind"], "text": row["text"],
             "sources": json.loads(row["sources_json"]), "writer": row["writer"], "written_at": row["written_at"]}
            for row in conn.execute(sql + " ORDER BY id", args).fetchall()
            if (kinds is None or row["kind"] in kinds) and _stands(ledger, conn, row, head) == standing]
    order = {kind: place for place, kind in enumerate(KINDS)}
    return sorted(rows, key=lambda item: (order.get(item["kind"], len(order)), item["id"]))


def lines(ledger, subject: str, *, kinds: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """The lines of one page that still stand, in page order."""
    with closing(ledger._connect()) as conn:
        return _read(ledger, conn, subject, kinds)


def add(ledger, subject: str, kind: str, text: Any, sources: Optional[Iterable[Any]], *,
        writer: str = AGENT) -> Optional[Dict[str, Any]]:
    """Write one line and return it. None when it has no text or no source.

    A line the page already holds in the same words keeps its place and gains the new sources.
    """
    if kind not in kinds_of(subject):
        raise ValueError(f"this page has no {kind!r} lines")
    text, cited = clean(text), _sources(sources)
    if not text or not cited:
        return None
    with closing(ledger._connect()) as conn, conn:
        same = next((row for row in _read(ledger, conn, subject, (kind,)) if row["text"] == text), None)
        mark, at = _head(conn), ledger_stamp()
        if same is not None:
            cited = _sources([*same["sources"], *cited])
            conn.execute("UPDATE page_lines SET sources_json=?, erasure_mark=? WHERE id=?",
                         (json.dumps(cited), mark, same["id"]))
            return {**same, "sources": cited}
        line_id = conn.execute(_INSERT, (subject, kind, text, json.dumps(cited), writer, mark, at)).lastrowid
    return {"id": line_id, "subject": subject, "kind": kind, "text": text, "sources": cited, "writer": writer,
            "written_at": at}


def drop(ledger, subject: str, line_id: Any) -> bool:
    """Remove one line of a page by its id."""
    try:
        line_id = int(str(line_id).strip().lstrip("#"))
    except ValueError:
        return False
    with closing(ledger._connect()) as conn, conn:
        return conn.execute("DELETE FROM page_lines WHERE id=? AND subject=?", (line_id, subject)).rowcount > 0


def rewrite(ledger, subject: str, kinds: Sequence[str], entries: Iterable[Mapping[str, Any]], *,
            mark: int, at: str) -> int:
    """Replace the night's lines of ``kinds`` on one page with ``entries`` (``kind``, ``text``, ``sources``).

    Lines the agent wrote itself stay as they are. An entry with no source is not written. ``mark`` and ``at``
    are what ``begin`` read before the entries' sources were read. Returns how many lines were written.
    """
    written = 0
    with closing(ledger._connect()) as conn, conn:
        conn.execute(f"DELETE FROM page_lines WHERE subject=? AND writer=? AND kind IN ({','.join('?' * len(kinds))})",
                     (subject, NIGHT, *kinds))
        held = {(row["kind"], row["text"]) for row in _read(ledger, conn, subject, kinds)}
        for entry in entries:
            kind, text, cited = entry.get("kind"), clean(entry.get("text")), _sources(entry.get("sources"))
            if kind not in kinds or not text or not cited or (kind, text) in held:
                continue
            held.add((kind, text))
            conn.execute(_INSERT, (subject, kind, text, json.dumps(cited), NIGHT, int(mark), at))
            written += 1
        conn.execute("INSERT INTO page_rewrites(subject, rewritten_at) VALUES (?,?) "
                     "ON CONFLICT(subject) DO UPDATE SET rewritten_at=excluded.rewritten_at", (subject, at))
    return written


def sweep(ledger) -> int:
    """Remove every line whose source was erased or forgotten. Returns how many went."""
    with closing(ledger._connect()) as conn, conn:
        gone = [(row["id"],) for row in _read(ledger, conn, None, None, standing=False)]
        conn.executemany("DELETE FROM page_lines WHERE id=?", gone)
    return len(gone)


def move(ledger, old: str, new: str) -> int:
    """Two records of one person became one: the lines of ``old`` join the page of ``new``."""
    with closing(ledger._connect()) as conn, conn:
        conn.execute("DELETE FROM page_rewrites WHERE subject=?", (old,))
        return conn.execute("UPDATE page_lines SET subject=? WHERE subject=?", (new, old)).rowcount


def render(rows: Sequence[Mapping[str, Any]], *, detail: bool = False) -> str:
    """A page as text: a heading per kind, then its lines. ``detail`` adds each line's id and source turns."""
    parts: List[str] = []
    for kind, heading in KINDS.items():
        chosen = [row for row in rows if row["kind"] == kind]
        if not chosen:
            continue
        parts.append(heading + ":")
        parts += [f"- #{row['id']} {row['text']} [{', '.join(row['sources'])}]" if detail else f"- {row['text']}"
                  for row in chosen]
    return "\n".join(parts)
