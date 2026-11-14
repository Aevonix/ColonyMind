"""The atlas pages: plain HTML with one inline stylesheet pinned by its hash, no scripts, nothing fetched.

Pages: one per person (only their own words), per month (by when it was said, in the owner's
timezone), per topic (a claim's subject, whoever said it) and per source (the cited messages, with
the quoted spans marked), plus an index and three directories. Every line carries its quote, its
date and a link to the exact message it came from; history lines link to what replaced them.
Names are cross-linked by exact, unambiguous match. The output is a pure function of its input,
so an unchanged ledger renders byte-identical pages.
"""
from __future__ import annotations

import base64
import hashlib
import html
import posixpath
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from colonymind.beliefs.source_claims import norm_value

from .snapshot import Line, utc

Chunk = List[Tuple[str, List[Line]]]

CSS = (":root{color-scheme:light dark;--bg:#fbfaf7;--fg:#1d1d1f;--mute:#67676d;--rule:#e2dfd8;--acc:#1f5f8b;"
       "--mark:#fff0a0}@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#e8e6e1;--mute:#9d9da3;"
       "--rule:#2d2d31;--acc:#86bde3;--mark:#5c4d00}}body{margin:0;background:var(--bg);color:var(--fg);"
       "font:16px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif}nav{padding:.6rem 1rem;border-bottom:1px solid "
       "var(--rule);font-size:.9rem}main{max-width:52rem;margin:0 auto;padding:1rem}a{color:var(--acc)}"
       "ul{list-style:none;padding:0}li{margin:.35rem 0}li.line{border-left:3px solid var(--acc);padding:.35rem .8rem;"
       "margin:.7rem 0}li.line:not(.current){border-left-color:var(--rule);color:var(--mute)}"
       "li.line:not(.current) .fact b{text-decoration:line-through}q,blockquote{display:block;margin:.25rem 0;"
       "white-space:pre-wrap;overflow-wrap:anywhere}q{font-style:italic}.meta,.pred,.said,.note{color:var(--mute);"
       "font-size:.86rem}.state{font-weight:600}mark{background:var(--mark);color:inherit}"
       "h1{font-size:1.5rem}h2{font-size:1.05rem;margin-top:1.8rem;border-bottom:1px solid var(--rule)}")
CSP = ("default-src 'none'; style-src 'sha256-" + base64.b64encode(hashlib.sha256(CSS.encode()).digest()).decode()
       + "'; base-uri 'none'; form-action 'none'")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
STATE_RANK = {"current": 0, "upcoming": 1, "disputed": 2, "expired": 3, "superseded": 4, "corrected": 5, "withdrawn": 6}
MESSAGE_CHARS, WINDOW_CHARS, INDEX_LINES = 4000, 400, 20
_WORD = re.compile(r"\w+")


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def plural(count: int, one: str, many: str = "") -> str:
    return f"{count} {one if count == 1 else many or one + 's'}"


def slug(kind: str, key: str) -> str:
    return hashlib.sha256(f"{kind}\0{key}".encode()).hexdigest()[:16]


def person_page(contact_id: str) -> str:
    return f"people/{slug('person', contact_id)}.html"


def topic_page(subject_key: str) -> str:
    return f"topics/{slug('topic', subject_key)}.html"


def source_page(turn_id: str) -> str:
    return f"sources/{slug('source', turn_id)}.html"


def anchor(claim_id: str) -> str:
    return "c-" + hashlib.sha256(claim_id.encode()).hexdigest()[:16]


def message_anchor(message_hash: str) -> str:
    return "m-" + message_hash[:16]


def numbered(path: str, number: int) -> str:
    return path if number == 1 else f"{path[:-5]}-{number}.html"


def rel(page: str, target: str) -> str:
    return posixpath.relpath(target, posixpath.dirname(page) or ".")


def is_topic(line: Line) -> bool:
    return line.kind != "episode" and line.subject_key not in {"", "speaker"} and not line.subject_key.startswith("episode:")


@dataclass
class Stats:
    name_links: int = 0
    cite_links: int = 0


@dataclass
class Rendered:
    pages: Dict[str, bytes]
    stats: Stats
    counts: Dict[str, int] = field(default_factory=dict)


class Names:
    """Exact-name cross-links: a whole-word, case-sensitive match of a unique name. Candidates are
    found by the name's first word, so the cost is per word of text, not per name known."""

    def __init__(self, targets: Dict[str, str]):
        self.targets = targets
        self.by_first: Dict[str, List[str]] = defaultdict(list)
        for name in sorted(targets):
            first = _WORD.match(name)
            if (first and len(name) >= 3 and re.search(r"[^\W\d_]", name) and re.search(r"\w$", name)):
                self.by_first[first.group(0)].append(name)
        for names in self.by_first.values():
            names.sort(key=lambda name: (-len(name), name))

    def link(self, text: str, page: str, stats: Stats, skip: Iterable[str] = ()) -> str:
        skip = set(skip)
        out, position = [], 0
        for word in _WORD.finditer(text):
            start = word.start()
            if start < position:
                continue
            for name in self.by_first.get(word.group(0), ()):
                end = start + len(name)
                if not text.startswith(name, start) or (end < len(text) and _WORD.match(text[end])):
                    continue
                target = self.targets[name]
                if target in skip:
                    break
                out.append(esc(text[position:start]))
                out.append(f'<a class="name" href="{esc(rel(page, target))}">{esc(name)}</a>')
                stats.name_links += 1
                position = end
                break
        out.append(esc(text[position:]))
        return "".join(out)


def _chunks(groups: Sequence[Tuple[str, List[Line]]], size: int) -> List[Chunk]:
    """Whole groups per page while they fit; a group larger than a page is split across pages."""
    pages: List[Chunk] = [[]]
    used = 0
    for title, lines in groups:
        while lines:
            room = size - used
            if room <= 0 or (used and len(lines) > room):
                pages.append([])
                used, room = 0, size
            part, lines = lines[:room], lines[room:]
            pages[-1].append((title, part))
            used += len(part)
    return pages if pages[0] else [[]]


class Site:
    def __init__(self, lines: List[Line], *, names: Dict[str, str], messages: Dict[Tuple[str, str], str],
                 owner_id: Optional[str], owner_name: Optional[str], tz: tzinfo, page_lines: int,
                 dropped: int, notes: Optional[Dict[Tuple[str, str], list]] = None):
        self.lines, self.messages, self.tz, self.notes = lines, messages, tz, notes or {}
        self.owner_id, self.page_lines, self.dropped = owner_id, max(1, page_lines), dropped
        self.stats = Stats()
        self.by_id = {line.id: line for line in lines}
        speakers = sorted({line.speaker for line in lines})
        self.person_names = {speaker: names.get(speaker) or (owner_name if speaker == owner_id and owner_name else None)
                             or speaker for speaker in speakers}
        surfaces: Dict[str, Counter] = defaultdict(Counter)
        for line in lines:
            if is_topic(line):
                surfaces[line.subject_key][line.subject] += 1
        self.topic_names = {key: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
                            for key, counter in surfaces.items()}
        # Exact names: a unique person wins; a topic keeps every spelling it was said with.
        people: Dict[str, List[str]] = defaultdict(list)
        for speaker, name in self.person_names.items():
            people[name].append(person_page(speaker))
        targets: Dict[str, str] = {}
        for key, counter in surfaces.items():
            for surface in counter:
                targets[surface] = topic_page(key)
        for name, pages in people.items():
            if len(pages) == 1:
                targets[name] = pages[0]
        self.names = Names(targets)
        self.home: Dict[str, str] = {}

    # -- small pieces ---------------------------------------------------------------------------

    def local(self, value: Optional[str]) -> Optional[datetime]:
        moment = utc(value)
        return moment.astimezone(self.tz) if moment else None

    def day(self, value: Optional[str]) -> str:
        moment = self.local(value)
        return moment.date().isoformat() if moment else "?"

    def last_day(self, end_exclusive: Optional[str]) -> str:
        """The last local day before an exclusive end."""
        moment = self.local(end_exclusive)
        return (moment - timedelta(microseconds=1)).date().isoformat() if moment else "?"

    def month(self, line: Line) -> str:
        moment = self.local(line.when)
        return moment.strftime("%Y-%m") if moment else "undated"

    def person_title(self, speaker: str) -> str:
        name = self.person_names[speaker]
        return f"{name} (you)" if speaker == self.owner_id else name

    def href(self, page: str, target: str, fragment: str = "") -> str:
        path = "" if target == page else rel(page, target)
        return esc(path + ("#" + fragment if fragment else ""))

    def subject_html(self, line: Line, page: str, skip: Iterable[str]) -> str:
        if line.subject_key == "speaker":
            return f'<span class="subj">{esc(self.person_names[line.speaker])}</span>'
        return f'<span class="subj">{self.names.link(line.subject, page, self.stats, skip)}</span>'

    def short(self, line: Line) -> str:
        text = line.value if line.kind == "fact" else line.evidence
        return text if len(text) <= 80 else text[:79] + "…"

    def state_html(self, line: Line, page: str) -> str:
        if line.status == "current":
            return ""
        if line.status == "expired":
            return f'<span class="state">no longer valid after {esc(self.day(line.valid_to))}</span>'
        if line.status == "upcoming":
            return f'<span class="state">not yet valid: from {esc(self.day(line.valid_from))}</span>'
        if line.status == "disputed":
            return '<span class="state">disputed: not current while a correction of its message is on record</span>'
        if line.status == "withdrawn":
            return '<span class="state">withdrawn: what corrected it is no longer on record</span>'
        verb = "corrected by" if line.status == "corrected" else "superseded by"
        successor = self.by_id.get(line.successor or "")
        if successor is None or successor.id not in self.home:
            return f'<span class="state">{verb} a later statement not shown here</span>'
        return (f'<span class="state">{verb} <a class="successor" href="{self.href(page, self.home[successor.id], anchor(successor.id))}">'
                f'{esc(self.short(successor))}</a> ({esc(self.day(successor.when))})</span>')

    def line_html(self, line: Line, page: str, *, speaker: bool, skip: Iterable[str] = ()) -> str:
        skip = list(skip)
        if line.kind == "fact":
            fact = (f'{self.subject_html(line, page, skip)} <span class="pred">{esc(line.predicate.replace("_", " "))}</span> '
                    f'<b>{self.names.link(line.value, page, self.stats, skip)}</b>')
        elif line.kind == "quote":
            fact = f'{self.subject_html(line, page, skip)} <span class="pred">{esc(line.predicate.replace("_", " "))}</span>'
        else:
            fact = '<span class="pred">reported episode</span>'
        said = line.said_at is not None
        cite = (f'<a class="cite" href="{self.href(page, source_page(line.turn_id), message_anchor(line.message_hash))}">'
                f'{"said" if said else "recorded"} <time datetime="{esc(line.when)}">{esc(self.day(line.when))}</time></a>'
                + ("" if said else " (time said unknown)"))
        self.stats.cite_links += 1
        meta = [cite]
        if speaker:
            meta.append(f'<a class="who" href="{self.href(page, person_page(line.speaker))}">{esc(self.person_names[line.speaker])}</a>')
        if line.valid_from and line.valid_to:
            meta.append(f"valid {esc(self.day(line.valid_from))} to {esc(self.day(line.valid_to))}")
        elif line.valid_from:
            meta.append(f"valid from {esc(self.day(line.valid_from))}")
        elif line.valid_to:
            meta.append(f"valid until {esc(self.day(line.valid_to))}")
        if line.event_at and line.event_until:
            meta.append(f"happened sometime between {esc(self.day(line.event_at))} and "
                        f"{esc(self.last_day(line.event_until))}")
        elif line.event_at:
            meta.append(f"happened {esc(self.day(line.event_at))}")
        if line.transcript:
            meta.append("from a voice transcript, unverified")
        if line.annotated:
            meta.append("a correction is on record for this message")
        state = self.state_html(line, page)
        if state:
            meta.append(state)
        return (f'<li id="{anchor(line.id)}" class="line {line.status}" data-claim="{esc(line.id)}">'
                f'<div class="fact">{fact}</div><q>{self.names.link(line.evidence, page, self.stats, skip)}</q>'
                f'<div class="meta">{" · ".join(meta)}</div></li>')

    def document(self, page: str, title: str, body: str, *, attrs: str = "") -> bytes:
        nav = " · ".join(f'<a href="{esc(rel(page, target))}">{label}</a>' for target, label in (
            ("index.html", "Atlas"), ("people.html", "People"), ("months.html", "Months"), ("topics.html", "Topics")))
        return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                f'<meta http-equiv="Content-Security-Policy" content="{CSP}">'
                '<meta name="referrer" content="no-referrer"><meta name="robots" content="noindex, nofollow">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{esc(title)} · Atlas</title><style>{CSS}</style></head>'
                f'<body{attrs}><nav>{nav}</nav><main><h1>{esc(title)}</h1>{body}</main></body></html>\n').encode()

    @staticmethod
    def pager(page: str, base: str, count: int) -> str:
        if count <= 1:
            return ""
        return '<nav class="pager">Pages: ' + " ".join(
            f'<a href="{esc(rel(page, numbered(base, number)))}">{number}</a>' for number in range(1, count + 1)) + "</nav>"

    # -- page families -------------------------------------------------------------------------------

    def plan(self, base: str, groups: List[Tuple[str, List[Line]]], *, register: bool) -> Dict[str, Chunk]:
        """Split a page family into numbered pages; a person page is each of its lines' home."""
        chunks = _chunks(groups, self.page_lines)
        planned = {numbered(base, number): chunk for number, chunk in enumerate(chunks, 1)}
        if register:
            for page, chunk in planned.items():
                for _, lines in chunk:
                    for line in lines:
                        self.home.setdefault(line.id, page)
        return planned

    def render_grouped(self, base: str, chunks: Dict[str, Chunk], title: str, intro: str, *, speaker: bool,
                       skip: Iterable[str] = ()) -> Dict[str, bytes]:
        pages = {}
        count = len(chunks)
        for page, chunk in chunks.items():
            body = [intro, self.pager(page, base, count)]
            for heading, lines in chunk:
                body.append(f"<h2>{heading}</h2><ul>")
                body.extend(self.line_html(line, page, speaker=speaker, skip=skip) for line in lines)
                body.append("</ul>")
            body.append(self.pager(page, base, count))
            pages[page] = self.document(page, title, "".join(body))
        return pages

    def directory(self, base: str, title: str, entries: List[str], intro: str = "") -> Dict[str, bytes]:
        size = self.page_lines
        chunks = [entries[start:start + size] for start in range(0, len(entries), size)] or [[]]
        pages = {}
        for number, chunk in enumerate(chunks, 1):
            page = numbered(base, number)
            body = intro + self.pager(page, base, len(chunks)) + "<ul>" + "".join(chunk) + "</ul>"
            pages[page] = self.document(page, title, body)
        return pages

    @staticmethod
    def by_state(lines: List[Line]) -> List[Line]:
        return sorted(lines, key=lambda line: STATE_RANK[line.status])  # stable: newest first within a state

    def render(self) -> Rendered:
        lines = self.lines
        speakers = sorted({line.speaker for line in lines},
                          key=lambda speaker: (speaker != self.owner_id, self.person_names[speaker].casefold(), speaker))
        by_speaker: Dict[str, List[Line]] = defaultdict(list)
        by_topic: Dict[str, List[Line]] = defaultdict(list)
        month_lines: Dict[str, List[Line]] = defaultdict(list)
        for line in lines:                                  # newest first, so every bucket is too
            by_speaker[line.speaker].append(line)
            if is_topic(line):
                by_topic[line.subject_key].append(line)
            month_lines[self.month(line)].append(line)
        # Person pages: only that person's own sources, one group per fact they stated.
        person_chunks = {}
        for speaker in speakers:
            groups: Dict[Tuple[str, str], List[Line]] = {}
            for line in by_speaker[speaker]:
                groups.setdefault((line.subject_key, line.predicate if line.kind != "episode" else ""), []).append(line)
            titled = []
            for (key, predicate), members in groups.items():
                first = members[0]
                if first.kind == "episode":
                    heading = "Reported episode"
                else:
                    subject = self.person_names[speaker] if key == "speaker" else self.topic_names.get(key, first.subject)
                    heading = f"{esc(subject)} · {esc(predicate.replace('_', ' '))}"
                titled.append((heading, self.by_state(members)))
            person_chunks[speaker] = self.plan(person_page(speaker), titled, register=True)
        topic_keys = sorted(self.topic_names, key=lambda key: (self.topic_names[key].casefold(), key))
        topic_chunks = {}
        for key in topic_keys:
            groups2: Dict[str, List[Line]] = {}
            for line in by_topic[key]:
                groups2.setdefault(line.predicate, []).append(line)
            titled = [(esc(predicate.replace("_", " ")), self.by_state(members)) for predicate, members in groups2.items()]
            topic_chunks[key] = self.plan(topic_page(key), titled, register=False)
        month_chunks = {}
        for month, members in month_lines.items():
            days: Dict[str, List[Line]] = {}
            for line in members:
                days.setdefault(self.day(line.when), []).append(line)
            month_chunks[month] = self.plan(f"months/{month}.html", [(esc(day), group) for day, group in days.items()],
                                            register=False)

        pages: Dict[str, bytes] = {}
        for speaker in speakers:
            base = person_page(speaker)
            own = by_speaker[speaker]
            name = self.person_names[speaker]
            said = len({(line.turn_id, line.message_hash) for line in own})
            intro = (f'<p class="note">{plural(len(own), "line")} from {plural(said, "message")}, in '
                     f"{'your' if speaker == self.owner_id else 'their'} own words. Each quote links to "
                     "the message it came from.</p>")
            key = norm_value(name)
            if key in self.topic_names and speaker != self.owner_id:
                intro += (f'<p><a href="{esc(rel(base, topic_page(key)))}">What others said about {esc(name)}</a></p>')
            pages.update(self.render_grouped(base, person_chunks[speaker], self.person_title(speaker), intro,
                                             speaker=False, skip=[base]))
        for key in topic_keys:
            base = topic_page(key)
            name = self.topic_names[key]
            target = self.names.targets.get(name)
            intro = '<p class="note">Everything said about this, by whoever said it.</p>'
            if target and target.startswith("people/"):
                intro += f'<p>Also a person you know: <a class="name" href="{esc(rel(base, target))}">{esc(name)}</a></p>'
                self.stats.name_links += 1
            pages.update(self.render_grouped(base, topic_chunks[key], name, intro, speaker=True, skip=[base]))
        months = sorted(month_chunks, reverse=True)
        for month in months:
            base = f"months/{month}.html"
            title = f"{MONTHS[int(month[5:]) - 1]} {month[:4]}" if month != "undated" else "Undated"
            pages.update(self.render_grouped(base, month_chunks[month], title,
                                             '<p class="note">What was said this month, newest first.</p>', speaker=True))
        pages.update(self.sources())
        pages.update(self.directory("people.html", "People", [
            f'<li><a href="{esc(person_page(speaker))}">{esc(self.person_title(speaker))}</a> '
            f'<span class="note">{plural(len(by_speaker[speaker]), "line")}</span></li>'
            for speaker in speakers]))
        pages.update(self.directory("months.html", "Months", [
            f'<li><a href="months/{esc(month)}.html">{esc(month)}</a> '
            f'<span class="note">{plural(len(month_lines[month]), "line")}</span></li>'
            for month in months]))
        pages.update(self.directory("topics.html", "Topics", [
            f'<li><a href="{esc(topic_page(key))}">{esc(self.topic_names[key])}</a> '
            f'<span class="note">{plural(len(by_topic[key]), "line")}</span></li>'
            for key in topic_keys]))
        pages["index.html"] = self.index(speakers, months, topic_keys)
        counts = {"people": len(speakers), "months": len(months), "topics": len(topic_keys),
                  "sources": sum(1 for page in pages if page.startswith("sources/"))}
        return Rendered(pages=pages, stats=self.stats, counts=counts)

    def sources(self) -> Dict[str, bytes]:
        pages: Dict[str, bytes] = {}
        by_turn: Dict[str, Dict[str, List[Line]]] = defaultdict(lambda: defaultdict(list))
        for line in self.lines:
            by_turn[line.turn_id][line.message_hash].append(line)
        for turn_id in sorted(by_turn):
            page = source_page(turn_id)
            sections = []
            speaker = ""
            for message_hash in sorted(by_turn[turn_id], key=lambda h: (by_turn[turn_id][h][0].when, h)):
                members = by_turn[turn_id][message_hash]
                first = members[0]
                speaker = first.speaker
                text = self.messages[(turn_id, message_hash)]
                derived = "".join(
                    f'<li class="derived"><a href="{self.href(page, self.home[line.id], anchor(line.id))}">'
                    f'{esc(self.short(line))}</a> <span class="note">{esc(line.status)}</span></li>'
                    for line in self.by_state(members) if line.id in self.home)
                when = (f'said <time datetime="{esc(first.when)}">{esc(self.day(first.when))}</time>' if first.said_at else
                        f'recorded <time datetime="{esc(first.when)}">{esc(self.day(first.when))}</time> (time said unknown)')
                sections.append(
                    f'<section id="{message_anchor(message_hash)}" class="message"><p class="said">'
                    f'<a class="who" href="{self.href(page, person_page(speaker))}">{esc(self.person_names[speaker])}</a> · {when}</p>'
                    f'<blockquote>{self.excerpt(text, members, page)}</blockquote>{self.corrections(turn_id, message_hash)}'
                    f'<p class="note">Taken from it:</p><ul>{derived}</ul></section>')
            pages[page] = self.document(page, f"Message from {self.person_names[speaker]}", "".join(sections),
                                        attrs=f' data-speaker="{slug("person", speaker)}"')
        return pages

    def corrections(self, turn_id: str, message_hash: str) -> str:
        out = []
        for at, correction in self.notes.get((turn_id, message_hash), ()):
            if correction is None:
                out.append('<p class="note">A correction was on record for this message; it has since been erased.</p>')
            else:
                out.append(f'<p class="note">A correction is on record ({esc(self.day(at))}): <q>{esc(correction)}</q></p>')
        return "".join(out)

    def excerpt(self, text: str, lines: List[Line], page: str) -> str:
        """The message with its quoted spans marked; a long one shows windows around the quotes."""
        spans = []
        for line in lines:
            span = line.span if line.span and text[line.span[0]:line.span[1]] == line.evidence else None
            if span is None and line.evidence and line.evidence in text:
                start = text.index(line.evidence)
                span = (start, start + len(line.evidence))
            if span:
                spans.append(span)
        spans = sorted(spans)
        marks: List[Tuple[int, int]] = []
        for start, end in spans:
            if marks and start <= marks[-1][1]:
                marks[-1] = (marks[-1][0], max(marks[-1][1], end))
            else:
                marks.append((start, end))
        if len(text) <= MESSAGE_CHARS or not marks:
            windows = [(0, min(len(text), MESSAGE_CHARS))]
        else:
            windows = []
            for start, end in marks:
                window = (max(0, start - WINDOW_CHARS), min(len(text), end + WINDOW_CHARS))
                if windows and window[0] <= windows[-1][1]:
                    windows[-1] = (windows[-1][0], max(windows[-1][1], window[1]))
                else:
                    windows.append(window)
        out = []
        for index, (low, high) in enumerate(windows):
            if index or low:
                out.append("… ")
            cursor = low
            for start, end in marks:
                if end <= low or start >= high:
                    continue
                start, end = max(start, low), min(end, high)
                out.append(self.names.link(text[cursor:start], page, self.stats))
                out.append(f"<mark>{self.names.link(text[start:end], page, self.stats)}</mark>")
                cursor = end
            out.append(self.names.link(text[cursor:high], page, self.stats))
            if high < len(text):
                out.append(" …")
        return "".join(out)

    def index(self, speakers: List[str], months: List[str], topics: List[str]) -> bytes:
        page = "index.html"
        messages = len({(line.turn_id, line.message_hash) for line in self.lines})
        body = [f'<p>{plural(len(self.lines), "cited line")} from {plural(messages, "message")}: '
                f'{plural(len(speakers), "person", "people")}, {plural(len(months), "month")}, '
                f'{plural(len(topics), "topic")}. Every line quotes the message it came from '
                "and links to it; changed and corrected values stay as history.</p>"]
        if self.dropped:
            body.append(f'<p class="note">{plural(self.dropped, "older line")} {"is" if self.dropped == 1 else "are"} '
                        "not shown: the atlas is size-bounded.</p>")
        body.append('<p><a href="people.html">People</a> · <a href="months.html">Months</a> · '
                    '<a href="topics.html">Topics</a></p><h2>Newest</h2><ul>')
        body.extend(self.line_html(line, page, speaker=True) for line in self.lines[:INDEX_LINES])
        body.append("</ul>")
        return self.document(page, "Atlas", "".join(body))
