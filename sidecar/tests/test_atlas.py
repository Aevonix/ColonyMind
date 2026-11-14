"""The atlas: a static, owner-only, offline site over the ledger's cited claims.

Every scenario runs the real source ledger and the real claim projection (a controlled extractor
supplies the model's output); the atlas reads that ledger read-only and is checked from the files it
writes: what each page says, where each line links, and what is absent after an erasure.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import pytest

from colonymind.atlas import AtlasError, build_atlas, check_atlas
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.turns import TurnIdempotencyLedger
from test_source_claim_projection import Model, claim

OWNER, DANA, SAM, GONE = "owner-1", "contact-dana", "contact-sam", "contact-gone"
NOW = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
SECRET = "amber-cobalt-42"


class Instance:
    def __init__(self, root: Path):
        self.root = root
        self.ledger = TurnIdempotencyLedger(root / "turn-idempotency.db")
        self.projection = SourceClaimProjection(self.ledger)
        self.contacts = root / "contacts.db"
        schema = Path(__file__).resolve().parents[1] / "colonymind/contacts/migrations/001_contacts_schema.sql"
        with sqlite3.connect(self.contacts) as conn:
            conn.executescript(schema.read_text())
        self.out = root / "atlas"

    def person(self, contact_id, name, *, deleted=False):
        with sqlite3.connect(self.contacts) as conn:
            conn.execute("INSERT INTO contacts(contact_id, display_name, deleted_at) VALUES (?,?,?)",
                         (contact_id, name, "2026-03-30T00:00:00Z" if deleted else None))

    async def say(self, turn, contact, text, extracted, *, occurred="2026-03-01T09:00:00+00:00"):
        self.ledger.record_source(turn, contact_id=contact, session_id=f"s-{contact}",
                                  messages=[{"role": "user", "content": text}], occurred_at=occurred)
        model = Model({text: extracted})
        while await self.projection.process_one(model):
            pass

    def build(self, **kwargs):
        options = dict(ledger_path=self.root / "turn-idempotency.db", contacts_path=self.contacts,
                       out_dir=self.out, owner_id=OWNER, owner_name="Pat", now=NOW)
        options.update(kwargs)
        return build_atlas(**options)

    def pages(self, out=None):
        out = out or self.out
        return {str(path.relative_to(out)): path.read_text() for path in sorted(out.rglob("*.html"))}

    def page_of(self, contact_id, out=None):
        slug = hashlib.sha256(("person\0" + contact_id).encode()).hexdigest()[:16]
        return (out or self.out) / "people" / f"{slug}.html"


@pytest.fixture
async def world(tmp_path):
    inst = Instance(tmp_path)
    inst.person(OWNER, "Pat")
    inst.person(DANA, "Dana Reyes")
    inst.person(SAM, "Sam Ortiz")
    inst.person(GONE, "Robin Vale", deleted=True)
    old, new = "My office is in River.", "Correction: My office is in Lake, not River."
    await inst.say("t-own-1", OWNER, old, claim(old, "River"), occurred="2026-03-01T09:00:00+00:00")
    await inst.say("t-own-2", OWNER, new, claim(new, "Lake", operation="correct", match_prior=True),
                   occurred="2026-03-03T09:00:00+00:00")
    moved = "Sam Ortiz moved to Lisbon last week."
    await inst.say("t-own-3", OWNER, moved, claim(moved, "Lisbon", subject="Sam Ortiz", predicate="home_city"),
                   occurred="2026-02-10T18:30:00+00:00")
    first, then = "From 2026-03-01, my office is in River.", "Starting 2026-03-12, my office is now in Lake."
    await inst.say("t-dana-1", DANA, first, claim(first, "River", valid_from_text="2026-03-01"),
                   occurred="2026-03-01T10:00:00+00:00")
    await inst.say("t-dana-2", DANA, then, claim(then, "Lake", operation="change", match_prior=True,
                                                  valid_from_text="2026-03-12"), occurred="2026-03-02T10:00:00+00:00")
    code = f"My locker code is {SECRET}."
    await inst.say("t-sam-1", SAM, code, claim(code, SECRET, predicate="locker_code"),
                   occurred="2026-03-05T08:00:00+00:00")
    hostile = "My site is <script>alert(1)</script> https://example.invalid/x now."
    await inst.say("t-sam-2", SAM, hostile, claim(hostile, "<script>alert(1)</script> https://example.invalid/x",
                                                   predicate="website"), occurred="2026-03-06T08:00:00+00:00")
    hidden = "My garden is in Hollow."
    await inst.say("t-gone-1", GONE, hidden, claim(hidden, "Hollow", predicate="garden"),
                   occurred="2026-03-07T08:00:00+00:00")
    return inst


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.hrefs, self.tags = set(), [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for key, value in attrs:
            if key == "id":
                self.ids.add(value)
            if key in {"href", "src"}:
                self.hrefs.append(value or "")


def parsed(text):
    parser = Links()
    parser.feed(text)
    return parser


def lines_of(text):
    """Each cited line: its li element's inner html."""
    return re.findall(r'<li id="c-[0-9a-f]+"[^>]*>(.*?)</li>', text, re.S)


def resolve(out: Path, page: str, href: str):
    target, _, fragment = href.partition("#")
    path = (out / page).parent / unquote(target) if target else out / page
    return path.resolve(), fragment


async def test_every_line_quotes_its_source_with_a_date_and_links_to_the_exact_message(world):
    world.build()
    pages = world.pages()
    checked = 0
    for page, text in pages.items():
        for line in lines_of(text):
            quote = re.search(r"<q>(.*?)</q>", line, re.S)
            date = re.search(r'<time datetime="([^"]+)">(\d{4}-\d{2}-\d{2})</time>', line)
            cite = re.search(r'<a class="cite" href="([^"]+)"', line)
            assert quote and date and cite, (page, line)
            path, fragment = resolve(world.out, page, cite.group(1))
            assert fragment.startswith("m-") and path.is_file()
            target = path.read_text()
            section = re.search(rf'<section id="{fragment}"[^>]*>(.*?)</section>', target, re.S)
            quoted = re.sub(r"<[^>]+>", "", quote.group(1))
            assert section and quoted in re.sub(r"<[^>]+>", "", section.group(1)), (page, fragment)
            checked += 1
    assert checked >= 20
    owner = world.page_of(OWNER).read_text()
    assert '<time datetime="2026-03-03T09:00:00+00:00">2026-03-03</time>' in owner


async def test_each_person_page_holds_only_that_persons_own_words(world):
    world.build()
    owner, dana, sam = (world.page_of(who).read_text() for who in (OWNER, DANA, SAM))
    assert SECRET in sam and SECRET not in owner and SECRET not in dana
    assert "Lisbon" in owner and "Lisbon" not in sam and "Lisbon" not in dana
    for who, text in ((OWNER, owner), (DANA, dana), (SAM, sam)):
        for href in parsed(text).hrefs:
            if "sources/" in href:
                path, _ = resolve(world.out, "people/x.html", href)
                assert f'data-speaker="{hashlib.sha256(("person" + chr(0) + who).encode()).hexdigest()[:16]}"' \
                    in path.read_text(), (who, href)


async def test_a_deleted_contact_has_no_page_and_no_quote_anywhere(world):
    world.build()
    assert not world.page_of(GONE).exists()
    assert not any("Hollow" in text or "Robin Vale" in text for text in world.pages().values())


async def test_superseded_and_corrected_values_are_history_linked_to_their_successor(world):
    world.build()
    owner, dana = world.page_of(OWNER).read_text(), world.page_of(DANA).read_text()
    river = next(line for line in lines_of(owner) if "River" in line and "Correction" not in line)
    lake = re.search(r'<li id="(c-[0-9a-f]+)"[^>]*class="line current"[^>]*>(?:(?!</li>).)*Correction: My office',
                     owner, re.S)
    assert lake, owner
    assert "corrected" in river and f'href="#{lake.group(1)}"' in river
    moved = next(line for line in lines_of(dana) if "From 2026-03-01" in line)
    assert "superseded" in moved and "2026-03-12" in moved
    # History sits after the current value in the same fact group, and is marked, not hidden.
    river_id = re.search(r'<li id="(c-[0-9a-f]+)"[^>]*class="line corrected"[^>]*>(?:(?!</li>).)*'
                         r'<q>My office is in River\.</q>', owner, re.S)
    assert river_id and owner.index(lake.group(1)) < owner.index(river_id.group(1))


async def test_a_value_not_yet_valid_or_no_longer_valid_is_marked_by_the_build_clock(world):
    later = "From 2026-05-01, my parking spot is P7."
    await world.say("t-sam-3", SAM, later, claim(later, "P7", predicate="parking_spot", valid_from_text="2026-05-01"),
                    occurred="2026-03-10T08:00:00+00:00")
    ended = "Until 2026-03-15, my desk is by the door."
    await world.say("t-sam-4", SAM, ended, claim(ended, "by the door", predicate="desk", valid_to_text="2026-03-15"),
                    occurred="2026-03-11T08:00:00+00:00")
    world.build()
    sam = world.page_of(SAM).read_text()
    upcoming = next(line for line in lines_of(sam) if "P7" in line)
    expired = next(line for line in lines_of(sam) if "by the door" in line)
    assert "not yet valid: from 2026-05-01" in upcoming and "no longer valid after 2026-03-15" in expired
    assert 'class="line upcoming"' in sam and 'class="line expired"' in sam
    world.build(now=datetime(2026, 5, 2, tzinfo=timezone.utc))
    assert 'class="line upcoming"' not in world.page_of(SAM).read_text()


async def test_links_resolve_and_the_site_loads_nothing_from_anywhere(world):
    world.build()
    pages = world.pages()
    assert {"index.html", "people.html", "months.html", "topics.html"} <= set(pages)
    for page, text in pages.items():
        tree = parsed(text)
        assert not {"script", "img", "link", "iframe", "object", "embed", "form"} & set(tree.tags), page
        assert "Content-Security-Policy" in text and "default-src 'none'" in text
        assert "url(" not in text and "@import" not in text
        for href in tree.hrefs:
            assert not re.match(r"^[a-z][a-z0-9+.-]*:|^/", href, re.I), (page, href)
            path, fragment = resolve(world.out, page, href)
            assert path.is_file() and world.out.resolve() in path.parents, (page, href)
            if fragment:
                assert fragment in parsed(path.read_text()).ids, (page, href)
    hostile = world.page_of(SAM).read_text()
    assert "<script>" not in hostile and "&lt;script&gt;" in hostile
    assert check_atlas(world.out)["problems"] == []


async def test_names_cross_link_exactly_to_the_person_and_the_topic(world):
    report = world.build()
    owner = world.page_of(OWNER).read_text()
    sam_href = re.search(r'<a class="name" href="([^"]+)">Sam Ortiz</a>', owner)
    assert sam_href and resolve(world.out, "people/x.html", sam_href.group(1))[0] == world.page_of(SAM).resolve()
    topic = next(text for page, text in world.pages().items() if page.startswith("topics/") and "Lisbon" in text)
    assert "<h1>Sam Ortiz</h1>" in topic and 'class="name"' in topic
    sam = world.page_of(SAM).read_text()
    assert "What others said about Sam Ortiz" in sam
    assert report["name_links"] >= 2


async def test_months_bucket_by_when_it_was_said_and_undated_sources_say_so(world):
    text = "My bike is blue."
    world.ledger.record_source("t-undated", contact_id=DANA, session_id="s-dana",
                               messages=[{"role": "user", "content": text}])
    model = Model({text: claim(text, "blue", predicate="bike_colour")})
    while await world.projection.process_one(model):
        pass
    world.build()
    pages = world.pages()
    assert "months/2026-02.html" in pages and "Lisbon" in pages["months/2026-02.html"]
    assert "Lisbon" not in pages["months/2026-03.html"]
    undated = next(line for line in lines_of(world.page_of(DANA).read_text()) if "bike" in line)
    assert "recorded" in undated and "time said unknown" in undated
    # A timezone moves a late-evening statement into the owner's local month and day.
    world.build(out_dir=world.root / "tz", timezone_name="Asia/Tokyo")
    assert "Lisbon" in world.pages(world.root / "tz")["months/2026-02.html"]
    assert "2026-02-11" in world.page_of(OWNER, world.root / "tz").read_text()


async def test_a_rebuild_rewrites_only_the_pages_whose_content_changed(world):
    first = world.build()
    assert first["written"] == first["pages"] and first["unchanged"] == 0
    sam_page = world.page_of(SAM)
    before = (sam_page.stat().st_mtime_ns, sam_page.stat().st_ino)
    second = world.build()
    assert second["written"] == 0 and second["deleted"] == 0 and second["unchanged"] == first["pages"]
    text = "My desk is by the window."
    await world.say("t-dana-3", DANA, text, claim(text, "by the window", predicate="desk"),
                    occurred="2026-03-20T10:00:00+00:00")
    third = world.build()
    assert 0 < third["written"] < third["pages"] // 2
    assert (sam_page.stat().st_mtime_ns, sam_page.stat().st_ino) == before
    assert third["lines"] == first["lines"] + 1


async def test_a_tampered_or_missing_page_is_restored(world):
    world.build()
    world.page_of(DANA).write_text("tampered")
    (world.out / "months.html").unlink()
    report = world.build()
    assert report["written"] == 2
    assert "tampered" not in world.page_of(DANA).read_text()
    assert check_atlas(world.out)["problems"] == []


async def test_episodes_and_quoted_preferences_are_quotes_on_person_and_month_pages_only(world):
    report = ("The nightly import took four minutes with the new index and eleven without it, "
              "measured on the same data twice.")
    await world.say("t-dana-ep", DANA, report, {
        "representation": "episode", "memory_kind": "substantive_event", "evidence": report,
        "recall_reason": "Use the measured import times when planning the next import.",
        "operation": "assert", "prior_claim_id": None, "event_at_text": None}, occurred="2026-03-22T10:00:00+00:00")
    liking = "I prefer tea without sugar unless I am ill."
    await world.say("t-dana-pref", DANA, liking, claim(liking, liking, predicate="tea", memory_kind="preference",
                                                        representation="preference"), occurred="2026-03-23T10:00:00+00:00")
    report_row = world.build()
    dana = world.page_of(DANA).read_text()
    assert "reported episode" in dana and "eleven without it" in dana and "I prefer tea without sugar" in dana
    pages = world.pages()
    assert "eleven without it" in pages["months/2026-03.html"]
    assert not any("eleven without it" in text for page, text in pages.items() if page.startswith("topics/"))
    assert report_row["topics"] == 1


async def test_the_size_bound_keeps_the_newest_lines_and_says_what_it_dropped(world):
    full = world.build(out_dir=world.root / "full")
    bounded = world.build(out_dir=world.root / "small", max_bytes=full["bytes"] * 2 // 3)
    assert bounded["bytes"] <= full["bytes"] * 2 // 3
    assert 0 < bounded["dropped_lines"] < full["lines"] and bounded["lines"] + bounded["dropped_lines"] == full["lines"]
    pages = world.pages(world.root / "small")
    assert "Lisbon" not in "".join(pages.values())         # the oldest statement goes first
    assert f"{bounded['dropped_lines']} older" in pages["index.html"]
    assert check_atlas(world.root / "small")["problems"] == []
    with pytest.raises(AtlasError):
        world.build(out_dir=world.root / "tiny", max_bytes=100)


async def test_the_site_is_owner_only_deterministic_and_never_writes_the_ledger(world):
    db = world.root / "turn-idempotency.db"
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    world.build()
    world.build(out_dir=world.root / "again")
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest
    assert world.pages() == world.pages(world.root / "again")
    assert stat.S_IMODE(world.out.stat().st_mode) == 0o700
    for path in world.out.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, path


async def test_it_never_writes_into_a_directory_that_is_not_an_atlas(world):
    world.out.mkdir()
    keep = world.out / "notes.txt"
    keep.write_text("mine")
    with pytest.raises(AtlasError):
        world.build()
    assert keep.read_text() == "mine" and sorted(world.out.iterdir()) == [keep]


async def test_the_cli_builds_into_the_instance_and_checks_it(world, monkeypatch, capsys):
    import json
    from colonymind import cli
    # main() exports the selected instance into the process environment: give it a private copy.
    environ = {key: value for key, value in os.environ.items() if not key.startswith("COLONYMIND_")}
    environ.update(HOME=str(world.root), COLONYMIND_OWNER_CONTACT_ID=OWNER)
    monkeypatch.setattr(os, "environ", environ)
    out = world.root / "cli-atlas"
    for argv in (["colonymind", "--instance", str(world.root), "atlas", "build", "--out", str(out), "--json"],
                 ["colonymind", "--instance", str(world.root), "atlas", "check", "--out", str(out), "--json"]):
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as done:
            cli.main()
        assert done.value.code == 0, capsys.readouterr()
        result = json.loads(capsys.readouterr().out)
    assert result["problems"] == [] and (out / "index.html").is_file()
    assert world.page_of(OWNER, out).is_file() and not world.page_of(GONE, out).exists()
    monkeypatch.setattr("sys.argv", ["colonymind", "--instance", str(world.root), "atlas", "build"])
    with pytest.raises(SystemExit) as done:
        cli.main()
    assert done.value.code == 0 and (world.root / "atlas" / "index.html").is_file()
    assert os.path.getsize(world.root / "atlas" / "index.html") > 0


async def test_a_planted_link_in_the_atlas_directory_never_redirects_a_write(world, tmp_path_factory):
    world.build()
    victim = tmp_path_factory.mktemp("victim") / "ledger-copy.db"
    victim.write_bytes(b"precious ledger bytes")
    (world.out / f".atlas-manifest.json.{os.getpid()}.tmp").symlink_to(victim)
    (world.out / f".index.html.{os.getpid()}.tmp").symlink_to(victim)
    world.page_of(DANA).unlink()
    world.page_of(DANA).symlink_to(victim)
    try:
        world.build()
    except AtlasError:
        pass
    assert victim.read_bytes() == b"precious ledger bytes"
    # A folder of the atlas replaced by a link to somewhere else is refused, and nothing is written there.
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    sources = world.out / "sources"
    for page in sources.iterdir():
        page.unlink()
    sources.rmdir()
    sources.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(AtlasError):
        world.build()
    assert list(elsewhere.iterdir()) == []


def _manifest(out):
    import json
    return json.loads((out / "atlas-manifest.json").read_text())


def _nowhere(out, text):
    return not any(text in path.read_text(errors="ignore") for path in out.rglob("*") if path.is_file())


async def test_the_size_bound_covers_every_file_the_atlas_writes(world):
    world.build(out_dir=world.root / "full")
    html = sum(path.stat().st_size for path in (world.root / "full").rglob("*.html"))
    report = world.build(out_dir=world.root / "exact", max_bytes=html)
    total = sum(path.stat().st_size for path in (world.root / "exact").rglob("*") if path.is_file())
    assert total <= html and report["bytes"] == total


async def test_the_line_cap_keeps_the_newest_said_and_counts_every_line_it_leaves_out(tmp_path):
    inst = Instance(tmp_path)
    inst.person(DANA, "Dana Reyes")
    # Recorded in this order; said September, January and June of one year.
    for turn, said, city in (("t-sep", "2025-09-10T10:00:00+00:00", "Sepville"),
                             ("t-jan", "2025-01-10T10:00:00+00:00", "Janton"),
                             ("t-jun", "2025-06-10T10:00:00+00:00", "Junburg")):
        text = f"I stayed in {city}."
        await inst.say(turn, DANA, text, claim(text, city, predicate=f"stay_{turn[2:]}"), occurred=said)
    report = inst.build(max_lines=1)
    pages = inst.pages()
    assert "Sepville" in "".join(pages.values()) and "Junburg" not in "".join(pages.values())
    assert report["lines"] == 1 and report["dropped_lines"] == 2
    assert "2 older lines" in pages["index.html"]


async def test_a_message_the_owner_disputed_is_not_a_current_fact(world):
    import json
    from colonymind.turns.idempotency import canonical_turn_digest
    text = "My desk is by the window."
    await world.say("t-dana-3", DANA, text, claim(text, "by the window", predicate="desk"),
                    occurred="2026-03-20T10:00:00+00:00")
    with sqlite3.connect(world.root / "turn-idempotency.db") as conn:
        messages = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id='t-dana-3'").fetchone()[0])
    world.ledger.append_source_annotation(
        contact_id=DANA, session_id="s-contact-dana", annotation_id="review-2", source_id="t-dana-3",
        source_version=canonical_turn_digest(messages), excerpt="My desk is by the window",
        correction="That is false. The desk is by the door.", author_principal="owner")
    world.build()
    shown = [(state, body) for text in world.pages().values()
             for state, body in re.findall(r'<li id="c-[0-9a-f]+" class="line (\w+)"[^>]*>(.*?)</li>', text, re.S)
             if "by the window" in body]
    assert shown and all(state == "disputed" for state, _ in shown), shown
    assert all("a correction is on record" in body for _, body in shown)


@pytest.mark.parametrize("hops", [1, 2])
async def test_an_atlas_built_through_a_linked_folder_checks_clean_through_it(world, tmp_path_factory, hops):
    real = tmp_path_factory.mktemp("real") / "atlas"
    real.mkdir(mode=0o700)
    via = real
    for hop in range(hops):
        link = tmp_path_factory.mktemp(f"hop{hop}") / "atlas"
        link.symlink_to(via, target_is_directory=True)
        via = link
    world.build(out_dir=via)
    assert check_atlas(via)["problems"] == []
    assert check_atlas(real)["problems"] == []
    world.build(out_dir=via)
    assert check_atlas(via, ledger_path=world.root / "turn-idempotency.db")["problems"] == []
    # A folder inside the atlas replaced by a link is still refused, however the atlas itself is reached.
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    people = real / "people"
    for page in people.iterdir():
        (elsewhere / page.name).write_bytes(page.read_bytes())
        page.unlink()
    people.rmdir()
    people.symlink_to(elsewhere, target_is_directory=True)
    problems = check_atlas(via)["problems"]
    assert problems and all("a link, not a page the atlas wrote" in p for p in problems if p.startswith("people/"))
    assert any(p.startswith("people/") and "a link" in p for p in problems)


from colonymind.memory import curated
async def test_an_erasure_removes_the_source_page_and_every_quote_on_rebuild(world):
    world.build()
    sources_before = {page for page in world.pages() if page.startswith('sources/')}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(world.ledger)
    curated.forget_sources(world.ledger, SAM, source_ids=['t-sam-1'])
    assert historical_bytes(world.ledger) == _native_before_withdrawal
    report = world.build()
    pages = world.pages()
    assert report['deleted'] >= 1 and report['written'] >= 1
    assert not any((SECRET in text for text in pages.values()))
    assert not any((SECRET in path.read_text(errors='ignore') for path in world.out.rglob('*') if path.is_file()))
    assert len({page for page in pages if page.startswith('sources/')}) == len(sources_before) - 1
    assert check_atlas(world.out, ledger_path=world.root / 'turn-idempotency.db')['problems'] == []

async def test_an_erasure_after_the_build_makes_the_check_fail_until_rebuilt(world):
    world.build()
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(world.ledger)
    curated.forget_sources(world.ledger, SAM, source_ids=['t-sam-1'])
    assert historical_bytes(world.ledger) == _native_before_withdrawal
    problems = check_atlas(world.out, ledger_path=world.root / 'turn-idempotency.db')['problems']
    assert any(('erased after' in problem for problem in problems))
    world.build()
    assert check_atlas(world.out, ledger_path=world.root / 'turn-idempotency.db')['problems'] == []

async def test_a_build_that_dies_midway_leaves_no_page_a_later_erasure_cannot_find(world, monkeypatch):
    from colonymind.atlas import site
    world.build()
    text = 'My badge number is 7781-OLIVE.'
    await world.say('t-dana-3', DANA, text, claim(text, '7781-OLIVE', predicate='badge'), occurred='2026-03-21T10:00:00+00:00')
    real = site._private_file

    def dies_after_a_source_page(path, data):
        real(path, data)
        if path.parent.name == 'sources':
            raise OSError('disk went away')
    monkeypatch.setattr(site, '_private_file', dies_after_a_source_page)
    with pytest.raises(OSError):
        world.build()
    monkeypatch.setattr(site, '_private_file', real)
    assert any(('7781-OLIVE' in text for text in world.pages().values()))
    (world.out / 'sources' / '.leftover.html.1.tmp').write_text('7781-OLIVE')
    assert any(('did not finish' in problem for problem in check_atlas(world.out)['problems']))
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(world.ledger)
    curated.forget_sources(world.ledger, DANA, source_ids=['t-dana-3'])
    assert historical_bytes(world.ledger) == _native_before_withdrawal
    world.build()
    assert not any(('7781-OLIVE' in path.read_text(errors='ignore') for path in world.out.rglob('*') if path.is_file()))
    assert check_atlas(world.out, ledger_path=world.root / 'turn-idempotency.db')['problems'] == []

async def test_an_attributed_correction_is_shown_on_its_message_and_leaves_with_its_erasure(world):
    import json
    from colonymind.turns.idempotency import canonical_turn_digest
    with sqlite3.connect(world.root / 'turn-idempotency.db') as conn:
        messages = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id='t-dana-1'").fetchone()[0])
    note = world.ledger.append_source_annotation(contact_id=DANA, session_id='s-contact-dana', annotation_id='review-1', source_id='t-dana-1', source_version=canonical_turn_digest(messages), excerpt='my office is in River', correction='It was the River annex, not the main building.', author_principal='owner')
    world.build()
    line = next((line for line in lines_of(world.page_of(DANA).read_text()) if 'From 2026-03-01' in line))
    assert 'a correction is on record' in line
    cite = re.search('<a class="cite" href="([^"]+)"', line).group(1)
    path, fragment = resolve(world.out, 'people/x.html', cite)
    section = re.search(f'<section id="{fragment}"[^>]*>(.*?)</section>', path.read_text(), re.S).group(1)
    assert 'It was the River annex, not the main building.' in section
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(world.ledger)
    curated.forget_sources(world.ledger, DANA, source_ids=[note['source_id']])
    assert historical_bytes(world.ledger) == _native_before_withdrawal
    world.build()
    assert 'River annex' not in path.read_text() and 'since been erased' in path.read_text()

async def test_overlapping_builds_leave_no_page_a_later_erasure_cannot_find(world, monkeypatch):
    import threading
    from colonymind.atlas import site
    world.build()
    real, b_read, resume, errors = (site.read_snapshot, threading.Event(), threading.Event(), [])

    def paused(*args, **kwargs):
        snapshot = real(*args, **kwargs)
        if threading.current_thread().name == 'builder-b' and (not b_read.is_set()):
            b_read.set()
            resume.wait(10)
        return snapshot

    def run():
        try:
            world.build()
        except Exception as error:
            errors.append(error)
    monkeypatch.setattr(site, 'read_snapshot', paused)
    b = threading.Thread(target=run, name='builder-b')
    b.start()
    assert b_read.wait(10)
    text = 'My badge number is 7781-OLIVE.'
    await world.say('t-dana-3', DANA, text, claim(text, '7781-OLIVE', predicate='badge'), occurred='2026-03-21T10:00:00+00:00')
    a = threading.Thread(target=run, name='builder-a')
    a.start()
    a.join(2)
    resume.set()
    b.join(20)
    a.join(20)
    assert not errors and (not a.is_alive()) and (not b.is_alive())
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(world.ledger)
    curated.forget_sources(world.ledger, DANA, source_ids=['t-dana-3'])
    assert historical_bytes(world.ledger) == _native_before_withdrawal
    world.build()
    assert _nowhere(world.out, '7781-OLIVE')
    assert check_atlas(world.out, ledger_path=world.root / 'turn-idempotency.db')['problems'] == []

async def test_a_page_dropped_from_the_manifest_is_still_removed_when_its_source_is_erased(world):
    import json
    from colonymind.atlas.render import source_page
    text = 'My badge number is 7781-OLIVE.'
    await world.say('t-dana-3', DANA, text, claim(text, '7781-OLIVE', predicate='badge'), occurred='2026-03-21T10:00:00+00:00')
    world.build()
    manifest = _manifest(world.out)
    assert manifest['pages'].pop(source_page('t-dana-3'))
    (world.out / 'atlas-manifest.json').write_text(json.dumps(manifest))
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(world.ledger)
    curated.forget_sources(world.ledger, DANA, source_ids=['t-dana-3'])
    assert historical_bytes(world.ledger) == _native_before_withdrawal
    world.build()
    assert _nowhere(world.out, '7781-OLIVE')
    assert check_atlas(world.out, ledger_path=world.root / 'turn-idempotency.db')['problems'] == []

async def test_erasing_a_source_with_no_line_on_the_site_still_brings_the_check_up_to_date(world):
    world.build()
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(world.ledger)
    curated.forget_sources(world.ledger, GONE, source_ids=['t-gone-1'])
    assert historical_bytes(world.ledger) == _native_before_withdrawal
    world.build()
    world.build()
    assert check_atlas(world.out, ledger_path=world.root / 'turn-idempotency.db')['problems'] == []

async def test_erasures_that_outrun_every_attempt_fail_the_build_and_the_check(world, monkeypatch):
    from colonymind.atlas import site
    world.build()
    real = site.read_snapshot
    turns = [(DANA, 't-dana-1'), (OWNER, 't-own-3'), (DANA, 't-dana-2'), (OWNER, 't-own-1')][:site.SNAPSHOT_ATTEMPTS - 1]
    turns.append((SAM, 't-sam-1'))

    def racing(*args, **kwargs):
        snapshot = real(*args, **kwargs)
        if turns:
            contact, turn = turns.pop(0)
            from test_curated_memory import historical_bytes
            _native_before_withdrawal = historical_bytes(world.ledger)
            curated.forget_sources(world.ledger, contact, source_ids=[turn])
            assert historical_bytes(world.ledger) == _native_before_withdrawal
        return snapshot
    monkeypatch.setattr(site, 'read_snapshot', racing)
    with pytest.raises(AtlasError):
        world.build()
    assert _manifest(world.out)['state'] != 'complete'
    assert check_atlas(world.out)['problems']
    monkeypatch.setattr(site, 'read_snapshot', real)
    world.build()
    assert _nowhere(world.out, SECRET)
    assert check_atlas(world.out, ledger_path=world.root / 'turn-idempotency.db')['problems'] == []
