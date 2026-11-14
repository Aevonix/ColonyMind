"""Behaviour signals and a bounded standing-lesson learner; no weights train on reply text."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable


@dataclass(frozen=True)
class DashCounts:
    characters: int
    long_dashes: int
    em_dashes: int
    en_dashes: int
    horizontal_bars: int
    two_em_dashes: int
    three_em_dashes: int
    small_em_dashes: int
    en_dash_ranges: int
    spaced_hyphens: int
    double_hyphens: int


def _prose(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(">"):
            continue
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence is not None:
            if (marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence)
                    and not marker[2].strip()):
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        lines.append(line)
    text = "".join(lines)
    text = re.sub(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", "", text, flags=re.DOTALL)
    return re.sub(r"(?:https?://|ftp://|www\.)[^\s<>]+", "", text, flags=re.IGNORECASE)


def long_dash_counts(text: str) -> DashCounts:
    """Count glyphs in prose, retaining inline quotations and numeric en-dash ranges.

    Each glyph counts once (including U+2E3A/U+2E3B). Characters are Unicode code
    points after exclusions, including remaining whitespace. Substitutes are occurrences.
    """
    text = _prose(text)
    counts = [text.count(char) for char in "\u2014\u2013\u2015\u2e3a\u2e3b\ufe58"]
    return DashCounts(len(text), sum(counts), *counts,
                      len(re.findall(r"(?<=\d)\u2013(?=\d)", text)),
                      len(re.findall(r"(?<=[^\W\d_])[ \t]+-[ \t]+(?=[^\W\d_])", text)), text.count("--"))


@dataclass(frozen=True)
class ReplyRow:
    at: datetime
    platform: str
    model: str
    text: str
    turn_id: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class OperationRow:
    at: datetime
    platform: str
    model: str
    turn_id: str
    tool_call_id: str
    tool: str
    success: bool | None
    failure_kind: str | None


def _key(row):
    at = row.at if row.at.tzinfo else row.at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc).date().isoformat(), row.platform, row.model


def _group(key):
    return dict(zip(("day", "platform", "model"), key))


def reply_counts(rows: Iterable[ReplyRow]) -> list[dict]:
    """Daily reply counts and rates by captured platform/model; no text in the result."""
    groups = defaultdict(Counter)
    sessions = defaultdict(Counter)
    for row in rows:
        counts = long_dash_counts(row.text)
        group = groups[_key(row)]
        group.update(asdict(counts))
        group["replies"] += 1
        group["replies_with_long_dash"] += bool(counts.long_dashes)
        group["replies_with_substitutes"] += bool(counts.spaced_hyphens or counts.double_hyphens)
        sessions[_key(row)][row.session_id] += 1
    result = []
    for key, counts in sorted(groups.items()):
        per_session = sorted(sessions[key].values())
        result.append({**_group(key), **counts, "sessions": len(per_session),
                       "median_replies_per_session": statistics.median(per_session),
                       "long_dash_reply_share": counts["replies_with_long_dash"] / counts["replies"],
                       "long_dashes_per_1000_characters": (1000 * counts["long_dashes"] / counts["characters"]
                                                          if counts["characters"] else 0.0),
                       "substitute_reply_share": counts["replies_with_substitutes"] / counts["replies"]})
    return result


def tool_counts(rows: Iterable[OperationRow]) -> list[dict]:
    """Deduplicate captured call ids across envelopes and model splits.

    Model response markers never add calls or call failures. Failed parse markers add one request failure per
    ingress turn, attributed to the first such marker's model. Other request failures
    keep their recorded kind (in particular a refusal is never teachable).
    """
    groups = {}
    seen, parsed = set(), set()
    for row in rows:
        identity = row.tool_call_id
        if identity in seen:
            continue
        seen.add(identity)
        kind = row.failure_kind or "unknown"
        if row.tool == "model_response" and kind == "parse_failure" and row.success is False:
            if row.turn_id in parsed:
                continue
            parsed.add(row.turn_id)
        key = _key(row)
        group = groups.setdefault(key, {"calls": 0, "call_failures": 0, "request_failures": 0,
                                        "teachable_failures": 0,
                                        "parse_failures": 0, "tool_error": 0,
                                        "unknown_success": 0, "by_kind": Counter(), "top_tools": Counter()})
        group["calls"] += row.tool != "model_response"
        group["unknown_success"] += row.tool != "model_response" and row.success is None
        if row.success is not False:
            continue
        group["request_failures" if row.tool == "model_response" else "call_failures"] += 1
        group["by_kind"][kind] += 1
        group["tool_error"] += kind == "tool_error"
        parse = row.tool == "model_response" and kind == "parse_failure"
        group["parse_failures"] += parse
        if kind == "malformed_arguments" or parse:
            group["teachable_failures"] += 1
            group["top_tools"][row.tool] += 1
    return [{**_group(key), **group,
             "failure_rate": group["call_failures"] / group["calls"] if group["calls"] else None,
             "teachable_failure_rate": group["teachable_failures"] / group["calls"] if group["calls"] else None,
             "by_kind": dict(sorted(group["by_kind"].items())),
             "top_tools": [{"tool": name, "teachable_failures": count} for name, count in
                           sorted(group["top_tools"].items(), key=lambda item: (-item[1], item[0]))[:10]]}
            for key, group in sorted(groups.items())]


def read_rows(conn, owner_id: str, *, since: datetime | None = None,
              until: datetime | None = None) -> tuple[list[ReplyRow], list[OperationRow], dict]:
    """Read canonical owner sources and ingress root/event envelopes in one ledger snapshot.

    ``since`` is inclusive, ``until`` exclusive. Reuse drift's SELF_TURN_SQL and its
    native-observation exclusion, without drift's time window, row cap or concatenation.
    Platform comes from source_channels in this same store. No other application's
    database, session naming convention or current configured model supplies metadata.
    """
    from colonymind.mind.consolidate import SELF_TURN_SQL
    from colonymind.reports.instance import parse_time
    from colonymind.work.ledger import current_refs, root_live

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    columns = {row[1] for row in conn.execute("PRAGMA table_info(turn_sources)")}
    channels = "source_channels" in tables
    missing_tool_tables = sorted({"turn_sources", "work_envelopes", "source_attribution_invalidations"} - tables)
    coverage = {"reply_platform_column": ("turn_sources.platform" if "platform" in columns else
                                           "source_channels.platform" if channels else None),
                "reply_model_column": "turn_sources.model" if "model" in columns else None,
                "operation_platform_column": ("turn_sources.platform" if "platform" in columns else
                                               "source_channels.platform" if channels else None),
                "operation_model_column": "work_envelopes.envelope_json.operations[].model",
                "turn_sources_available": "turn_sources" in tables,
                "work_envelopes_available": "work_envelopes" in tables,
                "tool_coverage": "unavailable" if missing_tool_tables else "available",
                "missing_tool_tables": missing_tool_tables,
                "reply_platform_known": 0, "reply_model_known": 0,
                "operation_platform_known": 0, "operation_model_known": 0,
                "skipped_rows": 0}
    replies, operations = [], []
    if "turn_sources" not in tables or not owner_id:
        return replies, operations, coverage
    join = "LEFT JOIN source_channels c ON c.turn_id=s.turn_id AND c.contact_id=s.contact_id" if channels else ""
    platform = "s.platform" if "platform" in columns else "c.platform" if channels else "NULL"
    sources = conn.execute(
        f"SELECT s.*, {platform} AS captured_platform FROM turn_sources s {join} "
        f"WHERE s.contact_id=? AND s.scope='person' AND {SELF_TURN_SQL} AND "
        "json_extract(s.messages_json, '$[0]._native_tool_observation') IS NULL ORDER BY s.rowid", (owner_id,))
    for source in sources:
        at = parse_time(source["occurred_at"] or source["ingested_at"])
        if at is None:
            coverage["skipped_rows"] += 1
            continue
        at = at.astimezone(timezone.utc)
        if (since is not None and at < since) or (until is not None and at >= until):
            continue
        messages = json.loads(source["messages_json"])
        platform = source["captured_platform"] or "unknown"
        model = (source["model"] if "model" in columns else None) or "unknown"
        said = next((item["content"] for item in reversed(messages)
                     if isinstance(item, dict) and item.get("role") == "assistant"
                     and isinstance(item.get("content"), str) and item["content"].strip()), None)
        if said is not None:
            replies.append(ReplyRow(at, platform, model, said, str(source["turn_id"]), str(source["session_id"])))
            coverage["reply_platform_known"] += platform != "unknown"
            coverage["reply_model_known"] += model != "unknown"
        if missing_tool_tables:
            continue
        eligible = []
        for envelope in conn.execute("SELECT * FROM work_envelopes WHERE source_id=? AND root_turn_id=? "
                                     "AND contact_id=? AND session_id=? AND kind IN ('root','event') AND invalidated=0 "
                                     "ORDER BY rowid", (source["turn_id"], source["turn_id"], owner_id,
                                                       source["session_id"])):
            if not root_live(conn, envelope) or not current_refs(
                    conn, json.loads(envelope["source_refs_json"]), owner_id):
                continue
            work = json.loads(envelope["envelope_json"])
            if work.get("root_turn_id") != source["turn_id"] or work.get("parent_turn_id") is not None:
                continue
            eligible.append((envelope["kind"], work))
        # Events supplement an eligible ingress root; they cannot revive a revoked root.
        if not any(kind == "root" for kind, _ in eligible):
            continue
        for _, work in eligible:
            for op in work.get("operations", []):
                if not current_refs(conn, op.get("source_refs", []), owner_id):
                    continue
                if not op.get("tool_call_id") or not op.get("tool"):
                    coverage["skipped_rows"] += 1
                    continue
                operations.append(OperationRow(at, platform, op.get("model") or "unknown", source["turn_id"],
                                                op["tool_call_id"], op["tool"], op.get("success"),
                                                op.get("failure_kind")))
    # Availability counts use the same call identities as the counter, before parse markers collapse per turn.
    unique = {row.tool_call_id: row for row in reversed(operations)}.values()
    for row in unique:
        coverage["operation_platform_known"] += row.platform != "unknown"
        coverage["operation_model_known"] += row.model != "unknown"
    return replies, operations, coverage


# The carrier holds lesson text; this learner stores only lesson identities and strength.
from contextlib import closing

import numpy as np

from ...identity.policy import replacements
from ...instinct.outcomes import Sources
from .. import _io
from ..admission import Admission
from ..budgets import Budget, BudgetCheck
from ..builds import Build
from ..certificates import GateRecord
from ..comparator import configure
from ..gates import Check, GateReport
from ..nightly import register_factory
from ..objectives import Objective

OBJECTIVE = Objective(name='behaviour', model='standing_lessons', kind='behaviour', target_type='binary',
                      losses={'verified_outcome': 'logistic'}, blend={'verified_outcome': 1.}, text=False)
from ...config import DEFAULTS as _CONFIG_DEFAULTS
DECLARATION = _CONFIG_DEFAULTS['plasticity']['learners']['behaviour-tool-failure']


def read_verdicts(conn, owner_id, since=None, until=None):
    """Read explicit, non-stale owner verdicts by ingress root identity."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'work_verdict_pending' not in tables:
        return []
    from ...reports.instance import parse_time
    result = []
    for row in conn.execute("SELECT root_id,verdict,verdict_at FROM work_verdict_pending "
                            "WHERE contact_id=? AND state!='stale' ORDER BY verdict_at,id", (owner_id,)):
        at = parse_time(re.sub(r':\d{8}$', '', row['verdict_at']))
        if at is not None and (since is None or at >= since) and (until is None or at < until):
            result.append((row['root_id'], row['verdict']))
    return result


def _signal(reply, operations):
    dash = long_dash_counts(reply.text)
    counts = tool_counts(operations)
    failures = sum(c['teachable_failures'] for c in counts)
    attempts = sum(c['calls'] + c['parse_failures'] for c in counts)
    return {'long_dash': bool(dash.long_dashes), 'tool_failure': bool(failures) if operations else None,
            'substitute_rate': bool(dash.spaced_hyphens or dash.double_hyphens),
            'median_length': len(reply.text),
            'tool_failure_rate': (failures, attempts) if operations else None}


SIGNALS = {name: (lambda reply, operations, name=name: {
    **(values := _signal(reply, operations)), 'target': values[name]})
           for name in ('long_dash', 'tool_failure')}


def _bytes(text):
    return np.frombuffer(text.encode(), dtype=np.uint8).copy()


def _text(state, key):
    return np.asarray(state.get(key, []), dtype=np.uint8).tobytes().decode()


@dataclass(frozen=True)
class Pin:
    learner: str
    target: str
    lesson_id: str
    level: int
    rank: int
    example: str
    closure: str = ''


def pins(runtime):
    """Read the immutable serving snapshot without entering the nightly lock."""
    return runtime.behaviour_pins if runtime is not None and runtime.enabled else ()


def publish_pins(runtime):
    """The current serving closures are the sole registry of standing lessons."""
    if runtime is None or not runtime.enabled:
        return []
    from ...mind.lessons import STANDING_LESSONS
    result = []
    with runtime.lock:
        for name, binding in runtime.loaded.items():
            if binding.declaration.get('implementation') != 'behaviour':
                continue
            pinned = runtime.pin(name, 'owner')
            if pinned is None or pinned.closure.mode == 'base':
                continue
            target = binding.declaration['target']
            state = pinned.state
            result.append(Pin(name, target['signal'], _text(state, 'lesson'), int(state['level'][0]),
                              target.get('rank', 1), binding.driver.example_text(state), pinned.closure.id))
        runtime.behaviour_pins = tuple(sorted(result, key=lambda p: (p.rank, p.learner))[:STANDING_LESSONS])


class Rows:
    """Read live sources; only the contrast example's reference is retained."""
    def __init__(self, driver, sources):
        self.driver, self.sources = driver, sources

    def rows(self, objective, *, since=None):
        return iter(())

    def source_of(self, row):
        return None

    def refs_resting_on(self, ids):
        example = self.driver.example()
        return ['example'] if example and example['turn_id'] in ids else []

    def erase(self, refs):
        if 'example' in refs:
            _io.remove(self.driver.store.directory / 'example.json')

    def snapshot(self):
        driver = self.driver
        if driver.lessons is None or not driver.owner_id:
            return [], [], []
        with closing(driver.lessons.ledger._connect()) as conn:
            conn.execute('BEGIN')
            replies, operations, _ = read_rows(conn, driver.owner_id, until=driver.clock())
            verdicts = read_verdicts(conn, driver.owner_id, until=driver.clock())
        hidden = self.sources.withheld(r.turn_id for r in replies) | self.sources.erased(r.turn_id for r in replies)
        return ([r for r in replies if r.turn_id not in hidden],
                [op for op in operations if op.turn_id not in hidden], verdicts)


class Driver:
    minimum_labels = {}  # Initial lesson gates declare no row minimum.

    def __init__(self, *, store, controller, source, declaration, clock):
        if declaration.get('chain') is not None:
            raise ValueError('behaviour forbids chain')
        if declaration.get('mode') != 'rebuild' or declaration.get('personal_data') is not True:
            raise ValueError('behaviour requires rebuild and personal_data')
        if declaration.get('budget', {}).get('type') != 'ladder' or declaration['budget']['bound'] != 3:
            raise ValueError('behaviour requires a three-level ladder')
        target, lesson = declaration.get('target', {}), declaration.get('lesson', {})
        if not isinstance(target, dict) or target.get('signal') not in SIGNALS:
            raise ValueError('behaviour: unknown signal')
        if type(target.get('rank', 1)) is not int or target.get('rank', 1) <= 0:
            raise ValueError('behaviour: rank must be a positive integer')
        if not isinstance(lesson, dict) or lesson.get('source') not in {'owner', 'relay', 'reflector'}:
            raise ValueError('behaviour: unknown lesson source')
        if lesson['source'] in {'owner', 'relay'} and not str(lesson.get('topic', '')).strip():
            raise ValueError('behaviour: a lesson needs its topic')
        from ...config import validate_behaviour_restarts
        validate_behaviour_restarts(getattr(controller, 'behaviour_restarts', []))
        self.contrast = replacements(declaration.get('contrast', []))
        self.store, self.controller, self.source, self.clock = store, controller, source, clock
        self.declaration, self.signal = declaration, target['signal']
        self.settings = configure(controller, declaration)
        self.objective = OBJECTIVE
        self.admission = Admission(OBJECTIVE, learner=store.name, source_of=source.source_of, clock=clock)
        self.rows = Rows(self, getattr(source.source, 'sources', None) or Sources(store.root.parent))
        source.source = self.rows
        self.lessons, self.owner_id, self.comparator, self.runtime = None, None, None, None
        self.last_period = None
        self.idle_reason = None
        store.budgets.declare(Budget(store.name, 'ladder', declaration['budget']['bound']))

    def bind_mind(self, mind):
        self.lessons, self.owner_id = mind.lessons, mind.lessons.owner_id
        if self.runtime is not None:
            self.runtime._publish_pins()

    def period(self):
        self.last_period = None

    def ladder(self):
        try:
            return _io.read_json(self.store.directory / 'behaviour-ladder.json')
        except FileNotFoundError:
            return {}

    def _lesson(self):
        if self.lessons is None or not self.lessons.enabled:
            return None
        spec = self.declaration['lesson']
        if spec['source'] == 'owner':
            from ...mind.lessons import OWNER_CLASS, CURRENT
            from ...mind.drives import slug
            topic = OWNER_CLASS + slug(spec['topic'])
            return next((l for l in self.lessons.all() if l.verified == 'owner'
                         and l.relayed is None and l.signature == topic and l.status in CURRENT), None)
        if spec['source'] == 'relay':
            fields = {key: spec[key] for key in ('kind', 'topic', 'title', 'when_to_use', 'content')}
            lesson, _ = self.lessons.relay(fields, said_at=spec['said_at'], note=spec['note'],
                recorded_by=f'host-config:plasticity.learners.{self.store.name}')
            return lesson if lesson and lesson.status in ('active', 'candidate') else None
        # Reflector admission and selection are P2. Core carries no teaching text.
        return None

    def build(self):
        self.idle_reason = None
        serving = self.store.revisions.serving()
        if serving.mode != 'base':
            state = self.store.revisions.state(serving)
        else:
            lesson = self._lesson()
            state = {}
            if lesson is None:
                self.idle_reason = ('awaiting_owner_lesson' if self.declaration['lesson']['source'] == 'owner'
                                    else 'awaiting_lesson')
            else:
                history = self.ladder().get(lesson.id, {})
                level = 1 + max(history.get('withdrawn', []), default=0)
                if level == 2 and not self.contrast:
                    level = 3
                if history.get('halted') or level > 3:
                    self.idle_reason = 'ladder_exhausted'
                else:
                    example_key = self._make_example() if level >= 2 and self.contrast else ''
                    state = {'lesson': _bytes(lesson.id), 'level': np.array([level]), 'example': _bytes(example_key)}
        return Build('rebuild', state, _io.state_digest(state), frozenset(), (), 'owner', 0,
                     teacher_losses={'verified_outcome': 0.}, teacher_rows={'verified_outcome': 0})

    def propose(self, build):
        serving = self.store.revisions.serving()
        if build.digest == serving.state:
            return serving
        closure = self.store.revisions.compose(state=build.digest, objective=OBJECTIVE.identity, calibration=(),
            dependents={}, parent=serving.id, manifests=(), periods=(), audience='owner', mode='rebuild')
        self.controller.propose(closure, build.state)
        self.store.budgets.record(BudgetCheck(self.store.name, build.digest, 'ladder', 3, int(build.state['level'][0])))
        return closure

    def example(self):
        try:
            return _io.read_json(self.store.directory / 'example.json')
        except FileNotFoundError:
            return None

    def _make_example(self):
        replies, operations, _ = self.rows.snapshot()
        ops = defaultdict(list)
        for op in operations:
            ops[op.turn_id].append(op)
        # The last evaluated window supplied the target failure that advances the ladder.
        window = self.comparator.summary().get('window')
        for reply in sorted(replies, key=lambda r: r.at, reverse=True):
            if window and reply.at < _io.parse(window):
                continue
            if not SIGNALS[self.signal](reply, ops[reply.turn_id])['target']:
                continue
            for match in re.finditer(r'[^.!?\n]+[.!?]?', reply.text):
                if not any(pattern.search(match[0]) for pattern, _ in self.contrast):
                    continue
                key = _io.digest([reply.turn_id, match.start(), match.end()])
                _io.write_json(self.store.directory / 'example.json',
                               {'turn_id': reply.turn_id, 'start': match.start(), 'end': match.end(), 'key': key})
                self.store.record_source_refs({'example': (reply.turn_id,)})
                return key
        return ''

    def example_text(self, state):
        example = self.example()
        if not example or example['key'] != _text(state, 'example'):
            return ''
        turn_id = example['turn_id']
        if self.lessons is None or turn_id in (self.rows.sources.withheld([turn_id]) |
                                               self.rows.sources.erased([turn_id])):
            return ''
        with closing(self.lessons.ledger._connect()) as conn:
            source = conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=? AND contact_id=?",
                                  (turn_id, self.owner_id)).fetchone()
        if source is None:
            return ''
        messages = json.loads(source['messages_json'])
        text = next((m['content'] for m in reversed(messages) if m.get('role') == 'assistant'
                     and isinstance(m.get('content'), str) and m['content'].strip()), '')
        before = text[example['start']:example['end']].strip()
        after = before
        for pattern, replacement in self.contrast:
            after = pattern.sub(replacement, after)
        return f'Before: {before} After: {after}' if before != after else ''

    def render(self, state):
        lesson = self.lessons.get(_text(state, 'lesson')) if self.lessons else None
        if lesson is None or lesson.status not in ('active', 'candidate'):
            return ''
        return lesson.standing_section(int(state['level'][0]), self.example_text(state))

    def promoted(self, closure):
        """Freeze counts at the first promotion of this lesson, before late ingestion."""
        self.comparator.freeze_baseline(self.observations(closure))

    def observations(self, closure):
        promoted = next(e['at'] for e in self.controller.events()
                        if e['event'] == 'promoted' and e['closure'] == closure.id)
        replies, operations, verdicts = self.rows.snapshot()
        by_turn, judged = defaultdict(list), dict(verdicts)
        for op in operations:
            by_turn[op.turn_id].append(op)
        exposed = set()
        for use in self.lessons._use_rows(self.clock()):
            context = use.context if isinstance(use.context, dict) else {}
            if context.get('standing', {}).get('learner') == self.store.name:
                exposed.add(context.get('turn_id'))
        rows = []
        for reply in replies:
            values = SIGNALS[self.signal](reply, by_turn[reply.turn_id])
            verdict = judged.get(reply.turn_id)
            rows.append({'at': _io.stamp(reply.at), 'session_id': reply.session_id, 'platform': reply.platform,
                         **values, 'owner_corrections': ('corrected' in verdict if verdict else None),
                         'exposed': reply.turn_id in exposed})
        restarts = self.runtime.config.get('plasticity.behaviour.restarts', []) if self.runtime else []
        from ...config import validate_behaviour_restarts
        validate_behaviour_restarts(restarts)
        return {'lesson': _text(self.store.revisions.state(closure), 'lesson'),
                'promoted': promoted, 'rows': rows, 'restarts': restarts}

    def gate(self, closure, build):
        record = GateRecord(learner=self.store.name, revision=closure.id, state=build.digest,
            serving_revision=self.store.revisions.serving().id, at=_io.stamp(self.clock()), loss=self.signal,
            thresholds={**self.settings, 'level': int(build.state['level'][0])})
        try:
            lesson = self.lessons.get(_text(build.state, 'lesson')) if self.lessons else None
            from ...mind.lessons import STANDING_LESSONS
            serving_count = sum(b.store.revisions.serving().mode != 'base' for b in self.runtime.loaded.values()
                                if b.declaration.get('implementation') == 'behaviour') if self.runtime else 0
            own_serving = closure.id == self.store.revisions.serving().id
            budget = self.store.budgets.result(self.store.name, build.digest)
            offline = self.render(self.store.revisions.state(closure))
            checks = [Check('lesson_current', lesson is not None and lesson.status in ('active', 'candidate')),
                      Check('standing_cap', serving_count + (not own_serving) <= STANDING_LESSONS),
                      Check('budget', budget is not None and budget.ok),
                      Check('finite', all(np.all(np.isfinite(v)) for v in build.state.values())),
                      Check('served_offline', offline == self.render(build.state))]
            record.parity = {'status': 'tested', 'passed': checks[-1].passed}
            if own_serving:
                result = self.comparator.evaluate(closure, self.observations(closure))
                record.comparisons = {k: result[k] for k in ('window', 'baseline_counts', 'target', 'guards', 'platforms',
                    'not_measured_guards', 'restarts', 'baseline_at', 'confounds')}
                record.counts = result['counts']
                record.arms = {name: result['target'][name]['rate'] for name in ('baseline', 'taught')}
                verdict = result['target']['verdict']
                checks.append(Check('baseline_target', verdict == 'passed', verdict, verdict == 'awaiting_evidence'))
                for name, guard in result['guards'].items():
                    checks.append(Check('guard_' + name, guard['verdict'] != 'breach', guard['verdict']))
                if result['verdict'] == 'failed':
                    history = self.ladder()
                    entry = history.setdefault(_text(build.state, 'lesson'), {'withdrawn': [], 'halted': False})
                    if any(g['verdict'] == 'breach' for g in result['guards'].values()):
                        entry['halted'] = True
                    else:
                        entry['withdrawn'] = sorted(set([*entry['withdrawn'], int(build.state['level'][0])]))
                    _io.write_json(self.store.directory / 'behaviour-ladder.json', history)
            report = GateReport(closure.id, 'behaviour-baseline', tuple(checks))
            record.checks = report.to_dict()['checks']
            record.outcome = 'passed' if report.passed else 'awaiting_evidence' if report.awaiting else 'failed'
            record.reason = 'trial: evidence is the live baseline comparison' if not own_serving else record.outcome
            return report
        except Exception as error:
            record.error, record.reason = type(error).__name__, 'gate evaluation error'
            raise
        finally:
            self.store.certificates.write(record)


register_factory('behaviour', Driver)
