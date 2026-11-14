"""Offline re-score of paired memory runs: evidence reach, traps and why stale values were shown as current.

Reads only what a paired results directory already recorded: the frozen plan, each attempt's graded result and its
private trace. It calls no model and changes nothing. Per arm and per question it scores

- **reach** from the probe turn's model requests: the memory section injected into the request and the results of
  the memory tools the agent called (session search, the memory search tool, the built-in notes tool). Reach is
  structural (``scoring.reach``): a gold message's hash, computed the way the ledger computes it from the session
  and the message, against the hashes each shown item cites. A session search result cites its session and the
  hashes of the messages it returned. Where the trace records them, the sources the probe turn's assembled context
  cited are a surface of their own (``cited``), so reach is structural on what the host injected whatever the
  rendered section shows of it. A citation names a source, and a source may hold many messages (a checkpoint holds
  a conversation): it reaches the messages its section's records name by hash (the trace's ``messages``), or the
  one message of a source that holds one (the episode's ledger map, ``ledger_sources``), never every message of its
  source. Only a section whose body reached a model request of the turn counts (``delivered``): an assembly the
  provider dropped (it answered too late) or a section it held back was never seen. Every citation of the turn's
  assemblies, delivered or not, is the ``assembled`` surface, coverage reported apart and never part of reach. A
  trace without citations, without the map or without the messages each section named (or with that list cut
  short) leaves both unscored, and one without the sections' body hashes leaves ``cited`` unscored. Beside it, the
  weaker checks: the expected answer values, and the gold
  statements' words, shown anywhere in those surfaces (``scoring.span_reach``). They read each surface's complete
  rendered body, prose and paragraph breaks included, where structural reach reads only the parsed items (prose
  cites nothing); the rank they give is the body's. A probe turn with no model request
  recorded shows nothing of what the reader was given: its reach is unscored and out of every reach denominator,
  never a miss;
- **right** as the paired report attributes it (``paired_report.summarize``);
- **traps** from the answer file and the gold's kind (``scoring.trap_outcome``): each field asserts its whole value,
  and no value is found by searching the answer's words. A conflict's ``asked`` needs the specified form: status
  "ask", the value fields empty and a clarification question; a status label alone asks nothing;
- **dead-value lines** (``paired_report.DEAD_VALUE_BASIS``: a line of the injected context that shows a forbidden
  value as current), each classified by what the section shows:

  - ``identical_key``: the stale value's claim and the claim of the correction share a key (subject and predicate
    as the ledger keys them), yet the old one was not superseded;
  - ``different_key``: the correction formed a claim under another key for the same fact;
  - ``raw_quotation``: the correction reached the section only as a quotation with no claim for its value, or the
    stale value did while the correction has its claim;
  - ``unknown``: the section shows neither side's key (the correction is not in it), or nothing tells which claim
    of the correction is the stale value's replacement;
  - ``decoy``: the forbidden value was never current (another thing's value, or the answer's raw form), so no
    supersession could apply.

  The correction is the gold evidence stated after the stale value was first stated. Its claim is an assertion that
  cites it and asserts an expected value of the answer field the stale value held (gold's ``stale_fields``, or the
  answer's one field); with no resolvable gold, any assertion of that value. A correction message can state other
  facts too: their claims are never compared. Where the field or its expected value is not known (an answer of
  several fields whose gold does not say, or a withdrawal whose answer is "unknown"), a claim of the correction
  cannot be told apart from another fact's, so the line is ``unknown``. "No claim shown" is read from the section:
  the store may hold a claim the section left out.

Gold comes from the plan's own artifacts when they carry it, else from a dataset rendered today from the same seed
whose scenario matches the planned one exactly (``RenderedGold``), else it is inferred from the owner's turns and
labelled ``inferred``: a forbidden value is stale when an owner statement that shares a content word with the probe
states it and a later statement states an expected value; evidence is the statements that state an expected value.

    python benchmarks/public_memory/paired_rescore.py --results RUN_DIR [--results ...] \
        [--gold-dataset RENDERED_DIR ...] --output OUT_DIR
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent


from benchmarks.public_memory import scoring

try:
    from colonymind.qualification import paired_report
except ImportError:  # a checkout without the package installed
    sys.path.insert(0, str(HERE.parents[1] / 'sidecar'))
    from colonymind.qualification import paired_report
from colonymind.beliefs.source_claims import norm_value  # noqa: E402
from colonymind.memory.compass import shows, tokens  # noqa: E402
from colonymind.qualification.paired_cases import load_generated_dataset  # noqa: E402
from colonymind.turns.idempotency import source_message_hash  # noqa: E402

PROTOCOL = 'paired-memory-rescore-1'
SOURCE_WORKER = 'paired-source-worker'
BOUNDARIES = {'native_turn', 'body_tick', 'episode', 'drain', 'history'}
MEMORY_TOOLS = {'session_search', 'colonymind_memory_search', 'memory'}
ASSEMBLE = '/v1/host/context/assemble'
NOTES_HEADER = re.compile(r'═{10,}\n((?:MEMORY|USER PROFILE)[^\n]*)\n═{10,}\n')
# The host's usage indicator ends a notes header: "[<percent>% <dash> <length>/<limit> chars]", where the length is
# the body's own (its entries joined by the entry delimiter).
NOTES_LENGTH = re.compile(r'(\d[\d,]*)/\d[\d,]* chars\]\s*$')
CLASSES = ('identical_key', 'different_key', 'raw_quotation', 'unknown', 'decoy')
UNKNOWN_LABEL = 'unknown'
UNIT_LABELS = {'minutes', 'minute', 'min', 'mins'}
# Words of a probe's format instruction or of setup filler: never the subject a stale statement shares with it.
STOPWORDS = {'write', 'answer', 'json', 'exactly', 'string', 'number', 'with', 'that', 'this', 'what', 'where',
             'when', 'which', 'from', 'have', 'your', 'told', 'never', 'know', 'unknown', 'word', 'nothing',
             'action', 'needed', 'there', 'about', 'just', 'want', 'noted', 'side', 'only', 'moment', 'right',
             'record', 'part', 'given', 'named', 'place', 'time', 'file', 'else', 'them', 'they', 'will', 'said',
             'says', 'does', 'into', 'here', 'then', 'than', 'were', 'been', 'being', 'also', 'each', 'some'}
BASIS = ('Offline re-score of recorded paired traces; no model call. Reach is structural on message hashes the '
         'shown items cite (gold messages hashed as the ledger hashes them); value and quotation reach are the '
         'weaker checks, on whole words. Right is the paired report\'s attribution. Dead-value lines follow the '
         'report\'s definition and are classified by the claim keys the injected section shows; "no claim shown" '
         'is read from the section, not the store. Gold is the plan\'s, a matching dataset rendered today, or '
         'inferred from the owner\'s turns and labelled so.')


# ------------------------------------------------------------------ trace reading

def _events(lines):
    events = []
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if isinstance(event, dict) and event.get('protocol') == paired_report.TRACE_PROTOCOL:
            events.append(event)
    return events


def _agent_request(event):
    return event.get('kind') == 'model_request' and event.get('thread') != SOURCE_WORKER


def _probe_index(case):
    episodes = (case.get('inputs') or {}).get('episodes') or []
    return max((index for index, entry in enumerate(episodes) if isinstance(entry, dict) and 'user' in entry),
               default=None)


def _turn_events(events, index):
    return [event for event in events if event.get('kind') == 'native_turn'
            and (event.get('data') or {}).get('index') == index]


def probe_window(events, probe):
    """The probe turn's events: after the last turn, tick or phase boundary before its recorded turn."""
    ends = [position for position, event in enumerate(events) if event.get('kind') == 'native_turn'
            and (event.get('data') or {}).get('index') == probe]
    end = ends[-1] if ends else len(events)
    start = max((position for position in range(end) if events[position].get('kind') in BOUNDARIES), default=-1)
    return events[start + 1:end]


def probe_requests(events, probe):
    """The agent model requests of the probe turn (``probe_window``)."""
    return [event for event in probe_window(events, probe) if _agent_request(event)]


def ledger_map(events):
    """``({source id: [message hash]}, complete)`` from the episode's ledger map (the worker's ``ledger_sources``
    event): the hash of each message each source holds, and whether the map lists every source the ledger held
    (the worker cuts it at its bound and says so, ``truncated``, listing the cited sources first); ``(None,
    False)`` when the trace has none."""
    recorded = [event for event in events if event.get('kind') == 'ledger_sources']
    data = (recorded[-1].get('data') or {}) if recorded else {}
    if not isinstance(data.get('sources'), list):
        return None, False
    return ({row['turn_id']: [value for value in row.get('message_hashes') or [] if isinstance(value, str)]
             for row in data['sources'] if isinstance(row, dict) and isinstance(row.get('turn_id'), str)},
            data.get('truncated') is False)


def delivered(section, blocks):
    """Whether an assembled section reached a model request: one of the request's memory blocks (``blocks``) holds,
    from the start of one of its lines, exactly the body the assembly returned, by the SHA-256 and length its trace
    event recorded (the provider puts each section's body after its heading line). None when the event recorded no
    body hash, so delivery cannot be told."""
    digest, length = section.get('body_sha256'), section.get('body_chars')
    if not isinstance(digest, str) or type(length) is not int:
        return None
    if length <= 0:
        return False
    for block in blocks:
        for start in (0, *(match.end() for match in re.finditer('\n', block))):
            if (len(block) - start >= length
                    and hashlib.sha256(block[start:start + length].encode('utf-8')).hexdigest() == digest):
                return True
    return False


def cited_items(events, probe, blocks):
    """``(cited, assembled, unresolved)``: the sources the probe turn's assembled contexts cited (``context_route``
    events of the assemble route with ``source_refs``), one item each, mapped to their messages through the ledger
    map. ``cited``
    keeps the sections whose body reached one of the turn's model requests (``delivered``, against the memory
    ``blocks`` those requests carried); ``assembled`` keeps them all, delivered or not. A cited source's item names
    the messages of that source the section's records named by hash (``messages``), and the source's one message
    when it holds exactly one: never every message of a source, since a citation does not show a whole
    conversation. Both None when the trace recorded no citation for the turn, has no ledger map to read them by, or
    does not say which messages a section named; ``cited`` also None when a section's delivery cannot be told. A
    surface is also None when one of its citations is unresolved: a source the section named no message of, missing
    from a map that was cut (``ledger_map``), so whether it held one message is unknown. ``unresolved`` says that
    happened to a delivered citation, which could have reached what nothing else did."""
    window = probe_window(events, probe)
    sections = [section for event in window if event.get('kind') == 'context_route'
                and (event.get('data') or {}).get('path') == ASSEMBLE
                and isinstance(event['data'].get('source_refs'), list)
                for section in event['data']['source_refs'] if isinstance(section, dict)]
    recorded = any(event.get('kind') == 'context_route' and 'source_refs' in (event.get('data') or {})
                   for event in window)
    sources, complete = ledger_map(events)
    if (not recorded or sources is None
            or any(not isinstance(section.get('messages'), list) or section.get('messages_truncated')
                   for section in sections)):
        return None, None, False

    def items(chosen):
        """The chosen sections' cited sources as items, or None when one of them is unresolved."""
        found = {}
        for section in chosen:
            named = {}
            for pair in section['messages']:
                if isinstance(pair, list) and len(pair) == 2 and all(isinstance(part, str) for part in pair):
                    named.setdefault(pair[0], set()).add('hash:' + pair[1])
            for ref in section.get('refs') or []:
                if not isinstance(ref, dict) or not isinstance(ref.get('source_id'), str):
                    continue
                source = ref['source_id']
                if source not in sources and not complete and not named.get(source):
                    return None
                held = sources.get(source) or []
                found.setdefault(source, {'turn:' + source}).update(
                    named.get(source, set()) | ({'hash:' + held[0]} if len(held) == 1 else set()))
        return [{'surface': 'cited', 'data': None, 'quote': None, 'strings': [], 'refs': refs}
                for refs in found.values()]
    reached = [delivered(section, blocks) for section in sections]
    if None in reached:
        return None, items(sections), False
    chosen = [section for section, ok in zip(sections, reached) if ok]
    cited = items(chosen)
    return cited, items(sections), cited is None


def _messages(event):
    return ((event.get('data') or {}).get('payload') or {}).get('messages') or []


def notes_bodies(text):
    """The built-in notes blocks of a system message, each body whole. Entries may hold blank lines, so no blank
    line ends a body: the header states the body's length, and the body is exactly that many characters. A header
    that states no length is read to the next block header or the end of the message, so no entry is lost (the
    host's trailing lines may be read with it)."""
    headers = list(NOTES_HEADER.finditer(text))
    bodies = []
    for index, header in enumerate(headers):
        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        length = NOTES_LENGTH.search(header.group(1))
        bodies.append(text[start:min(end, start + int(length.group(1).replace(',', '')))] if length
                      else text[start:end].rstrip())
    return bodies


def surfaces(requests):
    """What the probe turn's requests showed of memory: injected blocks, memory-tool results and notes, in order."""
    blocks, tools, notes = [], {}, []
    for event in requests:
        messages = [message for message in _messages(event) if isinstance(message, dict)]
        names = {call.get('id'): (call.get('function') or {}).get('name') for message in messages
                 for call in message.get('tool_calls') or [] if isinstance(call, dict)}
        for message in messages:
            text = paired_report._message_text(message.get('content'))
            if message.get('role') == 'user':
                blocks.extend(block for block in paired_report._MEMORY_CONTEXT.findall(text) if block not in blocks)
            elif message.get('role') == 'tool' and names.get(message.get('tool_call_id')) in MEMORY_TOOLS:
                tools.setdefault(message['tool_call_id'], (names[message['tool_call_id']], text))
            elif message.get('role') == 'system':
                notes.extend(body for body in notes_bodies(text) if body not in notes)
    return blocks, list(tools.values()), notes


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _refs(value):
    """The turn ids and message hashes an item cites, anywhere in its structure."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('source', 'source_id', 'source_turn_id', 'source_uri') and isinstance(item, str) and item:
                found.add('turn:' + item.removeprefix('turn:'))
            elif key == 'source_message_hash' and isinstance(item, str) and item:
                found.add('hash:' + item)
            else:
                found |= _refs(item)
    elif isinstance(value, list):
        for item in value:
            found |= _refs(item)
    return found


def section_items(text, surface):
    """One item per ``- `` line of a memory section: its parsed record, its quotation and the refs it cites. A card
    (claims under one key) also cites the passages its assertions name by ``evidence_ref``, and keeps what each
    assertion cites on its own (``assertion_refs``)."""
    items = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped.startswith('- '):
            continue
        body, data, quote = stripped[2:], None, None
        try:
            data, end = json.JSONDecoder().raw_decode(body)
            rest = body[end:].strip()
            if rest:
                try:
                    quote = json.loads(rest)
                except ValueError:
                    quote = rest
        except ValueError:
            data = None
        data = data if isinstance(data, dict) else None
        quote = quote if isinstance(quote, str) else None
        strings = [*_strings(data), *([quote] if quote is not None else [])] if data else [body]
        items.append({'surface': surface, 'line': line, 'data': data, 'quote': quote,
                      'refs': _refs(data) if data else set(), 'strings': strings})
    passages = {item['data']['evidence_ref']: item for item in items
                if item['data'] and isinstance(item['data'].get('evidence_ref'), str) and 'quote' in item['data']}
    for item in items:
        item['assertion_refs'] = []
        for assertion in _assertions(item):
            cited = passages.get(assertion.get('evidence_ref'))
            refs = _refs(assertion) | (cited['refs'] if cited is not None else set())
            item['assertion_refs'].append(refs)
            item['refs'] = item['refs'] | refs
    return items


def _card(item):
    content = (item.get('data') or {}).get('content')
    return (isinstance(content, dict) and {'subject', 'predicate', 'assertions'} <= set(content)
            and isinstance(content['assertions'], list))


def _assertions(item):
    return [assertion for assertion in item['data']['content']['assertions'] if isinstance(assertion, dict)] \
        if _card(item) else []


def claim_key(item):
    """The key the ledger groups claims under: the subject (``speaker`` for the owner's "I") and the predicate."""
    content = item['data']['content']
    subject = str(content.get('subject') or '')
    return ('speaker' if subject.strip().lower() == 'i' else norm_value(subject), norm_value(content['predicate']))


def _quotation(item):
    data = item.get('data') or {}
    return not _card(item) and (item.get('quote') is not None or 'quote' in data
                                or data.get('kind') == 'conversation_pair' or data.get('state') == 'quotation')


def tool_items(name, text):
    """Items from a memory tool's result. A session search result cites its session and its messages' hashes."""
    try:
        data = json.loads(text)
    except ValueError:
        return [{'surface': 'tools', 'data': None, 'quote': None, 'refs': set(), 'strings': [text]}]
    if name == 'colonymind_memory_search' and isinstance(data, dict) and isinstance(data.get('content'), str):
        return section_items(data['content'], 'tools')
    if name == 'session_search' and isinstance(data, dict) and isinstance(data.get('results'), list):
        items = []
        for result in data['results']:
            if not isinstance(result, dict) or not isinstance(result.get('session_id'), str):
                continue
            session = result['session_id']
            messages = [message for key in ('bookend_start', 'messages', 'bookend_end')
                        for message in result.get(key) or [] if isinstance(message, dict)
                        and message.get('role') in ('user', 'assistant') and isinstance(message.get('content'), str)]
            refs = {'session:' + session} | {'hash:' + source_message_hash(session, {
                'role': message['role'], 'content': message['content']}) for message in messages}
            items.append({'surface': 'tools', 'session': session, 'data': None, 'quote': None, 'refs': refs,
                          'strings': [message['content'] for message in messages]
                          + [value for value in (result.get('snippet'),) if isinstance(value, str)]})
        return items
    return [{'surface': 'tools', 'data': None, 'quote': None, 'refs': set(), 'strings': list(_strings(data))}]


def readable(text):
    """One rendered memory body whole, as the weaker checks read it: prose and paragraph breaks as shown, and each
    JSON item line as the strings it carries (an escaped value reads as it was written)."""
    lines = []
    for line in text.split('\n'):
        items = section_items(line, None) if line.strip().startswith('- ') else []
        lines.append('\n'.join(items[0]['strings']) if items and items[0]['data'] else line)
    return '\n'.join(lines)


def tool_body(text):
    """A memory tool's whole result, as the weaker checks read it: every string it carries, or its raw text."""
    try:
        data = json.loads(text)
    except ValueError:
        return readable(text)
    return '\n'.join(readable(value) for value in _strings(data))


# ------------------------------------------------------------------ gold

def _label_values(spec):
    """The artifact's expected values a statement can carry: labels (not "unknown", not unit spellings) and numbers."""
    values = []
    for item in spec.get('assertions', []):
        if not isinstance(item, dict) or not item.get('path'):
            continue
        if item.get('op') == 'label_one_of':
            values.extend(value for value in item.get('value', []) if isinstance(value, str) and value.strip()
                          and value.casefold() != UNKNOWN_LABEL and value.casefold() not in UNIT_LABELS)
        elif item.get('op') == 'number' and isinstance(item.get('value'), (int, float)):
            number = item['value']
            values.append(str(int(number)) if float(number).is_integer() else str(number))
    return values


def _statements(case):
    """The owner's statements before the probe, in the order they were made: history messages, then turns."""
    inputs = case.get('inputs') or {}
    said = [({'session': session['id'], 'message': index}, message['content'])
            for session in inputs.get('history') or [] for index, message in enumerate(session['messages'])
            if message.get('role') == 'user']
    probe = _probe_index(case)
    said.extend(({'turn': index}, entry['user']) for index, entry in enumerate(inputs.get('episodes') or [])
                if isinstance(entry, dict) and 'user' in entry and index != probe)
    return said


def _position(case, ref):
    """Where an owner statement falls in the order things were said: history sessions, then the episode's turns."""
    if 'session' in ref:
        sessions = [session['id'] for session in (case.get('inputs') or {}).get('history') or []]
        return (0, sessions.index(ref['session']), ref['message'])
    return (1, ref['turn'], 0)


def correction(case, gold):
    """The gold evidence that came after the stale value was first stated: what replaced or withdrew it. Evidence
    may also hold the first statement (its other fields still answer), which is never its own correction."""
    stale = gold.get('stale', [])
    first = min((_position(case, ref) for ref, text in _statements(case)
                 if any(scoring.shows(scoring.words(text), value) for value in stale)), default=None)
    return [index for index, ref in enumerate(gold.get('evidence', []))
            if first is None or _position(case, ref) > first]


def _field_values(spec):
    """``{field: expected values a statement can carry}`` for each one-key answer field (``_label_values``)."""
    fields = {}
    for item in spec.get('assertions', []):
        if (not isinstance(item, dict) or not isinstance(item.get('path'), list) or len(item['path']) != 1
                or item.get('op') not in ('label_one_of', 'number')):
            continue
        fields.setdefault(item['path'][0], []).extend(_label_values({'assertions': [item]}))
    return fields


def stale_fields(gold, spec):
    """Each stale value mapped to the expected values of the answer field it once held: the field gold names, else
    the answer's one field. None when the field is not known (several fields, and gold does not say)."""
    fields, named = _field_values(spec), gold.get('stale_fields') or {}
    only = next(iter(fields)) if len(fields) == 1 else None
    held = {}
    for value in gold.get('stale', []):
        field = named.get(value, only)
        held[value] = None if field is None else list(fields.get(field, []))
    return held


def _content_words(text):
    return {word for word in scoring.words(text) if len(word) >= 4 and word not in STOPWORDS}


def inferred_gold(case):
    """Gold for an artifact whose plan carries none, from the owner's statements (see the module docstring)."""
    inputs = case.get('inputs') or {}
    probe = _probe_index(case)
    subject = _content_words(inputs['episodes'][probe]['user']) if probe is not None else set()
    said = _statements(case)
    result = []
    for spec in (case.get('oracle') or {}).get('artifacts', []):
        labels = [item for item in spec.get('assertions', []) if isinstance(item, dict)
                  and item.get('op') == 'label_one_of' and item.get('path')]
        current = _label_values(spec)
        if any(item['path'] == ['status'] and item.get('value') == ['ask'] for item in labels):
            result.append({'kind': 'conflict', 'evidence': []})
            continue
        if labels and all([value.casefold() for value in item.get('value', [])] == [UNKNOWN_LABEL]
                          for item in labels):
            result.append({'kind': 'abstain', 'evidence': []})
            continue
        states = [index for index, (_, text) in enumerate(said) if any(scoring.shows(scoring.words(text), value)
                                                                       for value in current)]
        stale = [value for value in spec.get('forbidden', []) if isinstance(value, str) and any(
            scoring.shows(scoring.words(text), value) and _content_words(text) & subject
            and any(later > index for later in states) for index, (_, text) in enumerate(said))]
        gold = {'kind': 'update' if stale else 'recall', 'evidence': [said[index][0] for index in states]}
        if stale:
            gold['stale'] = stale
        result.append(gold)
    return result


def _without_gold(artifacts):
    return [{key: value for key, value in spec.items() if key != 'gold'} for spec in artifacts]


class RenderedGold:
    """Gold from datasets rendered today from the seeds of past runs. A scenario lends its gold to a planned case
    only when its turns, history, files and artifacts (gold aside) are exactly the planned ones."""

    def __init__(self, directories=()):
        self.scenarios, self.mismatches = {}, []
        for directory in directories:
            _, scenarios, _ = load_generated_dataset(directory)
            for scenario in scenarios:
                self.scenarios.setdefault(scenario['id'], []).append(scenario)

    def gold(self, case):
        candidates = self.scenarios.get(case.get('id'), [])
        inputs, oracle = case.get('inputs') or {}, case.get('oracle') or {}
        for scenario in candidates:
            if (inputs.get('episodes') == scenario['episodes'] and inputs.get('history') == scenario.get('history')
                    and inputs.get('initial_files') == scenario['initial_files']
                    and _without_gold(oracle.get('artifacts', [])) == _without_gold(scenario['oracle']['artifacts'])
                    and all('gold' in spec for spec in scenario['oracle']['artifacts'])):
                return [spec['gold'] for spec in scenario['oracle']['artifacts']]
        if candidates:
            self.mismatches.append(case.get('id'))
        return None


def case_gold(case, rendered=None):
    """(gold per artifact, basis): the plan's own, a matching rendered dataset's, or inferred."""
    artifacts = (case.get('oracle') or {}).get('artifacts', [])
    if artifacts and all('gold' in spec for spec in artifacts):
        return [spec['gold'] for spec in artifacts], 'dataset'
    found = rendered.gold(case) if rendered is not None else None
    if found is not None:
        return found, 'rendered'
    return inferred_gold(case), 'inferred'


def resolve(case, gold, events):
    """Each evidence item as (identifiers, session, text), or None when its turn was never recorded. A history
    message is hashed from the plan's own history; an episode turn from the message the trace recorded for it."""
    inputs = case.get('inputs') or {}
    history = {session['id']: session['messages'] for session in inputs.get('history') or []}
    resolved = []
    for ref in gold.get('evidence', []):
        if 'session' in ref:
            message = history[ref['session']][ref['message']]
            resolved.append(({'hash:' + source_message_hash(ref['session'], {
                'role': message['role'], 'content': message['content']})}, ref['session'], message['content']))
            continue
        turns = _turn_events(events, ref['turn'])
        data = (turns[-1].get('data') or {}) if turns else {}
        messages = [message for message in data.get('messages') or [] if isinstance(message, dict)]
        users = [position for position, message in enumerate(messages) if message.get('role') == 'user'
                 and isinstance(message.get('content'), str)]
        if not users or not isinstance(data.get('session_id'), str):
            resolved.append(None)
            continue
        chosen = messages[users[-1]]
        if ref.get('role') == 'assistant':
            replies = [message for message in messages[users[-1] + 1:] if message.get('role') == 'assistant'
                       and isinstance(message.get('content'), str) and message['content'].strip()]
            if not replies:
                resolved.append(None)
                continue
            chosen = replies[-1]
        text = chosen['content'] if ref.get('role') == 'assistant' else inputs['episodes'][ref['turn']]['user']
        resolved.append(({'hash:' + source_message_hash(data['session_id'], {
            'role': chosen['role'], 'content': chosen['content']})}, data['session_id'], text))
    return resolved


# ------------------------------------------------------------------ traps and dead lines

def _answer(raw):
    try:
        value = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _given(value):
    """A field's value as the answer gives it: one whole value (a list gives each of its items), never words inside.
    A null gives none."""
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values
            if item is not None and not isinstance(item, (dict, list)) and str(item).strip()]


def _declared_keys(spec):
    """The answer's keys as its artifact declares them (``keys_equal`` at the top), or None."""
    return next((list(item['value']) for item in spec.get('assertions', []) if isinstance(item, dict)
                 and item.get('op') == 'keys_equal' and item.get('path') == [] and isinstance(item.get('value'), list)),
                None)


def _asked(fields, keys):
    """A well-formed ask, as the paired conflict answer specifies it: status "ask", every value field there and an
    empty string (nothing chosen), and a clarification question with words in it. A status label alone asks
    nothing."""
    values = [key for key in keys if key not in ('status', 'question')]
    question = fields.get('question')
    return (str(fields.get('status', '')).strip().casefold() == 'ask'
            and all(isinstance(fields.get(key), str) and not fields[key].strip() for key in values)
            and isinstance(question, str) and bool(scoring.words(question)))


def trap(gold, answer, right, keys=None):
    """The trap outcome of one parsed answer file under the gold's kind (``scoring.trap_outcome``). The file's fields
    are its structured judgment: each field asserts its whole value. A field that names several values in prose
    asserts none of them here; reading prose is a judge's work. ``keys`` are the answer's declared keys (the file's
    own when None): an ask needs each value field there and empty."""
    fields = dict(answer or {})
    if gold['kind'] == 'conflict':
        chosen = [item for key, value in fields.items() if key not in ('status', 'question') for item in _given(value)]
        asked = _asked(fields, list(fields) if keys is None else keys)
        if gold.get('sides'):
            return scoring.trap_outcome('conflict', right=right, asserted=chosen, asked=asked, sides=gold['sides'])
        # No sides known: any value given picked one, whatever the status says.
        outcome = 'trap' if chosen else 'asked' if asked else 'other'
        return {'kind': 'conflict', 'outcome': outcome, 'matched': []}
    given = [item for value in fields.values() for item in _given(value)]
    if gold['kind'] == 'abstain':
        # A specific value where the right answer is "unknown" is the trap; anything else is judged by the verdict.
        specific = [item for item in given if item.casefold() != UNKNOWN_LABEL]
        return scoring.trap_outcome('abstain', right=right, asserted=specific,
                                    answerable=True if specific else None)
    return scoring.trap_outcome(gold['kind'], right=right, asserted=given, stale=gold.get('stale', []))


def _line_item(line):
    items = section_items(line, 'injected')
    return items[0] if items else {'surface': 'injected', 'data': None, 'quote': None, 'refs': set(),
                                   'strings': [line]}


def _asserts(card, values):
    return any(_asserts_value(assertion, values) for assertion in _assertions(card))


def _asserts_value(assertion, values):
    return any(shows(tokens(str(assertion.get('value', ''))), value) for value in values)


def _states(item, values):
    return any(shows(tokens(text), value) for text in item['strings'] for value in values)


def classify(line, items, *, stale, gold_ids):
    """Why a dead-value line showed a stale value as current (module docstring); ``decoy`` when none it shows was
    ever current. ``stale`` maps each stale value to the expected values of the field it held (``stale_fields``).
    ``gold_ids`` are the correction messages' identifiers: the correction's claim is an assertion that cites one and
    asserts the field's expected value; without them, any claim or quotation that states that value stands in."""
    bare = tokens(paired_report._unescaped(paired_report._SUPERSEDED_NOTE.sub(' ', line)))
    shown = [value for value in stale if shows(bare, value)]
    if not shown:
        return 'decoy'
    item = _line_item(line)
    cards = [candidate for candidate in items if _card(candidate)]
    if _card(item):
        stale_cards = [item]
    elif item['data'] and isinstance(item['data'].get('evidence_ref'), str):
        stale_cards = [card for card in cards if _asserts(card, shown) and any(
            assertion.get('evidence_ref') == item['data']['evidence_ref'] for assertion in _assertions(card))]
    else:
        stale_cards = [card for card in cards if card['refs'] & item['refs'] and _asserts(card, shown)]
    held = [stale[value] for value in shown]
    wanted = None if any(values is None for values in held) else [value for values in held for value in values]
    if gold_ids:
        # What the correction formed: each card assertion that cites a correction message. The message may state
        # other facts too; only an assertion of the stale field's expected value is the correction's claim.
        formed = [(card, assertion) for card in cards
                  for assertion, refs in zip(_assertions(card), card['assertion_refs']) if refs & gold_ids]
        quoted = any(_quotation(candidate) and candidate['refs'] & gold_ids for candidate in items)
    else:
        formed = [(card, assertion) for card in cards for assertion in _assertions(card)]
        quoted = bool(wanted) and any(_quotation(candidate) and _states(candidate, wanted) for candidate in items)
    update_cards = [card for card, assertion in formed if wanted and _asserts_value(assertion, wanted)]
    stale_keys = {claim_key(card) for card in stale_cards}
    update_keys = {claim_key(card) for card in update_cards}
    if stale_keys and update_keys:
        return 'identical_key' if stale_keys & update_keys else 'different_key'
    if update_keys:
        return 'raw_quotation' if _quotation(item) else 'unknown'
    if gold_ids and formed and not wanted:
        # The correction formed claims, and with no known value for the stale field none can be told to be its.
        return 'unknown'
    return 'raw_quotation' if quoted else 'unknown'


def _line_kind(line):
    item = _line_item(line)
    return 'claim' if _card(item) else 'quotation' if _quotation(item) else 'other'


# ------------------------------------------------------------------ one episode

def episode(case, lines, *, completion, answer, rendered=None):
    """One arm's attempt at one question, scored offline. No trace text leaves this function except the values
    an oracle already names (a trap's matched stale value)."""
    events = _events(lines)
    golds, basis = case_gold(case, rendered)
    gold = golds[0] if golds else {'kind': 'recall', 'evidence': []}
    artifacts = (case.get('oracle') or {}).get('artifacts', [])
    spec = artifacts[0] if artifacts else {}
    current = _label_values(spec)
    requests = probe_requests(events, _probe_index(case))
    # With no model request of the probe turn recorded, nothing shows what the reader was given: every reach is
    # unscored (None) and stays out of the denominators, where a request with none of the gold in it is a miss.
    observed = bool(requests)
    blocks, tools, notes = surfaces(requests)
    injected = [item for block in blocks for item in section_items(block, 'injected')]
    from_tools = [item for name, text in tools for item in tool_items(name, text)]
    cited, assembled, citation_unresolved = cited_items(events, _probe_index(case), blocks)
    shown = injected + from_tools + (cited or [])
    # The weaker checks read every body whole; structural reach reads the parsed items above.
    by_surface = {'injected': [readable(block) for block in blocks], 'tools': [tool_body(text) for _, text in tools],
                  'notes': notes}
    texts = by_surface['injected'] + by_surface['tools'] + by_surface['notes']
    resolved = resolve(case, gold, events)
    unresolved = sum(item is None for item in resolved)
    reach = surface = session = None
    if observed and resolved and not unresolved:
        ids = [identifiers for identifiers, _, _ in resolved]
        reach = scoring.reach(ids, [item['refs'] for item in shown])
        if citation_unresolved and not reach['all']:
            # A delivered citation the cut map could not resolve may hold a gold message nothing else reached.
            reach = None
        surface = {'injected': scoring.reach(ids, [item['refs'] for item in injected]),
                   'tools': scoring.reach(ids, [item['refs'] for item in from_tools]),
                   'cited': scoring.reach(ids, [item['refs'] for item in cited]) if cited is not None else None,
                   'assembled': (scoring.reach(ids, [item['refs'] for item in assembled])
                                 if assembled is not None else None)}
        sessions = {name for _, name, _ in resolved if name}
        returned = [item['session'] for item in from_tools if item.get('session')]
        session = scoring.session_reach(sessions, returned) if sessions and returned else None
    quotes = [item[2] for item in resolved if item]
    parsed = _answer(answer)
    row = {'question': case.get('id'), 'scenario': case.get('id'), 'type': (case.get('inputs') or {}).get('scenario'),
           'kind': gold['kind'],
           'gold_basis': basis, 'gold_evidence': len(gold.get('evidence', [])), 'gold_unresolved': unresolved,
           'observed': observed, 'reach': reach, 'reach_surfaces': surface, 'reach_session': session,
           'reach_value': scoring.span_reach(current, texts) if observed else None,
           'reach_value_surfaces': {name: scoring.span_reach(current, values)['any']
                                    for name, values in by_surface.items()} if observed else None,
           'reach_quote': scoring.span_reach(quotes, texts) if quotes and observed else None,
           'right': completion, 'trap': trap(gold, parsed, completion, _declared_keys(spec)),
           'answered': parsed is not None,
           'dead_lines': []}
    context = paired_report.last_memory_context(lines)
    pairs = paired_report.forbidden_pairs(case)
    if context is not None and pairs:
        items = [item for block in context for item in section_items(block, 'injected')]
        after = correction(case, gold)
        gold_ids = {identifier for index, item in enumerate(resolved) if item and index in after
                    for identifier in item[0]}
        stale = {}
        for entry, artifact in zip(golds, artifacts):
            stale.update(stale_fields(entry, artifact))
        for block in context:
            for line in block.split('\n'):
                if paired_report.dead_value_line(line, pairs):
                    row['dead_lines'].append({
                        'question': case.get('id'), 'kind': gold['kind'], 'line': _line_kind(line),
                        'class': classify(line, items, stale=stale, gold_ids=gold_ids),
                        'basis': 'inferred' if basis == 'inferred' else 'gold'})
    return row


# ------------------------------------------------------------------ a results directory

def _trace_lines(directory, member):
    root = Path(directory).resolve()
    path = root / member['path'] / 'attempts' / member['case']['id'] / 'private-trace.jsonl'
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.exists():
        return []
    return path.read_text().splitlines()


def _seed(label):
    return int.from_bytes(hashlib.sha256(str(label).encode()).digest()[:8], 'big')


def rescore(directory, *, rendered=None):
    """Every arm of one paired results directory: rows, per-type tables, dead-line classes and paired contrasts."""
    manifest = paired_report.load_manifest(directory)
    arms, reference, _, _ = paired_report.arms_of(manifest)
    summary = paired_report.summarize(directory)
    graded = {pair['episode_id']: pair for pair in summary['pairs']}
    rows = {arm: [] for arm in arms}
    members = {arm: [] for arm in arms}
    for declared in manifest['pairs']:
        pair = graded[declared['episode_id']]
        for arm in arms:
            member = declared['arms'][arm]
            members[arm].append(member)
            result = (pair.get('results') or {}).get(arm) or {}
            answer = ((result.get('effects') or {}).get('artifacts') or {}).get('answer.json')
            row = episode(member['case'], _trace_lines(directory, member), completion=pair['completion'][arm],
                          answer=answer, rendered=rendered)
            # A repetition reuses its case id: the attempt keeps the plan's episode id, and the scenario it repeats
            # is the unit its contrast averages over.
            row.update(question=declared['episode_id'],
                       scenario=declared.get('workflow_id', declared['episode_id']))
            for line in row['dead_lines']:
                line['question'] = declared['episode_id']
            rows[arm].append(row)
    output = {'protocol': PROTOCOL, 'basis': BASIS, 'label': manifest.get('label'),
              'dataset': {key: (manifest.get('dataset') or {}).get(key) for key in ('version', 'split')},
              'reference_arm': reference, 'arms': {}, 'contrasts': {},
              'gold_basis': dict(Counter(row['gold_basis'] for arm in arms for row in rows[arm]))}
    for arm in arms:
        dead = [line for row in rows[arm] for line in row['dead_lines']]
        classified = [line for line in dead if line['class'] in ('identical_key', 'different_key', 'raw_quotation')]
        output['arms'][arm] = {
            'summary': scoring.summarize(rows[arm]),
            'dead_lines': {'lines': len(dead), 'by_class': dict(Counter(line['class'] for line in dead)),
                           'stale_lines': sum(line['class'] != 'decoy' for line in dead),
                           'classified_lines': len(classified),
                           'report_lines': paired_report._dead_values(directory, members[arm])['dead_value_lines'],
                           'by_basis': dict(Counter(line['basis'] for line in dead)), 'lines_detail': dead},
            'probe_unobserved': sum(not row['observed'] for row in rows[arm]),
            'rows': [{key: value for key, value in row.items() if key != 'dead_lines'} for row in rows[arm]]}
    seed = _seed(manifest.get('label'))
    for arm in arms:
        if arm == reference:
            continue
        # Every declared question in both arms: one the paired report could not attribute withholds the contrast.
        # Units are scenarios, repetitions averaged (the paired statistics contract).
        contrasts = {'right': scoring.contrast(rows[arm], rows[reference], key='right', seed=seed,
                                               unit_key='scenario'), 'reach_any': None}
        # Reach applies to the questions whose gold names evidence; one unscored there withholds its contrast.
        reached = [({'question': a['question'], 'scenario': a['scenario'], 'reached': (a['reach'] or {}).get('any')},
                    {'question': b['question'], 'scenario': b['scenario'], 'reached': (b['reach'] or {}).get('any')})
                   for a, b in zip(rows[arm], rows[reference]) if a['gold_evidence'] or b['gold_evidence']]
        if reached:
            contrasts['reach_any'] = scoring.contrast([a for a, _ in reached], [b for _, b in reached],
                                                      key='reached', seed=seed, unit_key='scenario')
        output['contrasts'][f'{arm} vs {reference}'] = contrasts
    return output


# ------------------------------------------------------------------ rendering

def _cell(rate):
    if not rate:
        return 'n/a'
    return f"{rate['successes']}/{rate['trials']} [{100 * rate['lower']:.0f}, {100 * rate['upper']:.0f}]"


def structural_surfaces(rows):
    """``{surface: (reached, scored)}``: how many questions each surface reached structurally, of those it was
    scored on (a trace without citations leaves ``cited`` and ``assembled`` unscored). ``assembled`` is coverage:
    what the assemblies cited, delivered or not; it is never part of reach."""
    table = {}
    for name in ('injected', 'tools', 'cited', 'assembled'):
        scored = [row['reach_surfaces'][name] for row in rows if row.get('reach_surfaces')
                  and row['reach_surfaces'].get(name) is not None]
        table[name] = (sum(bool(item['any']) for item in scored), len(scored))
    return table


def _two(table):
    return (f"{table['reached_right']}/{table['reached_wrong']}/{table['missed_right']}/{table['missed_wrong']}"
            f" ({table['unscored']} unscored)")


def markdown(result):
    """A plain summary: counts, rates and classes only; no trace text and no value."""
    lines = [f"# Memory re-score: {result['label']}", '',
             f"Dataset {result['dataset'].get('version')} ({result['dataset'].get('split')}); reference arm "
             f"{result['reference_arm']}. Gold: " + ', '.join(f'{name} {count}' for name, count in
                                                             sorted(result['gold_basis'].items())) + '.', '',
             result['basis'], '',
             'Probe turns with no model request recorded (reach unscored, out of every reach denominator): '
             + ', '.join(f"{arm} {data['probe_unobserved']}" for arm, data in result['arms'].items()) + '.', '',
             '| Arm | Right | Reach any (structural) | Reach all | Reach (values, weaker) | Reached right/wrong, '
             'missed right/wrong | Traps |', '| --- | --- | --- | --- | --- | --- | --- |']
    for arm, data in result['arms'].items():
        overall = data['summary']['overall']
        values = [row['reach_value'] for row in data['rows'] if row['reach_value'] and row['reach_value']['any']
                  is not None]
        weaker = scoring.wilson(sum(item['any'] for item in values), len(values)) if values else None
        traps = ', '.join(f'{name} {count}' for name, count in overall['traps'].items()) or 'none'
        lines.append(f"| {arm} | {_cell(overall['right'])} | {_cell(overall['reach_any'])} | "
                     f"{_cell(overall['reach_all'])} | {_cell(weaker)} | {_two(overall['two_by_two'])} | {traps} |")
    lines += ['', 'Where the gold messages were reached structurally, by surface (cited: the sources the host\'s '
              'assembled context cited in a section that reached a model request, read through the ledger map; '
              'assembled: every source the assemblies cited, delivered or not, which is coverage and never reach):',
              '', '| Arm | Injected section | Memory tools | Cited by a delivered assembly | Cited by any assembly |',
              '| --- | --- | --- | --- | --- |']
    for arm, data in result['arms'].items():
        table = structural_surfaces(data['rows'])
        lines.append(f"| {arm} | " + ' | '.join(f'{reached}/{scored}' for reached, scored in table.values()) + ' |')
    lines += ['', 'Where the weaker value check found the expected values, by surface (a host\'s own notes are its '
              'rewritten memory and cite no source, so only this check sees them):', '',
              '| Arm | Injected section | Memory tools | Notes | Session search reached a gold session |',
              '| --- | --- | --- | --- | --- |']
    for arm, data in result['arms'].items():
        rows = [row for row in data['rows'] if row['reach_value'] and row['reach_value']['any'] is not None]
        found = {name: sum(bool(row['reach_value_surfaces'][name]) for row in rows)
                 for name in ('injected', 'tools', 'notes')}
        sessions = [row['reach_session']['any'] for row in data['rows'] if row['reach_session']]
        lines.append(f"| {arm} | {found['injected']}/{len(rows)} | {found['tools']}/{len(rows)} | "
                     f"{found['notes']}/{len(rows)} | {sum(sessions)}/{len(sessions)} |")
    lines += ['', '## Per type', '']
    for arm, data in result['arms'].items():
        lines += [f'### {arm}', '', '| Type | Questions | Right | Reach any | Reached right/wrong, missed right/wrong '
                  '| Traps |', '| --- | --- | --- | --- | --- | --- |']
        for name, table in data['summary']['types'].items():
            traps = ', '.join(f'{key} {count}' for key, count in table['traps'].items()) or 'none'
            lines.append(f"| {name} | {table['questions']} | {_cell(table['right'])} | {_cell(table['reach_any'])} "
                         f"| {_two(table['two_by_two'])} | {traps} |")
        lines.append('')
    lines += ['## Dead-value lines', '',
              '| Arm | Lines (report) | Stale | Identical key | Different key | Raw quotation | Unknown | Decoy | '
              'Basis |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for arm, data in result['arms'].items():
        dead = data['dead_lines']
        by = dead['by_class']
        basis = ', '.join(f'{name} {count}' for name, count in sorted(dead['by_basis'].items())) or 'none'
        lines.append(f"| {arm} | {dead['lines']} ({dead['report_lines']}) | {dead['stale_lines']} | "
                     + ' | '.join(str(by.get(name, 0)) for name in CLASSES) + f' | {basis} |')
    lines += ['', 'Classes: identical key (a missed supersession on one key), different key (the same fact under '
              'another key), raw quotation (a side shown only as a quotation), unknown (neither key shown), decoy '
              '(the value was never current).', '', '## Contrasts', '']
    for name, contrasts in result['contrasts'].items():
        for key, value in contrasts.items():
            if value is None:
                continue
            if value['verdict'] == 'unavailable':
                lines.append(f"- {name}, {key.replace('_', ' ')}: unavailable, {value['unavailable_units']} of "
                             f"{value['declared_units']} scenarios have an unattributed attempt in an arm; no partial "
                             "cohort is tested.")
                continue
            lines.append(f"- {name}, {key.replace('_', ' ')}: {value['wins']}/{value['ties']}/{value['losses']} "
                         f"wins/ties/losses over {value['units']} scenarios (repetitions averaged), "
                         f"exact sign test p = "
                         f"{value['sign_test']['p_two_sided']:.4f}, delta {value['delta_pp']:+.1f} pp, "
                         f"95% interval [{value['ci_pp']['lower']:+.1f}, {value['ci_pp']['upper']:+.1f}].")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', type=Path, action='append', required=True, help='a paired results directory')
    parser.add_argument('--gold-dataset', type=Path, action='append', default=[],
                        help='a generated dataset rendered today from the run\'s seed (lends gold on an exact match)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    rendered = RenderedGold(args.gold_dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in args.results:
        result = rescore(directory, rendered=rendered)
        name = re.sub(r'[^A-Za-z0-9_.-]', '-', str(result['label'] or directory.name))
        (args.output / f'{name}.json').write_text(json.dumps(result, indent=1, sort_keys=True, default=sorted) + '\n')
        (args.output / f'{name}.md').write_text(markdown(result))
        print(json.dumps({'label': result['label'], 'gold_basis': result['gold_basis'],
                          'dead_lines': {arm: data['dead_lines']['by_class'] for arm, data in result['arms'].items()}}))
    if rendered.mismatches:
        print(json.dumps({'rendered_gold_mismatches': sorted(set(rendered.mismatches))}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
