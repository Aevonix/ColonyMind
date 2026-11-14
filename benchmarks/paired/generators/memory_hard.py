"""Dev templates for the harder memory family (mind-memory-2).

mind-memory-1 states every fact in the episode's own turns, minutes before the probe, where stock Hermes
reaches it as easily as recall does. Here the facts live in a long seeded history: 28 to 40 earlier owner sessions over
about three months (``history``, imported into every arm before the first turn without model calls),
each fact the probe needs stated once and many sessions back, among near-duplicates of the same kind.
The live episode is one neutral owner turn, one night crossed (a day of body clock, then a tick), a
process restart and the probe in a new session, the only request in the episode.

What makes a template hard:

- ``buried-fact``: a place stated once in the first sessions; the item comes up again without it.
- ``buried-correction``: a venue stated early and changed once, at least fifteen sessions later, among
  the same item's review with another contact and the same contact's review of another item.
- ``channel-switch``: an e-mail-style session lends an object out, a later session says it came back,
  and a terse text-message session names who has it now and where: the probe comes in prose, on
  another channel, with other words.
- ``assistant-misquote``: the assistant restated the owner's place with a near-duplicate wrong one, and
  restated it twice more in later sessions; the owner never repeated it. The probe asks what the owner
  said: recall that ties the owner's words with the assistant's restatement picks the wrong one.
- ``restated-stale``: the assistant restated a venue in two later recaps; the owner then changed it once.
- ``buried-preference``: a standing preference (durations in minutes) stated early, applied to a
  duration stated many sessions later, beside other sessions' durations.
- ``spread-total``: a quantity stated in three sessions, to be added up, among the same contact's
  quantities of other files and other contacts' quantities of the same files.
- ``near-miss`` (abstain): the item comes up in three sessions, never with a place, while other things
  are placed in those same sessions.
- ``retracted`` (abstain): a place stated, then withdrawn with no replacement.

Every probe that asks for a name allows "unknown", in the same words whatever the type, so a probe
never tells a recall type from an abstention (the number probes ask for a number). Setup text and history
never say "today" or "yesterday": the history is months old when the probe arrives, and every fact in it
is one that does not go stale by itself (where things are kept, who has them, where a monthly review is
held, standing preferences, counts). Contact ids are fixed width; every contact the history
names is in contacts.json. Sessions are named by their channel (``chat-NN``, ``email-NN``, ``sms-NN``);
the history importer records them all on one platform, so a channel is a session with its own style.

Every artifact carries its ``gold`` (paired_cases.validate_gold): the owner messages in the history whose words
answer the probe (``{session, message}``), and the values that were current once and were replaced or withdrawn
(``stale``), with the answer field each held when the answer has several (``stale_fields``). An assistant's
misquote is forbidden but is no stale value: the owner never held it. Gold is taken from where the template planted
its statements, so it adds no draw and leaves the rendered history unchanged.
"""
import calendar
import json

FAMILY = 'mind-memory-2'
ROLE = 'reasoning'
ANSWER = 'answer.json'
# One night crossed before the probe, as in mind-memory-1: a day of body clock, then a tick.
NIGHT = [{'advance_clock': 86400}, {'tick': 1}]
SESSIONS = (28, 40)          # history sessions per scenario
EXCHANGES = (2, 4)           # filler exchanges (an owner turn and the assistant's reply) per history session
EARLY = 4                    # a buried statement falls in one of the first EARLY sessions
MIN_GAP = 15                 # sessions between a buried statement and its correction
TAIL = 4                     # every planted statement leaves at least TAIL later sessions
HISTORY_START = calendar.timegm((2026, 3, 2, 0, 0, 0))
CHANNELS = ('chat', 'email', 'sms')
ITEMS = ['quarterly report', 'signed lease', 'budget draft', 'travel itinerary', 'meeting agenda',
         'insurance form', 'parcel receipt', 'design brief', 'grant summary', 'audit checklist',
         'venue contract', 'reading list', 'tax return', 'floor plan', 'supplier invoice', 'training manual']
OBJECTS = ['blue folder', 'spare key', 'label printer', 'desk lamp', 'first-aid kit', 'tool box',
           'camera bag', 'extension cord', 'spare charger', 'measuring tape', 'step ladder', 'coffee grinder']
# No place or venue is a substring of another, so a forbidden stale value never matches a live one.
PLACES = ['hall cupboard', 'second drawer', 'storage room', 'back office', 'garage loft', 'green cabinet',
          'filing tray', 'boot room', 'metal locker', 'tall bookcase', 'plastic crate', 'utility closet']
VENUES = ['east room', 'corner office', 'main hall', 'garden annex', 'reading room', 'front lobby',
          'small studio', 'north wing', 'west room', 'board room']
# The assistant's near-duplicate restatement of an owner's place; no restated place is in PLACES, so only
# the assistant ever says it.
MISQUOTES = [('hall cupboard', 'kitchen cupboard'), ('second drawer', 'third drawer'), ('green cabinet', 'grey cabinet'),
             ('garage loft', 'garage shed'), ('filing tray', 'post tray'), ('metal locker', 'wooden locker'),
             ('plastic crate', 'wooden crate'), ('tall bookcase', 'low bookcase')]
HOURS = (2, 5)
BOXES = (2, 7)
CONTACTS = (6, 9)            # contacts named in one scenario's history
MINUTE_LABELS = ['minutes', 'minute', 'min', 'mins']
UNKNOWN = 'unknown'
SETUP = ['Back at my desk after a long stretch away. Nothing to do right now.',
         'Catching up on my notes this week. No action needed at the moment.',
         'A quiet stretch here. Nothing for you to do yet.']
# Talk that states no fact: the item and the contact come up, nothing is kept, lent, counted or scheduled.
# A session uses each pattern at most once.
CHATTER = [('Long week; the {item} took most of it.', 'Sounds like a lot.'),
           ('{contact} asked how the {item} is going. I said fine.', 'Noted.'),
           ('Spent the afternoon on the {item} with {contact}.', 'Sounds productive.'),
           ('Finally made progress on the {item}.', 'Good to hear.'),
           ('Had a long call with {contact}, mostly about the {item}.', 'Noted.'),
           ('The {item} is taking longer than I hoped.', 'That happens.')]
PLACE_SCHEMA = '{"place": string}'
PLACE_PROBES = [
    'Where do I keep the {thing}? Write ' + ANSWER + ' as exactly ' + PLACE_SCHEMA + ', the place as I named it, '
    'or "unknown" if you do not know.',
    'Write ' + ANSWER + ' as exactly ' + PLACE_SCHEMA + ' with where I keep the {thing}, the place as I named it; '
    'write "unknown" if you do not know.']
SAID_PROBES = [
    'Where did I say the {thing} should go? Write ' + ANSWER + ' as exactly ' + PLACE_SCHEMA + ', the place as I '
    'named it, or "unknown" if you do not know.',
    'Write ' + ANSWER + ' as exactly ' + PLACE_SCHEMA + ' with where I said the {thing} goes, the place as I named '
    'it; write "unknown" if you do not know.']
VENUE_PROBES = [
    'Where is the monthly {item} review with {contact} held now? Write ' + ANSWER + ' as exactly ' + PLACE_SCHEMA
    + ', the place as I named it, or "unknown" if you do not know.',
    'Write ' + ANSWER + ' as exactly ' + PLACE_SCHEMA + ' with where the monthly {item} review with {contact} '
    'is held now, the place as I named it; write "unknown" if you do not know.']
HOLDER_PROBES = [
    'Writing from my laptop this time. Who has my {thing} at the moment, and where is it kept? Write ' + ANSWER
    + ' as exactly {"contact": string, "place": string}, the contact as its id and the place as I named it; '
    'write "unknown" for any part you do not know.',
    'Different device, same me: write ' + ANSWER + ' as exactly {"contact": string, "place": string} for my '
    '{thing}, the id of the contact who has it now and the place it is kept as I named it; write "unknown" for '
    'any part you do not know.']
DURATION_PROBES = [
    'How long is the {item} session with {contact}? Write ' + ANSWER + ' as exactly '
    '{"duration": number, "unit": string}.',
    'Write ' + ANSWER + ' as exactly {"duration": number, "unit": string} for the length of the {item} session '
    'with {contact}.']
TOTAL_PROBES = [
    'How many boxes of the {item} files has {contact} dropped off in all? Write ' + ANSWER + ' as exactly '
    '{"boxes": number}.',
    'Write ' + ANSWER + ' as exactly {"boxes": number} with the total number of boxes of the {item} files '
    '{contact} has dropped off.']


def _fill(template, **values):
    """A probe with its names filled in; the JSON braces of the requested shape stay as they are."""
    for name, value in values.items():
        template = template.replace('{' + name + '}', value)
    return template


def _owner(text, session):
    return {'session_id': session, 'user': text}


def _exchange(owner, assistant):
    return [{'role': 'user', 'content': owner}, {'role': 'assistant', 'content': assistant}]


def _artifact(expect, forbidden=(), gold=None):
    """A JSON file with exactly the expected keys; strings compare case-insensitively, numbers exactly."""
    assertions = [{'path': [], 'op': 'keys_equal', 'value': sorted(expect)}]
    for key in sorted(expect):
        value = expect[key]
        if isinstance(value, list):
            assertions.append({'path': [key], 'op': 'label_one_of', 'value': value})
        elif isinstance(value, str):
            assertions.append({'path': [key], 'op': 'label_one_of', 'value': [value]})
        else:
            assertions.append({'path': [key], 'op': 'number', 'value': value})
    spec = {'path': ANSWER, 'format': 'json', 'assertions': assertions, 'forbidden': list(forbidden)}
    if gold is not None:
        spec['gold'] = gold
    return spec


class History:
    """A scenario's earlier sessions: planted exchanges at chosen sessions, filler around them.

    Filler never states a fact whose key is blocked (the template's own facts) and states every other
    fact at most once: each thing has at most one place, each object one borrower, each (item, contact)
    one review venue and one box count, so the filler never contradicts itself or the template.
    """

    def __init__(self, draw, contacts):
        self.draw, self.contacts = draw, contacts
        self.count = draw.integer(*SESSIONS)
        self.planted = [[] for _ in range(self.count)]
        self.channels = [draw.pick(CHANNELS) for _ in range(self.count)]
        self.used = set()
        self.placed = {}          # thing -> place, for things the filler or the template placed
        self.chatted = set()      # the chatter patterns the current session already used
        self.located = {}         # id of a planted exchange -> where render put its owner message

    def block(self, *keys):
        self.used.update(keys)

    def plant(self, index, owner, assistant, channel=None):
        """Plant an exchange in session ``index``; the returned exchange locates its owner message after render."""
        exchange = _exchange(owner, assistant)
        self.planted[index].append(exchange)
        if channel is not None:
            self.channels[index] = channel
        return exchange

    def evidence(self, *exchanges):
        """Gold evidence: where render put the owner message of each planted exchange."""
        return [self.located[id(exchange)] for exchange in exchanges]

    def place(self, index, thing, place):
        """A planted place for another thing: the owner states it and it counts as that thing's place."""
        self.block(('place', thing), ('loan', thing))
        self.placed[thing] = (index, place)
        self.plant(index, self.draw.pick([f'The {thing} is kept in the {place} now.',
                                          f'I put the {thing} in the {place}; that is where it lives.']),
                   self.draw.pick(['Noted.', f'Noted: the {thing} is in the {place}.']))

    def free(self, key):
        """Whether the filler may state this fact: never a blocked or repeated one, and a thing is either
        kept somewhere or lent out, never both."""
        if key in self.used:
            return False
        if key[0] in ('place', 'loan'):
            return ('place', key[1]) not in self.used and ('loan', key[1]) not in self.used
        return True

    def filler(self, index):
        draw = self.draw
        for _ in range(64):
            kind = draw.pick(['place', 'place', 'meeting', 'loan', 'quantity', 'chatter', 'chatter'])
            contact, item = draw.pick(self.contacts), draw.pick(ITEMS)
            if kind == 'chatter':
                pattern = draw.integer(0, len(CHATTER) - 1)
                if pattern in self.chatted:
                    continue
                self.chatted.add(pattern)
                owner, assistant = CHATTER[pattern]
                return _exchange(owner.replace('{item}', item).replace('{contact}', contact), assistant)
            if kind == 'place':
                thing, place = draw.pick(ITEMS + OBJECTS), draw.pick(PLACES)
                key = ('place', thing)
                texts = (draw.pick([f'The {thing} is kept in the {place} now.',
                                    f'I put the {thing} in the {place}; that is where it lives.']),
                         draw.pick(['Noted.', f'Noted: the {thing} is in the {place}.', 'Got it.']))
            elif kind == 'meeting':
                venue = draw.pick(VENUES)
                key = ('meeting', item, contact)
                texts = (draw.pick([f'The monthly {item} review with {contact} is held in the {venue}.',
                                    f'{contact} and I hold the monthly {item} review in the {venue}.']),
                         draw.pick(['Noted.', f'Noted: the {item} review with {contact} is in the {venue}.']))
            elif kind == 'loan':
                thing = draw.pick(OBJECTS)
                key = ('loan', thing)
                texts = (draw.pick([f'{contact} borrowed my {thing} for a while.', f'Lent the {thing} to {contact}.']),
                         'Noted.')
            else:
                key = ('quantity', contact, item)
                texts = (f'{contact} dropped off {draw.integer(*BOXES)} boxes of the {item} files.', 'Noted.')
            if not self.free(key):
                continue
            self.used.add(key)
            if kind == 'place':
                self.placed[thing] = (index, place)
            return _exchange(*texts)
        return _exchange('Nothing new on my side.', 'Noted.')

    def render(self):
        """``[{id, at, messages}]``: every session holds its planted exchanges among 2..4 filler ones."""
        sessions, day = [], 0
        for index in range(self.count):
            day += 0 if index == 0 else self.draw.integer(1, 3)
            self.chatted = set()
            exchanges = [self.filler(index) for _ in range(self.draw.integer(*EXCHANGES))]
            for exchange in self.planted[index]:
                exchanges.insert(self.draw.integer(0, len(exchanges)), exchange)
            at = HISTORY_START + day * 86400 + self.draw.integer(8, 20) * 3600 + self.draw.integer(0, 59) * 60
            session = f'{self.channels[index]}-{index + 1:02d}'
            for position, exchange in enumerate(exchanges):
                if any(exchange is planted for planted in self.planted[index]):
                    self.located[id(exchange)] = {'session': session, 'message': 2 * position}
            sessions.append({'id': session, 'at': at,
                             'messages': [message for exchange in exchanges for message in exchange]})
        return sessions


def _people(draw):
    return [draw.contact() for _ in range(draw.integer(*CONTACTS))]


def _contacts(draw, contacts):
    records = {c: {'channel': draw.pick(CHANNELS), 'address': 'capture:' + c} for c in sorted(contacts)}
    return {'contacts.json': json.dumps(records, indent=1, sort_keys=True) + '\n'}


def _episode(draw, history, contacts, probe, expect, forbidden=(), *, kind, evidence=(), stale=(),
             stale_fields=None):
    """The live episode: one neutral setup turn, the night, a restart and the probe in a new session.

    ``evidence`` are the planted exchanges whose owner messages answer the probe; the draws happen in the order
    they always have (the setup turn, the contact records, then the history)."""
    episodes = [_owner(draw.pick(SETUP), 'owner-1'), *[dict(entry) for entry in NIGHT], _owner(probe, 'owner-2')]
    files = _contacts(draw, contacts)
    sessions = history.render()
    gold = {'kind': kind, 'evidence': history.evidence(*evidence)}
    if stale:
        gold['stale'] = list(stale)
    if stale_fields is not None:
        gold['stale_fields'] = dict(stale_fields)
    return {'initial_files': files, 'history': sessions, 'episodes': episodes,
            'workflow': {'restart_before': [3], 'snapshot_after': [], 'read_failures': []},
            'artifacts': [_artifact(expect, forbidden, gold)]}


def _later(draw, history, after, gap=1):
    """A session at least ``gap`` after ``after`` that still leaves TAIL later sessions."""
    return draw.integer(after + gap, history.count - 1 - TAIL)


def buried_fact(draw):
    contacts = _people(draw)
    history = History(draw, contacts)
    thing, place = draw.pick(ITEMS + OBJECTS), draw.pick(PLACES)
    history.block(('place', thing), ('loan', thing))
    stated = draw.integer(0, EARLY - 1)
    statement = history.plant(stated, draw.pick([f'The {thing} is kept in the {place} now.',
                                                 f'I put the {thing} in the {place}; that is where it lives.']),
                              'Noted.')
    for index in sorted(draw.picks(range(stated + 1, history.count), 2)):
        contact = draw.pick(contacts)
        history.plant(index, draw.pick([f'{contact} asked about the {thing}. Nothing to do there.',
                                        f'Used the {thing} again for the audit with {contact}.']), 'Noted.')
    probe = _fill(draw.pick(PLACE_PROBES), thing=thing)
    return _episode(draw, history, contacts, probe, {'place': place}, kind='recall', evidence=[statement])


def buried_correction(draw):
    contacts = _people(draw)
    contact, other_contact = draw.picks(contacts, 2)
    history = History(draw, contacts)
    item, other_item = draw.picks(ITEMS, 2)
    old, new, near, far = draw.picks(VENUES, 4)
    history.block(('meeting', item, contact), ('meeting', item, other_contact), ('meeting', other_item, contact))
    first = draw.integer(0, EARLY - 1)
    change = _later(draw, history, first, MIN_GAP)
    history.plant(first, draw.pick([f'The monthly {item} review with {contact} is held in the {old}.',
                                    f'{contact} and I hold the monthly {item} review in the {old}.']), 'Noted.')
    moved = history.plant(change, draw.pick([
        f'A change to something from a while back: the {item} review with {contact} has moved to the {new} for good.',
        f'The {item} review with {contact} is not in the {old} any more; it is in the {new} from now on.']), 'Noted.')
    history.plant(draw.integer(first + 1, history.count - 1),
                  f'The monthly {item} review with {other_contact} is held in the {near}.', 'Noted.')
    history.plant(draw.integer(first + 1, history.count - 1),
                  f'{contact} and I hold the monthly {other_item} review in the {far}.', 'Noted.')
    probe = _fill(draw.pick(VENUE_PROBES), item=item, contact=contact)
    return _episode(draw, history, contacts, probe, {'place': new}, forbidden=[old], kind='update', evidence=[moved],
                    stale=[old])


def channel_switch(draw):
    contacts = _people(draw)
    before, holder = draw.picks(contacts, 2)
    history = History(draw, contacts)
    thing = draw.pick(OBJECTS)
    old_place, new_place = draw.picks(PLACES, 2)
    history.block(('loan', thing), ('place', thing))
    lent = draw.integer(0, EARLY - 1)
    returned = draw.integer(lent + 4, lent + 10)
    moved = _later(draw, history, returned, 3)
    history.plant(lent, f'Hello. I am lending my {thing} to {before} for the next few weeks; it will be kept in '
                        f'the {old_place} at their end. Nothing needed from you.', 'Noted.', channel='email')
    history.plant(returned, draw.pick([f'{before} gave the {thing} back.', f'The {thing} is back from {before}.']),
                  'Good to know.', channel='chat')
    now = history.plant(moved, draw.pick([
        f'fyi {holder} has the {thing} now. kept in the {new_place}. no need to reply',
        f'{holder} took the {thing}. its in the {new_place} now. nothing needed']), 'Noted.', channel='sms')
    probe = _fill(draw.pick(HOLDER_PROBES), thing=thing)
    return _episode(draw, history, contacts, probe, {'contact': holder, 'place': new_place},
                    forbidden=[before, old_place], kind='update', evidence=[now], stale=[before, old_place],
                    stale_fields={before: 'contact', old_place: 'place'})


def assistant_misquote(draw):
    contacts = _people(draw)
    history = History(draw, contacts)
    thing = draw.pick(ITEMS + OBJECTS)
    said, restated = draw.pick(MISQUOTES)
    history.block(('place', thing), ('loan', thing))
    stated = draw.integer(0, EARLY + 1)
    first_recap = draw.integer(stated + 4, stated + 10)
    second_recap = _later(draw, history, first_recap, 4)
    statement = history.plant(stated, draw.pick([f'The {thing} goes in the {said} from now on.',
                                                 f'From now on the {thing} lives in the {said}.']),
                              draw.pick([f'Got it: the {thing} goes in the {restated}.',
                                         f'Noted, the {thing} lives in the {restated}.']))
    others = draw.picks([name for name in ITEMS + OBJECTS if name != thing], 2)
    places = draw.picks([name for name in PLACES if name != said], 2)
    for other, place in zip(others, places):
        history.place(draw.integer(0, first_recap - 1), other, place)
    history.plant(first_recap, draw.pick(['Can you recap where a few things are kept?',
                                          'Remind me where some things are kept, please.']),
                  f'From what you have told me: the {thing} is in the {restated}, the {others[0]} is in the '
                  f'{places[0]}, and the {others[1]} is in the {places[1]}.')
    history.plant(second_recap, f'Where did the {thing} end up again?', f'The {thing} is in the {restated}, as you told me.')
    probe = _fill(draw.pick(SAID_PROBES), thing=thing)
    return _episode(draw, history, contacts, probe, {'place': said}, forbidden=[restated], kind='recall',
                    evidence=[statement])


def restated_stale(draw):
    contacts = _people(draw)
    contact = draw.pick(contacts)
    history = History(draw, contacts)
    item = draw.pick(ITEMS)
    old, new = draw.picks(VENUES, 2)
    history.block(('meeting', item, contact))
    first = draw.integer(0, EARLY - 1)
    recap = draw.integer(first + 3, first + 6)
    again = draw.integer(recap + 3, recap + 6)
    change = _later(draw, history, again, 3)
    history.plant(first, draw.pick([f'The monthly {item} review with {contact} is held in the {old}.',
                                    f'{contact} and I hold the monthly {item} review in the {old}.']), 'Noted.')
    history.plant(recap, f'What do I have with {contact} each month?', f'The monthly {item} review with {contact}, held in the {old}.')
    history.plant(again, f'Remind me where the {item} review with {contact} happens.',
                  f'The {item} review with {contact} is held in the {old}.')
    moved = history.plant(change, draw.pick([f'The {item} review with {contact} is in the {new} from now on.',
                                             f'{item} review with {contact}: {new} from now on.']), 'Noted.')
    probe = _fill(draw.pick(VENUE_PROBES), item=item, contact=contact)
    return _episode(draw, history, contacts, probe, {'place': new}, forbidden=[old], kind='update', evidence=[moved],
                    stale=[old])


def buried_preference(draw):
    contacts = _people(draw)
    contact, *others = draw.picks(contacts, 3)
    history = History(draw, contacts)
    item, *other_items = draw.picks(ITEMS, 3)
    hours, *other_hours = draw.picks(range(HOURS[0], HOURS[1] + 1), 3)
    preference = draw.integer(0, EARLY - 1)
    fact = _later(draw, history, preference, 8)
    rule = history.plant(preference, draw.pick([
        'A standing preference: whenever you give me a duration, state it in minutes, never in hours.',
        'Please remember how I like durations: always in minutes, not hours.']), 'Noted, minutes from now on.')
    booked = history.plant(fact, draw.pick([f'The {item} session with {contact} runs {hours} hours.',
                                            f'{contact} booked {hours} hours for the {item} session.']),
                           f'Noted: {hours} hours.')
    for other, other_item, length in zip(others, other_items, other_hours):
        history.plant(draw.integer(0, history.count - 1),
                      f'The {other_item} session with {other} runs {length} hours.', 'Noted.')
    probe = _fill(draw.pick(DURATION_PROBES), item=item, contact=contact)
    return _episode(draw, history, contacts, probe, {'duration': hours * 60, 'unit': list(MINUTE_LABELS)},
                    kind='recall', evidence=[rule, booked])


def spread_total(draw):
    contacts = _people(draw)
    contact, other = draw.picks(contacts, 2)
    history = History(draw, contacts)
    item, other_item = draw.picks(ITEMS, 2)
    counts = [draw.integer(*BOXES) for _ in range(3)]
    history.block(('quantity', contact, item), ('quantity', contact, other_item), ('quantity', other, item))
    first = draw.integer(0, EARLY)
    second = draw.integer(first + 4, first + 8)
    third = _later(draw, history, second, 4)
    texts = [f'{contact} dropped off {counts[0]} boxes of the {item} files.',
             f'Another {counts[1]} boxes of the {item} files came in from {contact}.',
             f'{contact} brought {counts[2]} more boxes of the {item} files.']
    drops = [history.plant(index, text, 'Noted.') for index, text in zip((first, second, third), texts)]
    history.plant(draw.integer(0, history.count - 1),
                  f'{contact} dropped off {draw.integer(*BOXES)} boxes of the {other_item} files.', 'Noted.')
    history.plant(draw.integer(0, history.count - 1),
                  f'{other} dropped off {draw.integer(*BOXES)} boxes of the {item} files.', 'Noted.')
    probe = _fill(draw.pick(TOTAL_PROBES), item=item, contact=contact)
    return _episode(draw, history, contacts, probe, {'boxes': sum(counts)}, kind='recall', evidence=drops)


def near_miss(draw):
    contacts = _people(draw)
    history = History(draw, contacts)
    item = draw.pick(ITEMS)
    history.block(('place', item))
    first, contact, other = draw.pick(contacts), draw.pick(contacts), draw.pick(contacts)
    history.block(('meeting', item, first), ('quantity', other, item))
    sessions = sorted(draw.picks(range(history.count), 3))
    venue = draw.pick(VENUES)
    mentions = [(f'The monthly {item} review with {first} is held in the {venue}.', 'Noted.'),
                (f'{other} dropped off {draw.integer(*BOXES)} boxes of the {item} files.', 'Noted.'),
                (f'{contact} asked about the {item}. Nothing to do there.', 'Noted.')]
    nearby = draw.picks([name for name in ITEMS + OBJECTS if name != item], 3)
    places = draw.picks(PLACES, 3)
    for index, mention, thing, place in zip(sessions, mentions, nearby, places):
        history.plant(index, *mention)
        history.place(index, thing, place)
    probe = _fill(draw.pick(PLACE_PROBES), thing=item)
    # Wrong answers the history offers: a place stated beside the item, and its review's venue.
    return _episode(draw, history, contacts, probe, {'place': UNKNOWN}, forbidden=[*places, venue], kind='abstain')


def retracted(draw):
    contacts = _people(draw)
    history = History(draw, contacts)
    thing, place = draw.pick(ITEMS + OBJECTS), draw.pick(PLACES)
    history.block(('place', thing), ('loan', thing))
    stated = draw.integer(0, EARLY - 1)
    withdrawn = _later(draw, history, stated, 6)
    history.plant(stated, draw.pick([f'The {thing} is kept in the {place} now.',
                                     f'I put the {thing} in the {place}; that is where it lives.']), 'Noted.')
    withdrawal = history.plant(withdrawn, draw.pick([
        f'Scratch what I told you about where the {thing} is kept: it is not there any more, and I do not know '
        'where it ended up.',
        f'The {thing} is no longer where I said, and I have no idea where it went.']), 'Noted.')
    probe = _fill(draw.pick(PLACE_PROBES), thing=thing)
    return _episode(draw, history, contacts, probe, {'place': UNKNOWN}, forbidden=[place], kind='abstain',
                    evidence=[withdrawal], stale=[place])


TEMPLATES = {
    'buried-fact': ('recall', buried_fact),
    'buried-correction': ('recall', buried_correction),
    'channel-switch': ('recall', channel_switch),
    'assistant-misquote': ('recall', assistant_misquote),
    'restated-stale': ('recall', restated_stale),
    'buried-preference': ('recall', buried_preference),
    'spread-total': ('recall', spread_total),
    'near-miss': ('abstain', near_miss),
    'retracted': ('abstain', retracted),
}
