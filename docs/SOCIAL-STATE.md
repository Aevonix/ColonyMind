# Contacts, appraisals and current world state

ColonyMind keeps one canonical contact identity across verified channel handles:
an E.164 number is one person on every gateway, an email on its lower-cased form.
Name similarity creates a proposal, never an authoritative merge; the owner
confirms or rejects it as an ask. Anyone may look people up and propose a link
through `colonymind_people` (`who`, `inspect`, `propose_link`; a guest sees only
who someone is). Merging two records and setting `may_contact` or a check-in
cadence are the owner's alone: `colonymind_people` `merge`, `set_permission` and
`set_cadence` in the owner's own session, or `colonymind people` on the CLI.
A merge moves every handle through an identity-correction receipt and the
dropped record's sources through receipts of their own, so the host's
reconciliation re-attributes them; recency, cadence and an opt-out survive it.
The owner can also correct a handle and selected source attribution
(`POST /v1/host/social/contacts/correct-identity`). For a phone number, this moves
its phone-format handles and bare digits on the requested gateway together.
If the number would still identify two people on any gateway, the correction
is refused with `identity_phone_ambiguous` and the conflicting `gateway:handle_id:contact_id`
values. Nothing is recorded. Open each named contact and correct or remove that one handle
using its address exactly as shown there and the named contact as `expected_contact_id`.
Then retry the original request with the same operation ID. If correcting a named handle
answers `identity_preimage_changed`, correct the number itself on the gateway named in the
refusal (`+` form, the right contact as subject, `expected_contact_id` whoever holds that
exact form there, usually nobody). Then retry the original request: it answers
`identity_preimage_changed` if its own handle has already moved, otherwise it succeeds.
Non-phone handles still use the exact correction.
A successful correction preserves its
receipt, invalidates dependent projections and reconciles surviving sources.
A relationship update grants no permission.

The existing source ledger holds preferences and temporary appraisals; the
agent's own person, topic and approach views live in the opinion store
(docs/OPINIONS.md) with their premises, revisions, processor provenance and owner
corrections, and `colonymind_self opinions` inspects them. Recollection selects
relevant records for the current participant; private owner opinions are not
disclosed to other contacts. Erasing or correcting supporting evidence removes its
influence: the erasure or the attribution change deletes the appraisal records
derived from that source version, with their heads and corrections, in the same
transaction (no tombstones), and an attribution change queues the moved sources'
appraisals again under the new contact. Delayed extraction cannot resurrect
invalidated evidence.

Appraisals use low/moderate intensity and bounded decay, not a validated measure
of feelings. Repeated evidence does not acquire extra votes. The same extraction
call also lists outcomes: how pieces of work went as the current evidence reports
them (failed, succeeded, dismissed or corrected, with the topic and the approach
used). Outcomes are counted occurrences, not votes: "the export failed twice" is
two entries, a restatement of an earlier report adds nothing, and only the owner's
own turns are stored, once per turn, deleted with their source. An outcome that
does not validate is dropped on its own, never failing the rest of the answer, and
a contact's turn is not held to outcomes at all. They are how the
owner's statements reach the agent's own feelings (docs/MIND.md). A cited repair report
can settle an exact prior incident; the report remains attributed evidence.
Each bounded supplied incident requires an explicit unchanged, uncertain or
resolved decision in the existing extraction call. Resolution retains its
citations and target ID in a separate history receipt, without a new temporary
mood or current head. A fresh appraisal cannot duplicate that incident's exact
normalized topic. Unsupported resolution and omitted decisions fail extraction;
unchanged or uncertain decisions leave the incident intact.
Contact response delay, flattery and claimed authority do not establish character
or trust, and the appraisal extractor never turns one incident into a reading of
a person. Describing a person is the work of the agent's page on them (Pages,
below): how they are, what they care about and how to treat them, in plain lines
that each cite the turns they rest on. Formal personality results can be retained
as attributed self-reports; ordinary conversation is not a Big Five assessment,
and a page carries no score. Historical numeric personality/closeness records
remain readable for migration but no longer determine active approach guidance.

Owner corrections remain explicit historical corrections. Relevant hints can
change verbosity or suggest another method for a repeatedly failing task. These
are fallible model outputs, not decisions about credentials or tool access.
The extraction protocol exposes one citation key per evidence message; canonical
source/contact/version metadata remains server-side. A failed extraction remains
visible as a failed job rather than an invented positive result.

Typed world properties use the existing SQLite world store. Reports, observations
and hypotheses remain distinct; repeated statements do not raise confidence.
Conflicting reports retain their authors and dates. Current property views mark
staleness and check source revisions again before context emission. The scheduled
source-report extractor accepts unqualified present-state assertions; past, future
and modal claims abstain from this narrow current-state path. Historical source
memory remains available independently.

The native tools use supported Hermes hooks, memory-provider context and scoped
HTTP APIs. No new service, database engine or Hermes core patch is required.
The guided setup stores chosen values, timezone and quiet hours in private
instance configuration. Existing identity and settings survive setup reruns.

Qualification combines canonical-store/API tests, actual native plugin loading
and small synthetic LAN-model corpora. Those establish specific behaviors,
including useful preference recall and rejection of tested junk cases. They do
not establish general psychological validity, universal inference quality,
subjective experience or sustained autonomous improvement. A deployment must
observe its own ordinary-use outcomes before making those claims.

## Pages

The agent keeps one page on each person it talks with and one on itself
(`sidecar/colonymind/pages/`). A page is a short list of lines of a few kinds:

| Page | Kind of line | Written by |
|---|---|---|
| A person's | `boundary`: how the agent treats them | the agent, through `colonymind_people boundary` |
| A person's | `about`: how they are, what they care about, how to treat them | the nightly pass |
| The agent's own | `about`: how the agent is and what it cares about | the nightly pass |
| The agent's own | `like`, `dislike`: the agent's own tastes | the agent, through `colonymind_self like` and `dislike`, and the nightly pass where the agent stated one |

Every line names the source turns it rests on, and a line with no source is not
written. The lines live in the source ledger's database (`page_lines`), beside
the turns they cite: there is one memory, and erasure reaches a page. A line
stops being read once a turn it cites is forgotten, or erased after the line was
written, and the nightly pass then removes it. Merging two records of one person
joins their pages. A page line grants no permission: `may_contact`, the cadence
and the trust tier stay the owner's.

**The nightly rewrite** is one stage of the nightly pass
([consolidation](CONSOLIDATION.md)). A page is written again when its person has
a turn newer than the page's last rewrite: the agent's own page and at most six
people a night, the page written longest ago first. The model is shown the page
as it stands and the newest conversation, each turn with an id, and returns the
lines as they should stand now, at most twelve. A line that cites no turn it
was shown is dropped. A person's page is written from that person's conversation
with the agent. The agent's own page is written from what the agent itself said,
to anyone, and names nobody else. The agent's boundaries, and the tastes it
wrote itself, are the agent's to change: the night keeps them as they are.

**The agent's own lines.** `colonymind_people boundary` adds a boundary to the
page on a person (`text`) or removes one by its line id (`drop`), and with
neither shows the page. `colonymind_self like` and `dislike` do the same for the
agent's tastes (`taste`, `id`). A written line cites the turn it was written in;
without a turn id it is refused. These are the agent's own lines, not the
owner's records, so a session that is not the owner's own may set them too: it
reaches the agent's own page and the page about its own person, and no other.

**In a turn.** A turn with a person carries the agent's page on that person as
the section "Your page on this person", kept whole, and the agent's page on
itself as "Your page on yourself", whose lines are chosen against the turn's
message like every other optional item. Both count toward the context budget
(`context.budget_chars`); a person's page longer than 2,000 characters keeps its
first lines, the boundaries before the rest. With `mind.faculties.people` off a
turn carries no page on its person and the night rewrites the agent's own page
only. A turn in a group chat is given a page line only when every turn the line
cites took place in that chat ([recall](RECALL-HYBRID.md#group-chats)).

**Reading them.** The owner reads every page: `colonymind_people inspect` shows
a person's page with the person's record, each line with its id and its source
turns, and `GET /v1/mind/pages/{who}` returns one page (`self` for the agent's
own). Any other viewer reads the agent's own page and the page about
themselves, the two a turn with them carries. The routes are in the owner's area
of the API (`/v1/mind/`): the plugin calls them with the instance's credential
and names the session's person as the viewer. A read or a write from a turn in a
group chat also names the chat (`chat_type`, `chat`, `session_id`), for the
owner's turn too: it is shown a line only when every turn the line cites took
place in that chat, and the line it wrote, and `inspect` there shows who the
person is without the page.
