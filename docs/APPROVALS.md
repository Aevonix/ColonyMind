# Owner approvals

The owner answers an ask in the chat where it arrived. A plain yes or no can
answer the ask when its message was the last thing that chat showed. The full
rules are in [MIND.md](MIND.md#which-ask-a-message-answers).

## Binding stays structural

Only the owner's authenticated message in their own direct chat can answer.
A reaction on the ask or a quoted reply binds by its delivery id, platform and
chat. Otherwise the chat ledger must show the ask last before the answer.
The existing settling and ordering windows still apply: 3 seconds after the
ask, no unordered sends within 5 seconds, and at most 12 hours for a floor or
widening ask. Full terms must have arrived before approval. A model's words,
a tool call, a typed code or a completion report cannot supply a binding.

## A hand waiting for approval

A hand holds an exact call, working directory and target values. It stops
until the owner answers. A native completion whose only result is that hold
is settled silently before a model request or streaming reply. This requires
the controller's stopped, blocked result and the executor's recorded hold.
Result prose cannot select silence. Mixed completions, artifacts, fetched
text, verified results and other observed work still receive their normal
review. A successful plain `pwd` only observes the working directory and does
not add a result to review. Failed or incomplete calls keep their review.

Every hand ask waits for the parent to finish handling that hand's native
completion, then the existing 5 second reply settling gap. This applies
whatever the hand ran before the hold, including a read-only directory look.
The controller fixes the deadline at the earlier of 3 minutes after the hold
and the hand's wall deadline. A hung hand, missing completion or hung review
cannot keep its notice waiting past that deadline. The wait survives a restart
through that stored deadline. The silent completion still writes no reply or
active turn to the answer ledger. Owner-started hand asks keep their exemption
from the conversational pause.

When an owner's answer grants the permit, the runtime automatically resumes
the owning hand exactly once. It waits for the current attempt to end before
starting its continuation. The continuation runs the held call through its exact,
single-use permit, then reports through its normal completion in the same thread.
The answer turn's context says the hand was resumed and the model must not
continue it again or perform the action itself. A duplicate continue is a no-op.
Only a grant resumes the hand; a no, a lapse or a refusal resumes nothing.
The console's yes remains refused and grants no authority.

## An answer that did not count

A clean yes or no with no binding approves and declines nothing. The answer
reader queues the most recently shown eligible open ask in that chat through
the existing show path, even when stale open asks were also shown there.
A floor or widening ask must have been shown within `FOLLOW_UP_HOURS`
(12 hours) and must not have
expired. The single-ask case uses the same path.

The turn context names the ask queued again here, after the current reply,
for a fresh yes or no. The model must not start another hand to ask again or
send the owner elsewhere. The repeated ask stays in the answer's chat even
when another owner chat becomes active. RCS uses that recorded owner message
as its native delivery reference. This routing grants no execution authority.

## Repeated held calls

An identical call digest, working directory and target values in the same
parent session replace the older open ask. The old waiting or granted permit
becomes stale before the new ask is created, under the hands write lock.
The older hand and its continuations cannot use it. The new ask needs a fresh
owner answer. Other sessions or changed bindings remain separate asks.

Permits stay exact and single use. Only the holding hand's valid continuation
may consume its permit, immediately before execution. See [HANDS.md](HANDS.md#authority).

## Direct owner requests and held calls

In the owner's own session, a front turn confirmed as owner-started by the
ingress ledger runs recognized floor effects directly without a hand, hold or
approval. A missing ingress ledger retains this owner-session exemption.
The exception is a call matching a waiting or granted permit held by a hand of
the same parent session: the same bound targets or the same call digest. The
front blocks that call because the runtime resumes the holding hand after
approval. A different recognized command on the same bound target remains
blocked, including in the owner's answer turn after approval.

Held targets come from the controller or synced hand records and survive a
plugin or gateway restart. If they cannot be read, owner floor effects are
blocked with a clear reason only while a hand of that session awaits approval.
When no hand is waiting, unavailable held targets do not block the owner's turn.

Front turns that are not owner-started, including completion reviews and
injected turns, cannot run recognized floor effects or borrow any hand's permit.
Hand effects outside the permitted continuation need their own hold and approval.
Stopped, waiting or already refused hand calls cannot consume a permit.
