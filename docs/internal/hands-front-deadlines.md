# Hands reply deadlines

Owner turns have no front time limit, token ceiling, tool-call cap or per-tool
timeout ceiling. A result review without a confirmed visible reply gets one
yield line after 10 seconds of tool work. The work clock starts at the first
`pre_tool_call` and counts only the union of intervals with an outstanding tool
call, ending at `post_tool_call`; concurrent tools count once and every model
wait counts zero. A turn without tools has no deadline, however slowly its
final reply arrives.
If Hermes abandons a tool callback, provider entry closes that outstanding
interval and ignores late tool notifications during the request. Waiting time
cannot be charged to a later tool boundary.

The watchdog reads live configuration and checks for a waiting owner. Time
alone never makes it redirect, interrupt or inject. At a completed tool
boundary, the review's yield line queues native `steer`, which the next request
carries. It asks for the requested reply using finished results, handing off
only unfinished work. There is no hard deadline or deadline continuation. A
long tool keeps its caller-supplied timeout or its native default.

A per-turn lock in `llm_execution` covers the provider call. A delayed tool
boundary cannot take that lock during the call and therefore cannot steer it.
This also covers Hermes advancing after a hook timeout. The owner's own model
request is never cut short by the front.

A waiting owner can interrupt an in-flight result review. The watchdog checks
that the review's own provider call is active, silences its output callbacks,
and calls `hard_interrupt` once. The front does not acquire Hermes' plain
interrupt lock around that call, which would deadlock when the native method
acquired it again. A review can also be deferred before its request starts.
After the owner turn settles, the deferred review is retried through gateway
injection. Verified owner ingress keeps owner provenance; internal completion
text must match owned result receipts to classify as a review.

A `send_message` tool marks the turn visible only when its strict success
receipt confirms nonempty visible text to this session's exact platform and
chat. Failed or skipped sends, silence markers, reactions, aliases, default
targets and ambiguous results do not qualify; repeated tool-call receipts are
ignored. A confirmed visible tool reply suppresses the review's timed yield.

There is no deadline breaker, delivery-ledger watcher or stream-delivery
observer. The front publishes neither a `deadline_status()` bridge nor a
`hands-spool/front-deadlines.json` snapshot. Unloading stops its watchers,
restores callbacks, closes settlement observation and removes its bridge.

`mind.faculties.hands: false` is read through the existing runtime config loader
on every tick and boundary, with no added cache. It latches deadlines off for
that turn within one tick and prevents deferred review injections too. The
older `hands.front.watchdog: false` switch remains supported. Re-enabling starts
budgets on later turns, without replaying a disabled turn's old deadlines.

Missing bindings and non-cooperating tools limit interruption; the host must
enable gateway injection for deferred reviews. Offline tests do not replace
qualification against the deployment model before switch-on.
