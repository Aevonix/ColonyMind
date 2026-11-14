Host identity and language policy
=================================

ColonyMind reads agent names, values and boundaries from `identity.yaml`. Hand
identity and style come from `hands.persona_addendum` and `hands.style`. Supplied
text keeps its spacing and punctuation. The neutral hand guidance specifies task,
authority and evidence requirements.

Other language policies are optional host settings in `colonymind.yaml`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `mind.drift.validators` | `{}` | Case-insensitive regex lists for `hedge`, `agency`, `causal`, `apology`, `adjustment` and `tone`. |
| `mind.status_replacements` | `[]` | Ordered regex replacements for the generated status block. |
| `context.recall_notice_replacements` | `[]` | Ordered regex replacements for recall notice previews, after withholding and before shortening. |
| `hands.notices.forbidden_patterns` | `[]` | Case-insensitive regexes rejecting notice text. The host adapter uses the same setting in its notice configuration. |

A replacement has `pattern` and `replacement` string fields and uses Python
regex substitution semantics. Empty policies preserve punctuation and impose no
persona. Notice length and single-line transport bounds still apply.

Drift language readings are unscored without their validators. Agency requires
both `hedge` and `agency`; apology versus correction requires both `apology` and
`adjustment`. Tone measures the increase in host-pattern matches after compression
relative to other replies. Invalid policies fail configuration loading.

Missing verification counts substantive replies lacking returned tool receipts.
Captured transcript receipts require a recorded invocation and matching tool
completion before the assistant reply and after the instruction. Empty results
count. Hermes root work captures record the invocation's call identifier, tool
and completion status; successful operations verify only their own turn and
session. Late and detached work events do not verify an earlier reply.
Native observations must
have current, validated original-result digests and source dependencies, belong to
the reply's session and exact supplied instruction, and precede the reply. The
reading retains receipt references. A receipt establishes an observed tool result;
it does not establish that every statement is correct or the task succeeded.

Drift audit rows and trend entries record the validator policy and signal version.
Language trends compare only runs with the same policy and receipt semantics.
Hosts upgrading from authored framework defaults must supply their validators and
text rules explicitly. The staged configuration fixture at
`sidecar/tests/fixtures/host-language-policy.yaml` contains those exact settings.
The parity test compares actual Release-A renderers with configured renderers for
identity/task prompts, status blocks, recall notices and hand notices. Adapter
notice configuration uses the fixture's `hands.notices` as `notices`.
Deployment configuration is outside this staging change; deployment parity still
requires the host to install these values. The verification-wording regex is
retired and has no replacement setting.
