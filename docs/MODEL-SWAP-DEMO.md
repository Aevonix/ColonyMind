# A correction across a model swap

This small demonstration checks ordinary capture, corrected recall, file-backed
task continuation and contact isolation. Facts enter through user turns, not
database fixtures. It uses the native Hermes conversation API with CLI and
simulated SMS transport identities. It sends no SMS and proves no physical phone
or voice delivery.

## Demonstration scope

| Turn | Result |
| --- | --- |
| Initial fact, model A | Created a draft delivery plan with a verified file write. The ordinary turn was captured as a ColonyMind source. |
| Correction, model A | Changed Monday 09:00 / East Dock to Wednesday 14:30 / West Dock. Captured the correction and left the plan file untouched. |
| Fresh session, model B, after process and sidecar restart | Recalled the corrected arrangement, cited the earlier correction's real source identifier, updated the existing plan and completed its three-item checklist. The artifact contained no access phrase. |
| Unresolved contact, model B, before repair | **Failed.** Hermes's global `USER.md` block exposed the owner's synthetic access phrase in the first physical request and the final response. |
| Enrolled non-owner contact, model B, after repair | **Passed.** All six observed physical model requests and the final response excluded the owner's phrase and delivery details. The native memory file remained byte-identical. |

The [adapter guide](HERMES-ADAPTER.md) describes the native profile boundary
and its renderer compatibility assumption. This demonstration covers a local
plan file and contact isolation, not background scheduling or shared task dispatch.
It is separate from the frozen benchmark and its scores.

## Reproducing it

For an end-to-end run anyone can repeat, `colonymind demo` goes through capture
and recall across sessions, a mind tick and an ask you answer yourself, in a
disposable instance with a scripted model: no network, no keys, and nothing of
your own instance or Hermes home is read or changed. It walks through the
mechanics; it is not a test of any model.
