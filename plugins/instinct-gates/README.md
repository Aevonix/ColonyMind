# ColonyMind gates for Hermes

`colonymind-gates` is an optional second plugin in the `colonymind-hermes` package. It puts three gates on seams stock
Hermes already has, each decided by ColonyMind's Instinct (the mind's quick typed decisions) and measured in shadow
before it may act:

| Gate | Seam | What it decides |
| --- | --- | --- |
| Cron wake | a cron job's pre-run script | whether a fire has anything new; if not, the agent run is skipped, never more than a few times in a row |
| Inbound triage | `pre_gateway_dispatch` | whether a non-owner's message is addressed to the assistant (logged only, never applied, in this release) |
| Tool-result screen | `transform_tool_result` | whether an untrusted tool's result carries instructions aimed at an AI assistant; live, it adds a notice line and deletes nothing |

Hermes itself is not changed, and the plugin adds no tool, command or prompt text. Every gate is off until its own
entry under `plugins.colonymind-gates` in Hermes' config turns it on, and every gate fails open: a down, slow or
erroring sidecar, a bad configuration or an unreadable source leaves Hermes doing what it did before. A gate acts
only on a decision the sidecar says to apply, which happens only once a tier of its question has been promoted on
evidence; until then `colonymind instinct log --point <name>` shows what it would have done, by reference only. A
secret is hidden from any text before it leaves Hermes.

```yaml
plugins:
  enabled: [colonymind, colonymind-gates]
  colonymind-gates:
    screen: {enabled: true}
```

See [Hermes gates](../../docs/HERMES-GATES.md) for each gate's configuration and [Instinct](../../docs/DECISIONS.md)
for how a decision tier is measured, promoted and demoted.
