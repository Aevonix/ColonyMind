# Connecting Hermes

Use the [install guide](../INSTALL.md) for the current ColonyMind baseline on
stock Hermes. This guide describes the integration boundaries. Older ColonyMind installers,
manual context plugins and migration procedures are outside that baseline.

## Hermes native integration

The matching `colonymind-hermes` package provides both native registrations:

| Repository path | Registration | Responsibility |
| --- | --- | --- |
| `plugins/hermes-plugin/` | `colonymind` general plugin | Capture ordinary turns once, expose configured tools, and observe native work. |
| `plugins/colonymind-memory/` | `colonymind-memory` provider | Prepare scoped recollection before inference and use Hermes' native compression checkpoint. |

They share ColonyMind's client and durable ingestion path. The general plugin owns
ordinary turn capture when both are active; the memory provider does not write
a second copy. A general tool catalog alone does not provide automatic memory.

Hermes owns channels, interactive execution, native scheduling and subagents.
ColonyMind retains evidence and shared work associations. The general plugin does
not start a second event subscriber, scheduler or voice service. Private
hardware and channel integrations remain with the agent's deployment.

Guided setup selects one Hermes home and the actual runtime interpreter. It
preserves that profile's unrelated identity, model and channel configuration.
See the setup guide for installation, copied-adapter refresh and recovery;
installing a package alone does not prove that the selected gateway loaded it.

The MCP server ColonyMind once offered coding harnesses is retired, and
`colonymind mcp` with it; `colonymind upgrade` removes what its setup wrote, or
names it when it was edited (see the [install guide](../INSTALL.md#update)).

For concurrent tasks and detached reviews, use the explicitly documented
qualification build. ColonyMind does not silently patch or replace Hermes.

## Verify the intended behavior

For native Hermes, tell the agent a harmless distinctive fact, start another
session and verify the answer against the retained source. Test shared work
through the intended channels separately. HTTP health and a loaded plugin do
not prove either behavior.

The [architecture guide](../ARCHITECTURE.md) describes state ownership. The
[native adapter guide](../HERMES-ADAPTER.md) covers lifecycle and evaluated learning,
and the [memory provider](../../plugins/colonymind-memory/README.md) describes context
scope and outages.
