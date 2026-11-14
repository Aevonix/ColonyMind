# ColonyMind hands plugin

This plugin runs durable delegated work in Hermes.
The manifest loads `__init__.py`; its public `MODULES` tuple fixes module names
and registration order. `register(ctx)` imports each module and calls that
module's `register(ctx)` entry point.

Registration installs the runtime hooks, tools, commands and guards. Delegation
wrappers become active only after all guards and unload callbacks register.
A failed registration refuses delegation. The plugin requires the ColonyMind
adapter; when that adapter is absent, it leaves the runtime inactive.

The public resolver seam is defined locally in `wrap.py` and exported from the
plugin root; it needs no sidecar import:

```python
def resolver(role: str, effort: int) -> Mapping[str, Any]: ...

register_route_resolver(resolver)
get_route_resolver()  # the exact callback, or None before registration
resolve(role, effort)  # explicit callback invocation; no dispatch
```

`RouteResolver` is the callable protocol for this signature. Registration rejects
a non-callable without changing the stored callback. The callback is retained
whether it is supplied before or after `register(ctx)`; plugin registration
does not invoke it merely to register the callback. An explicit `resolve` call
uses the callback when present and otherwise reads the native delegate
configuration. Runtime admission validates the selected route and contract.

Hermes loads this source package through its plugin manifest.

| Module | Responsibility |
|---|---|
| `wrap.py` | delegation wrapping |
| `dispatch.py` | shared detached dispatch |
| `spool.py` | durable event spooling |
| `budgets.py` | attempt budgets and state clocks |
| `allow.py` | tool admission |
| `effects.py` | effect journaling |
| `reconcile.py` | executor reconciliation |
| `degraded.py` | degraded operation |
| `tools.py` | hand tools |
| `commands.py` | hand commands |
| `notices.py` | notice emission |
| `front.py` | front deadlines, reviews and ingress |
| `checks.py` | result checks |
| `authority.py` | hand authority |
| `middleware.py` | request middleware |

The sidecar's `hands.platforms` defaults to an empty allow-list. Host
configuration must explicitly admit supported platforms after detached result
delivery is checked. `rcs` uses the same native dispatch and completion path as
`sms`; it requires verified owner admission, an explicit `owner.rcs_handle`, and
an instance credential owner grant for `rcs`. The faculty default is on; execution requires an admitted native runtime.

See [Hands](../../docs/HANDS.md) for configuration and runtime behavior.
