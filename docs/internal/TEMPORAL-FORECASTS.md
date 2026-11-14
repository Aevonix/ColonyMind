# Task forecasts and expected replies

Task-run and execution forecasts were removed in the job-watch build. The execution registry,
its observation endpoint, model-call readback and assessment route are gone. The owner work view no
longer carries shadow execution horizons, and no new task-turnaround or first-attempt probability
forecasts are issued. Upgrade backs up the stores before retiring the execution observation tables.

Declared reply forecasts and reply waits remain; see [Declared reply forecasts](EXPECTED-REPLY-FORECASTS.md).
Hermes owns native tasks and scheduling, and the selected delivery outbox owns attempted,
acknowledged and uncertain sends. The following transport boundaries still apply.


Trusted durable adapters can use `/v1/host/transport/ingress` to admit metadata,
claim one native handoff and read its processing receipt. The existing
communications database holds metadata; the adapter owns pending payloads.
Exact verified handles resolve contacts server-side. Canonical ingestion must
match the bound source, contact and session before processing settles. Unknown
handoffs remain pending, and erasure marks linked receipts for payload cleanup.
This does not prove successful generation or a provider send.

Prepared WhatsApp followup review additionally requires a fresh connected
interval, all sequence receipts above its connection floor through its watermark,
available media and no unresolved recipient activity. Missing coverage holds the
review. Authority verification for an original send remains independent. This
proves only local intake coverage, not complete provider history; deployment
adapters must separately qualify this protocol before claiming silence.
