# Receive-only agent mailbox: provisioning recommendation

How to give the agent of a ColonyMind instance a mailbox at a hosted mail
provider, for the [receive-only mailbox contract](agent-mailbox.md). This is a
proposal, not provisioning: nothing in ColonyMind creates an account, a DNS
record, a payment or a provider connection. The examples use the address
`nova@example.com` with the display name `Nova`. It is a separate mailbox with
its own storage and credentials, never the owner's own mail account.

Recommend **Migadu Micro with outbound disabled at the provider**, receiving via
IMAP over TLS. The application, retained mailbox and authorization stay on the
machine that runs the ColonyMind instance. No other machine and no public
endpoint of that machine is in the delivery path. Ordinary hosted SMTP
necessarily gives the mail provider custody of plaintext mail; this is private
local processing, not end-to-end encrypted or fully self-hosted mail transport.
If “self-hosted” requires owning the SMTP server too, neither hosted option
meets it; operating an SMTP service would add a service, and this mailbox adds
none.

## Two options (public documentation checked 2026-10-17)

| Option | Published cost, excluding existing domain/tax | How mail reaches the instance | Receive-only fit |
| --- | --- | --- | --- |
| **Migadu Micro** | **US$19/year**, 5 GB, 200 incoming/day per account; no monthly Micro plan | Internet → Migadu MX → dedicated mailbox → the instance's machine opens `imap.migadu.com:993` with TLS | Provider documents separate mailbox sending/receiving/IMAP permissions. Set sending off, receiving and IMAP on. Recommended. |
| Purelymail Simple | **US$10/year**; unusually high usage can require metered pricing | Internet → Purelymail MX → dedicated mailbox → the instance's machine opens `imap.purelymail.com:993` with TLS | Lower price; the reviewed docs establish IMAP but not an independent provider-side sending prohibition. Do not choose it before the owner has established that control. |

Sources: [Migadu pricing](https://migadu.com/pricing/),
[Migadu mailbox permissions](https://migadu.com/api/#mailboxes),
[Migadu client settings](https://www.migadu.com/support/#where-are-server-parameters-for-email-clients),
[Purelymail pricing](https://purelymail.com/pricing),
[Purelymail IMAP](https://purelymail.com/docs/setup/technical).
Migadu documents incoming rate limits and standard rather than end-to-end
encryption: [provider limitations](https://www.migadu.com/procon/).
Low-volume correspondence fits the proposed use; bulk automated ingestion does not.

## DNS proposal: every change requires the owner

**[OWNER: DNS]** First inspect the current authoritative zone and existing mail
users. MX selects a receiver for the entire domain, not one address. Never add
two unrelated providers as MX priorities to split mailboxes. If `example.com`
already receives mail elsewhere, preserve its receiver and have the owner arrange
recipient-specific routing to the agent's dedicated mailbox, or plan an explicit
whole-domain migration. An address on a subdomain is a different address; it
does not replace one on the domain itself. No current DNS configuration is
assumed here.

| Record | Migadu proposal | Purelymail alternative |
| --- | --- | --- |
| MX at `example.com` | Priority 10 `aspmx1.migadu.com.`, 20 `aspmx2.migadu.com.` | Priority 50 `mailserver.purelymail.com.` |
| SPF TXT at `example.com` | Receiving requires no SPF authorization. For an entirely non-sending domain: `v=spf1 -all`. Provider's sending template includes `spf.migadu.com`; unnecessary for a receive-only mailbox. | Same receive-only rule. Sending template includes `_spf.purelymail.com`; unnecessary for a receive-only mailbox. |
| DKIM | Not required to receive. Provider template has three CNAME selectors `key1`–`key3` under `_domainkey`, targeting the corresponding selector under `example.com._domainkey.migadu.com.` | Not required to receive. Template: `purelymail1`–`purelymail3` under `_domainkey` → `key1`–`key3.dkimroot.purelymail.com.` |
| DMARC | Receiving needs no DMARC change. For a domain with no authorized senders, propose TXT `_dmarc.example.com`: `v=DMARC1; p=reject;` | Same rule. Purelymail's template instead uses CNAME `_dmarc` → `dmarcroot.purelymail.com.`; keep explicit local policy if using it. |

Record references: [Migadu DNS template](https://migadu.com/api/#domainsrecords),
[Purelymail domain setup](https://purelymail.com/docs/domainDocs).
**[OWNER: DNS]** A provider also requires domain-ownership verification. Obtain its
account-specific value directly in its portal; it is deliberately absent here.
Use the selected account's current DNS instructions at provisioning time.
If provider diagnostics insist on outgoing records, the owner must resolve that
with the provider before activation, rather than silently enabling outbound mail.

**[OWNER: DNS]** SPF/DMARC apply to other legitimate domain senders too. Preserve
their authorizations and selectors, maintain exactly one SPF record, and do not
publish `-all` or tighten DMARC over existing senders without reviewing them.
DNS anti-spoofing is not an outbound access control; provider sending prohibition
and absence of sending code enforce the agent's receive-only behavior.

## Steps and responsibility

1. **[OWNER: ACCOUNT]** Choose provider and exact address; create the administrator
   account and the agent's dedicated mailbox. Keep administrator access separate
   from the runtime. For Migadu configure `may_send=false`, `may_receive=true`,
   `may_access_imap=true`; disable unused POP/Sieve, forwarding and autoresponders.
   Keep `is_internal=false` so external mail can arrive. Confirm restrictions in
   the chosen plan; Micro excludes configurable mailbox quota limits.
2. **[OWNER: PAYMENT]** Purchase the chosen plan directly. ColonyMind starts no
   trial and no payment.
3. **[OWNER: DNS]** Approve the domain routing plan and apply only the reviewed
   records above, including ownership verification. Preserve unrelated mail.
4. **[OWNER: CREDENTIAL / INSTANCE CONFIGURATION]** Set the address's domain as
   the instance's `COLONYMIND_AGENT_MAIL_DOMAIN`
   ([contract](agent-mailbox.md#data-and-trust-boundary)), so the local identity
   is registered under the provider's address. Supply the receive-only IMAP
   credential through the existing private credential mechanism, outside source,
   reports and chat. No provider admin credential belongs on the instance's
   machine or in Companion.
5. An IMAP adapter inside the existing local runtime is future work. It is
   invoked on owner refresh/reconnect using the runtime's existing lifecycle,
   with no new service or scheduler. Fetch with UID and `BODY.PEEK`, checkpoint
   by mailbox, UIDVALIDITY and UID only after durable acceptance; reconnect/replay
   safely. Mail waits at the provider while the machine sleeps. Provider
   deletion/read state is separate from local state. No webhook, public inbound
   port, relay through another machine, forwarding from the owner's own mail
   account, SMTP client, auto-reply or delivery/read receipt is added.
6. **[OWNER: ACTIVATION / INSTANCE CONFIGURATION]** After provider setup and the
   adapter exist, attest the local identity as active. The owner can send a sample
   **to** the agent's address from an existing external mailbox; the agent sends
   nothing. Check delivery and provider outbound prohibition.

The backend works with synthetic/local inbound records without any provider.
Internet mail does not reach the instance until the adapter of step 5 exists.
