# Security policy

## Report a vulnerability privately

Please do not report a security problem in a public issue, discussion or pull request.

Use GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a
vulnerability**, or go straight to <https://github.com/Aevonix/ColonyMind/security/advisories/new>. Only you and the
maintainers can see the report. You need a GitHub account; no email address is involved.

Include what you can:

- the versions: `colonymind doctor --json` names the Hermes version and whether the adapter matches the sidecar
  (read it first: when a configuration file does not parse, it can quote the line, key and all, so replace every
  key, token and password with `<redacted>`); `pip show colonymind colonymind-hermes` gives the rest;
- the part affected: the sidecar, the Hermes adapter (`colonymind-hermes`) or the memory provider;
- how to reproduce it, or a proof of concept;
- what an attacker gains, and what they need first.

## What happens next

The maintainers answer in the advisory and keep the discussion there, agree a disclosure date with you, and publish
the advisory with the release that fixes the problem. You are credited in the advisory and the release notes unless
you ask not to be named.

## Supported versions

Security fixes go into the latest release; earlier releases do not receive them. Upgrade with:

    uv tool upgrade colonymind
    colonymind upgrade
    hermes gateway restart

[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) lists the Hermes and Python versions each release supports.

## Scope

Anything in this repository, for example:

- acting, sending or approving as the owner without the owner's grant, or widening what a grant allows;
- reading or writing memory through the sidecar without the instance's key;
- a secret that arrives in conversation reaching memory, logs or a model's context in the clear (see
  [docs/SECRETS.md](docs/SECRETS.md));
- erased or forgotten material coming back.

Report these elsewhere:

- a vulnerability in Hermes itself: to the Hermes project;
- a model provider or endpoint you configured: to its operator;
- an attack that already needs the instance's API key file, or control of the account ColonyMind runs as: this is
  outside ColonyMind's protection, but tell us if ColonyMind makes it worse than it has to be.
