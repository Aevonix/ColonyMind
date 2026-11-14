"""The owner console: a read-only local web page over what the terminal already shows (docs/CONSOLE.md).

``colonymind console`` binds 127.0.0.1 on a free port and prints a link with a one-time token. The pages: the
mind's status, concerns, goals, log, why and asks; health; the faculties, Instinct, usage and bench reports; the
recall check; the memory inventory and a claim's history. No script runs on any page and nothing is fetched. Its one
control is the mind's off switch. It never answers an ask and never changes the level or a grant.

Modules: ``source`` (what it reads, and the routes it may call), ``pages`` (the pages), ``app`` (the gate and the
off switch), ``server`` (the loopback HTTP server), ``cli`` (the command).
"""
