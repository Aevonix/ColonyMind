"""The living memory twin: a live, read-only replica of an instance on a second machine.

``source`` reads each SQLite store's committed changes (its write-ahead log, or a consistent
snapshot), ``primary`` spools them as ``segment`` files with periodic ``trees`` snapshots of
the vector store and the other files, ``transport`` mirrors the spool, and ``standby``
applies it to replicas that serve reads and can be promoted. See docs/TWIN.md.
"""
