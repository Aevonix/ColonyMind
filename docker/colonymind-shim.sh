#!/bin/sh
# `colonymind` on the container's PATH, installed as /opt/colonymind/bin/colonymind.
#
# The instance belongs to the hermes user, as Hermes' own data does. Called as root (a plain `docker exec`), this
# runs the command as hermes, so nothing it writes ends up owned by root where the sidecar cannot read it.
set -e
REAL=/opt/colonymind/.venv/bin/colonymind
S6_SUID=/command/s6-setuidgid

if [ "$(id -u)" != 0 ]; then
    exec "$REAL" "$@"
fi
export HOME=/opt/data
exec "$S6_SUID" hermes "$REAL" "$@"
