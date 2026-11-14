#!/bin/sh
# The image's health check, installed as /opt/colonymind/docker/healthcheck: the sidecar answers at the address its
# instance records (sidecar.host and sidecar.port in colonymind.yaml, which COLONYMIND_SIDECAR_HOST and
# COLONYMIND_SIDECAR_PORT pin for the sidecar too; its loopback and port 7777 before init has run), read with the
# sidecar's own config loader. A container recreated on the same volumes keeps the port an earlier start recorded,
# whether or not COLONYMIND_INIT_PORT is given again. No bytecode is written: the check runs as root.
set -eu
python=/opt/colonymind/.venv/bin/python
url="$("$python" -I -B -c 'from colonymind.config import load_config; print(load_config().sidecar_url)')"
exec curl -fsS --max-time 4 -o /dev/null "$url/v1/host/health"
