# Stock Hermes plus ColonyMind, in one container.
#
# The base is Hermes' own published image of the qualified container floor (hermes-agent 0.21.3), pinned by
# digest. Nothing of Hermes is patched: the adapter is one more package in Hermes' environment, the sidecar has an
# environment of its own, and the base image's supervisor (s6) runs ColonyMind's setup and the sidecar as services
# of their own beside Hermes'.
#
# Build it from the repository root with the release's distributions in dist/ (CI uses the qualified ones):
#
#   python -m build --outdir dist . && python -m build --outdir dist ./sidecar
#   docker build -f docker/Dockerfile -t colonymind .
#
# docker/compose.yml runs it; docs/INSTALL.md ("Container") explains the settings.
FROM nousresearch/hermes-agent:v2026.9.14@sha256:99641e57ec762c59e54cb44aa6746b7fc68c18b3c5ddb088af54234c613d9294

LABEL org.opencontainers.image.title="ColonyMind" \
      org.opencontainers.image.description="Stock Hermes with ColonyMind as its memory and mind" \
      org.opencontainers.image.source="https://github.com/Aevonix/ColonyMind" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.base.name="docker.io/nousresearch/hermes-agent:v2026.9.14" \
      org.opencontainers.image.base.digest="sha256:99641e57ec762c59e54cb44aa6746b7fc68c18b3c5ddb088af54234c613d9294"

USER root

# The adapter in Hermes' environment: one added package, plus whatever it needs that Hermes lacks. Nothing Hermes
# had installed may change, and the install may add no dependency conflict: the build compares the environment and
# its `uv pip check` before and after, and stops on any difference. (A conflict Hermes' own image already carries
# is Hermes' to fix; it is left as it is.)
RUN --mount=type=bind,source=dist,target=/tmp/dist \
    set -eu; \
    set -- /tmp/dist/colonymind_hermes-*.whl; \
    if [ "$#" != 1 ] || [ ! -f "$1" ]; then echo "dist/ must hold exactly one colonymind_hermes wheel" >&2; exit 1; fi; \
    hermes=/opt/hermes/.venv/bin/python; \
    uv pip freeze --python "$hermes" | LC_ALL=C sort > /tmp/freeze-before.txt; \
    { uv pip check --python "$hermes" 2>&1 || true; } | grep '^The package' | LC_ALL=C sort > /tmp/check-before.txt || true; \
    uv pip install --no-cache --python "$hermes" "$1"; \
    uv pip freeze --python "$hermes" | LC_ALL=C sort > /tmp/freeze-after.txt; \
    { uv pip check --python "$hermes" 2>&1 || true; } | grep '^The package' | LC_ALL=C sort > /tmp/check-after.txt || true; \
    changed="$(LC_ALL=C comm -23 /tmp/freeze-before.txt /tmp/freeze-after.txt)"; \
    if [ -n "$changed" ]; then echo "installing the adapter changed Hermes' environment:" >&2; echo "$changed" >&2; exit 1; fi; \
    added="$(LC_ALL=C comm -13 /tmp/check-before.txt /tmp/check-after.txt)"; \
    if [ -n "$added" ]; then echo "installing the adapter added a conflict:" >&2; echo "$added" >&2; exit 1; fi; \
    rm /tmp/freeze-before.txt /tmp/freeze-after.txt /tmp/check-before.txt /tmp/check-after.txt

# The sidecar in an environment of its own, on uv's own build of Python 3.12 (the interpreter the release's
# constraints are qualified on), resolved with those constraints.
RUN --mount=type=bind,source=dist,target=/tmp/dist \
    --mount=type=bind,source=requirements/release-py312.txt,target=/tmp/constraints.txt \
    set -eu; \
    set -- /tmp/dist/colonymind-*.whl; \
    if [ "$#" != 1 ] || [ ! -f "$1" ]; then echo "dist/ must hold exactly one colonymind wheel" >&2; exit 1; fi; \
    UV_PYTHON_INSTALL_DIR=/opt/colonymind/python uv venv --no-cache --python 3.12 --python-preference only-managed \
        /opt/colonymind/.venv; \
    uv pip install --no-cache --python /opt/colonymind/.venv/bin/python --constraint /tmp/constraints.txt "$1"; \
    uv pip check --python /opt/colonymind/.venv/bin/python; \
    install -d -o hermes -g hermes -m 0700 /var/lib/colonymind

COPY --chmod=0755 docker/colonymind-shim.sh /opt/colonymind/bin/colonymind
COPY --chmod=0755 docker/init.sh /opt/colonymind/docker/init
COPY --chmod=0644 docker/endpoint.py /opt/colonymind/docker/endpoint.py
COPY --chmod=0755 docker/healthcheck.sh /opt/colonymind/docker/healthcheck
# Two services of ColonyMind's own (setup, then the sidecar) and the sidecar's entry in the user bundle.
COPY docker/s6-rc.d/ /etc/s6-overlay/s6-rc.d/

ENV COLONYMIND_HOME=/var/lib/colonymind \
    COLONYMIND_INIT_HERMES_PYTHON=/opt/hermes/.venv/bin/python \
    COLONYMIND_INIT_NO_SERVICE=1 \
    PATH=/opt/colonymind/bin:${PATH}

VOLUME ["/var/lib/colonymind"]

# The intervals are this image's choice: the first start runs init before the sidecar comes up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD ["/opt/colonymind/docker/healthcheck"]
