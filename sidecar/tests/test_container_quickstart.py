"""The container quickstart (``docker/``): stock Hermes' own image plus ColonyMind, in one container.

What the image may do is narrow: it starts from the upstream Hermes image of the supported release, pinned by digest,
patches nothing of Hermes, installs the adapter into Hermes' environment and the sidecar into one of its own, and
runs ColonyMind's setup and the sidecar as s6 services beside Hermes' own. CI builds it and runs the smoke inside it
(``docker/smoke.sh``); these tests hold the parts that can be checked without a container runtime: the image
definition, the service wiring, the setup step's behaviour (run here with stand-ins for the container's commands),
the compose example and the workflows that build and publish it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

from colonymind.init import INIT_ENVIRONMENT

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
DOCKERFILE = DOCKER / "Dockerfile"
SERVICES = DOCKER / "s6-rc.d"
WORKFLOWS = ROOT / ".github" / "workflows"
#: Stock Hermes' interpreter, for the checks that ask Hermes itself what it would do (CI names one, as for the install
#: tests); without it they skip.
HERMES_PYTHON = os.environ.get("COLONYMIND_TEST_HERMES_PYTHON") or (
    sys.executable if importlib.util.find_spec("hermes_cli") else "")
# The instance directory in the image, and Hermes' data (the upstream image's volume).
INSTANCE = "/var/lib/colonymind"
HERMES_DATA = "/opt/data"


def _instructions() -> list[tuple[str, str]]:
    """The Dockerfile as (instruction, arguments), continuation lines joined and comments dropped."""
    lines, current = [], ""
    for raw in DOCKERFILE.read_text().splitlines():
        stripped = raw.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.startswith("#"):
            continue
        current += (" " if current else "") + stripped.removesuffix("\\").strip()
        if not stripped.endswith("\\"):
            lines.append(current)
            current = ""
    return [(line.split(None, 1)[0].upper(), line.split(None, 1)[1] if " " in line else "") for line in lines]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


# --- the image definition ------------------------------------------------------------------------------------------

def test_the_image_starts_from_the_upstream_hermes_image_pinned_by_digest():
    froms = [arguments for instruction, arguments in _instructions() if instruction == "FROM"]
    assert len(froms) == 1, froms
    assert re.fullmatch(r"nousresearch/hermes-agent:v\d{4}\.\d+\.\d+@sha256:[0-9a-f]{64}", froms[0]), froms[0]


def test_the_base_is_the_hermes_release_ci_qualifies():
    """The tag names the release whose commit CI pins (``HERMES_REF``); the smoke checks the image's own provenance
    record against that commit, so a digest of another release cannot pass."""
    [base] = [arguments for instruction, arguments in _instructions() if instruction == "FROM"]
    tag = base.split(":", 1)[1].split("@", 1)[0]
    ci = (WORKFLOWS / "ci.yml").read_text()
    comment = re.search(r"#\s*Stock Hermes (v\d{4}\.\d+\.\d+)", ci)
    assert comment and comment.group(1) == tag
    smoke = (DOCKER / "smoke.sh").read_text()
    assert "/etc/hermes/image-provenance.json" in smoke and "HERMES_REF" in smoke


def test_the_bare_sidecar_image_is_gone():
    for leftover in ("Dockerfile", "docker-entrypoint.sh", ".dockerignore"):
        assert not (ROOT / "sidecar" / leftover).exists(), leftover


def test_the_adapter_goes_into_hermes_environment_and_the_sidecar_into_its_own():
    runs = " ".join(arguments for instruction, arguments in _instructions() if instruction == "RUN")
    assert re.search(r"colonymind_hermes-\*\.whl", runs)
    assert "hermes=/opt/hermes/.venv/bin/python;" in runs and 'uv pip install --no-cache --python "$hermes"' in runs
    assert re.search(r"/colonymind-\*\.whl", runs)
    assert "--python /opt/colonymind/.venv/bin/python" in runs
    # Both environments are checked. The build proves that nothing Hermes had installed changed and that the
    # adapter added no conflict (one Hermes' own image carries stays Hermes' to fix).
    assert runs.count("uv pip check") >= 3
    assert "pip freeze" in runs and "comm -23 /tmp/freeze-before.txt /tmp/freeze-after.txt" in runs
    assert "comm -13 /tmp/check-before.txt /tmp/check-after.txt" in runs
    # comm needs its inputs in its own order, and neither uv's output nor the locale guarantees one.
    assert runs.count("LC_ALL=C sort") == 4 and runs.count("LC_ALL=C comm") == 2
    # The sidecar resolves with the release's qualified constraints; Hermes' environment never gets them.
    assert "requirements/release-py312.txt" in runs


def test_the_distributions_are_mounted_for_the_build_and_never_copied_into_a_layer():
    for instruction, arguments in _instructions():
        if instruction in {"COPY", "ADD"}:
            assert "dist" not in arguments.split()[0:-1] and not arguments.startswith("dist"), arguments
    runs = [arguments for instruction, arguments in _instructions() if instruction == "RUN"]
    assert any("--mount=type=bind,source=dist" in arguments for arguments in runs)
    ignore = (DOCKER / "Dockerfile.dockerignore").read_text().split()
    assert ignore[0] == "*" and "!dist/*.whl" in ignore


def test_nothing_of_hermes_is_replaced():
    """Every file the image adds lands outside Hermes' install tree; its s6 additions are services of their own."""
    for instruction, arguments in _instructions():
        if instruction in {"COPY", "ADD"}:
            target = arguments.split()[-1]
            assert not target.startswith("/opt/hermes"), arguments
    for path in SERVICES.rglob("*"):
        if path.is_file():
            relative = path.relative_to(SERVICES)
            assert relative.parts[0].startswith("colonymind-") or (
                relative.parts[:2] == ("user", "contents.d") and relative.name.startswith("colonymind-")), relative


def test_no_secret_is_baked_into_the_image():
    for instruction, arguments in _instructions():
        if instruction in {"ENV", "ARG", "LABEL"}:
            for name in re.findall(r"([A-Za-z_][A-Za-z0-9_.]*)=", arguments):
                assert not re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name.upper()), (instruction, name)
    text = DOCKERFILE.read_text()
    assert "api.key" not in text and "colonymind init" not in text


def test_the_instance_directory_is_a_volume_of_its_own():
    instructions = _instructions()
    env = " ".join(arguments for instruction, arguments in instructions if instruction == "ENV")
    assert f"COLONYMIND_HOME={INSTANCE}" in env
    volumes = [arguments for instruction, arguments in instructions if instruction == "VOLUME"]
    assert any(INSTANCE in arguments for arguments in volumes)
    assert not INSTANCE.startswith(HERMES_DATA)
    runs = " ".join(arguments for instruction, arguments in instructions if instruction == "RUN")
    assert re.search(rf"install -d -o hermes -g hermes -m 0700 {INSTANCE}\b", runs)


def test_the_image_reports_its_health_and_publishes_no_port():
    instructions = _instructions()
    [health] = [arguments for instruction, arguments in instructions if instruction == "HEALTHCHECK"]
    assert "/opt/colonymind/docker/healthcheck" in health
    # The sidecar serves Hermes inside the container, on loopback; nothing is exposed.
    assert not [arguments for instruction, arguments in instructions if instruction == "EXPOSE"]


@pytest.mark.parametrize("recorded, environment, expected", [
    (None, {}, "http://127.0.0.1:7777"),
    ({"port": 8888}, {}, "http://127.0.0.1:8888"),
    ({"host": "0.0.0.0", "port": 8888}, {}, "http://127.0.0.1:8888"),
    ({"port": 8888}, {"COLONYMIND_SIDECAR_PORT": "8899"}, "http://127.0.0.1:8899"),
], ids=["before-init", "recorded-port", "all-interfaces", "pinned-by-environment"])
def test_the_health_check_asks_the_address_the_sidecar_serves(tmp_path, recorded, environment, expected):
    """The sidecar serves at the address its instance records (``sidecar.host`` and ``sidecar.port`` in
    ``colonymind.yaml``, which the environment can pin), so a container recreated on the same volumes without
    ``COLONYMIND_INIT_PORT`` keeps the port an earlier start recorded; the health check asks there."""
    instance = tmp_path / "instance"
    instance.mkdir()
    if recorded is not None:
        (instance / "colonymind.yaml").write_text(yaml.safe_dump({"sidecar": recorded}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    asked = tmp_path / "asked.log"
    (bin_dir / "curl").write_text(f'#!/bin/sh\necho "$@" > "{asked}"\n')
    (bin_dir / "curl").chmod(0o755)
    check = tmp_path / "healthcheck.sh"
    check.write_text((DOCKER / "healthcheck.sh").read_text().replace("/opt/colonymind/.venv/bin/python", sys.executable))
    result = subprocess.run(["sh", str(check)], capture_output=True, text=True, timeout=60, check=False,
                            env={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "COLONYMIND_HOME": str(instance),
                                 **environment})
    assert result.returncode == 0, result.stdout + result.stderr
    assert asked.read_text().split()[-1] == f"{expected}/v1/host/health"


def test_every_script_the_image_installs_is_executable_and_posix_shell():
    installed = {"init.sh", "colonymind-shim.sh", "healthcheck.sh"}
    copied = " ".join(arguments for instruction, arguments in _instructions() if instruction == "COPY")
    for name in installed:
        assert f"docker/{name}" in copied, name
    for path in [*(DOCKER / name for name in installed), SERVICES / "colonymind-sidecar" / "run"]:
        assert os.access(path, os.X_OK), path
        first = path.read_text().splitlines()[0]
        assert first in {"#!/bin/sh", "#!/command/with-contenv sh"}, (path, first)


# --- the s6 services -----------------------------------------------------------------------------------------------

def test_setup_runs_after_hermes_boot_and_before_the_sidecar():
    init, sidecar = SERVICES / "colonymind-init", SERVICES / "colonymind-sidecar"
    assert (init / "type").read_text().strip() == "oneshot"
    assert (sidecar / "type").read_text().strip() == "longrun"
    # After s6's base bundle (Hermes' own boot step: the user remapped, its home owned and seeded) ...
    assert [path.name for path in (init / "dependencies.d").iterdir()] == ["base"]
    # ... and the sidecar only once setup succeeded: a failed setup leaves it down instead of serving an
    # unconfigured instance.
    assert [path.name for path in (sidecar / "dependencies.d").iterdir()] == ["colonymind-init"]
    assert [path.name for path in (SERVICES / "user" / "contents.d").iterdir()] == ["colonymind-sidecar"]
    assert (init / "up").read_text().split() == ["/opt/colonymind/docker/init"]
    copied = [arguments for instruction, arguments in _instructions() if instruction == "COPY"]
    assert any(arguments.endswith("docker/init.sh /opt/colonymind/docker/init") for arguments in copied)
    # The helper it runs with the sidecar's interpreter, where it looks for it.
    assert any(arguments.endswith("docker/endpoint.py /opt/colonymind/docker/endpoint.py") for arguments in copied)
    assert _IMAGE_PATHS["endpoint"] in SETUP.read_text()
    assert any(arguments.endswith("docker/s6-rc.d/ /etc/s6-overlay/s6-rc.d/") for arguments in copied)


def test_the_sidecar_service_runs_colonymind_start_through_the_shim():
    run = (SERVICES / "colonymind-sidecar" / "run").read_text()
    assert run.splitlines()[0] == "#!/command/with-contenv sh"
    assert re.search(r"^exec colonymind start$", run, re.M)
    # The instance the environment selects (the image's default, or one given with -e COLONYMIND_HOME).
    assert 'cd "${COLONYMIND_HOME:?' in run
    copied = [arguments for instruction, arguments in _instructions() if instruction == "COPY"]
    assert any(arguments.endswith("docker/colonymind-shim.sh /opt/colonymind/bin/colonymind") for arguments in copied)
    env = " ".join(arguments for instruction, arguments in _instructions() if instruction == "ENV")
    assert "PATH=/opt/colonymind/bin:" in env


# --- the setup step and the shim, with stand-ins for the container's commands ----------------------------------------

STAND_INS = {
    # s6-setuidgid USER PROG...: runs PROG as USER (here: records who it would run as).
    "s6-setuidgid": 'user="$1"; shift; FAKE_AS="$user" exec "$@"',
    "id": 'case "$*" in "-u") echo "${FAKE_UID:-0}";; "-u hermes") echo "${FAKE_HERMES_UID}";;'
          ' "-g hermes") echo "${FAKE_HERMES_GID}";; *) echo "unexpected id $*" >&2; exit 2;; esac',
    "chown": 'echo "chown $* as=${FAKE_AS:-root}" >> "$FAKE_LOG"',
    "colonymind": 'echo "colonymind $* as=${FAKE_AS:-root} home=$HOME cwd=$PWD" >> "$FAKE_LOG"; '
                 'echo "  endpoint=${COLONYMIND_INIT_MODEL_URL:-}|${COLONYMIND_INIT_MODEL:-}" >> "$FAKE_ENDPOINTS"; '
                 '[ "$1" != init ] || env -0 > "$FAKE_INIT_ENV"; '
                 'eval "code=\\${FAKE_EXIT_$1:-0}"; exit "$code"',
}
# hermes ...: records the call and, for `config set KEY VALUE` and `config unset KEY`, changes the config the way Hermes'
# own CLI does (its dotted key in $HERMES_HOME/config.yaml; unsetting a key that is not set fails).
HERMES_STAND_IN = """\
import os, sys, yaml
from pathlib import Path
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write("hermes " + " ".join(sys.argv[1:]) + " as=" + os.environ.get("FAKE_AS", "root") + "\\n")
code = int(os.environ.get("FAKE_HERMES_EXIT") or 0)
if code or sys.argv[1:2] != ["config"] or sys.argv[2:3] not in (["set"], ["unset"]):
    sys.exit(code)
path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
data = (yaml.safe_load(path.read_text()) if path.exists() else None) or {}
*parents, leaf = sys.argv[3].split(".")
node = data
for part in parents:
    node = node.setdefault(part, {})
if sys.argv[2] == "set":
    node[leaf] = sys.argv[4]
elif leaf in node:
    del node[leaf]
else:
    sys.exit("Config key not set: " + sys.argv[3])
path.write_text(yaml.safe_dump(data, sort_keys=False))
"""


@pytest.fixture
def container(tmp_path):
    """A stand-in for the container: fake s6-setuidgid, id, chown, hermes and colonymind on PATH, an instance
    directory and a log of every call."""
    if not shutil.which("sh") or not shutil.which("find"):
        pytest.skip("the setup step is a POSIX shell script")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in STAND_INS.items():
        (bin_dir / name).write_text("#!/bin/sh\n" + body + "\n")
        (bin_dir / name).chmod(0o755)
    (bin_dir / "hermes").write_text(f"#!{sys.executable}\n" + HERMES_STAND_IN)
    (bin_dir / "hermes").chmod(0o755)
    log = tmp_path / "calls.log"
    log.touch()
    endpoints = tmp_path / "endpoints.log"
    endpoints.touch()
    init_env = tmp_path / "init-env"
    home = tmp_path / "instance"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()

    class Container:
        instance = home
        hermes = hermes_home
        # The setup step with its image paths pointed at this interpreter and the repository's helper.
        setup = _setup_with(tmp_path)

        @staticmethod
        def run(script: str, *args: str, **env: str) -> subprocess.CompletedProcess:
            environ = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_LOG": str(log),
                       "FAKE_ENDPOINTS": str(endpoints), "FAKE_INIT_ENV": str(init_env),
                       "HERMES_HOME": str(hermes_home),
                       "COLONYMIND_HOME": str(home), "FAKE_HERMES_UID": str(os.getuid()),
                       "FAKE_HERMES_GID": str(os.getgid()), **env}
            return subprocess.run(["sh", str(script), *args], env=environ, capture_output=True, text=True,
                                  timeout=60, check=False)

        @staticmethod
        def calls() -> list[str]:
            return log.read_text().splitlines()

        @staticmethod
        def endpoints() -> list[str]:
            """The model endpoint each ``colonymind`` call was given (``COLONYMIND_INIT_MODEL_URL|_MODEL``)."""
            return [line.split("endpoint=", 1)[1] for line in endpoints.read_text().splitlines()]

        @staticmethod
        def init_environment() -> dict[str, str]:
            """The environment the setup step ran the last ``colonymind init`` with."""
            entries = init_env.read_bytes().decode().split("\0")
            return dict(entry.split("=", 1) for entry in entries if "=" in entry)

    return Container()


SETUP = DOCKER / "init.sh"


def test_a_fresh_instance_is_set_up_as_the_hermes_user(container):
    result = container.run(container.setup, COLONYMIND_INIT_OWNER_HANDLES="telegram=5550100")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = container.calls()
    assert calls == [f"colonymind init --non-interactive as=hermes home=/opt/data cwd={container.instance}"]
    assert stat.S_IMODE(container.instance.stat().st_mode) == 0o700


def test_an_existing_instance_is_upgraded_first_then_given_the_answers(container):
    container.instance.mkdir()
    (container.instance / "colonymind.yaml").write_text("{}\n")
    result = container.run(container.setup)
    assert result.returncode == 0, result.stdout + result.stderr
    verbs = [line.split()[1] for line in container.calls()]
    assert verbs == ["upgrade", "init"]


def test_a_failed_upgrade_stops_setup_and_so_the_sidecar(container):
    container.instance.mkdir()
    (container.instance / "colonymind.yaml").write_text("{}\n")
    result = container.run(container.setup, FAKE_EXIT_upgrade="1")
    assert result.returncode != 0
    assert [line.split()[1] for line in container.calls()] == ["upgrade"]


def test_a_failed_init_fails_setup(container):
    result = container.run(container.setup, FAKE_EXIT_init="1")
    assert result.returncode != 0


def test_the_instance_directory_is_given_to_the_hermes_user_only_when_it_is_not_already(container):
    assert container.run(container.setup).returncode == 0
    assert not [line for line in container.calls() if line.startswith("chown")]
    # A remapped hermes user (HERMES_UID) owns nothing yet: the whole directory is handed over, links not followed.
    remapped = str(os.getuid() + 4242)
    result = container.run(container.setup, FAKE_HERMES_UID=remapped, FAKE_HERMES_GID=remapped)
    assert result.returncode == 0, result.stdout + result.stderr
    chowns = [line for line in container.calls() if line.startswith("chown")]
    assert chowns == [f"chown -hR {remapped}:{remapped} {container.instance} as=root"]


def test_a_symlinked_instance_directory_is_refused(container, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    container.instance.symlink_to(elsewhere)
    result = container.run(container.setup)
    assert result.returncode != 0
    assert "symbolic link" in result.stdout + result.stderr
    assert container.calls() == []


def test_one_model_endpoint_configures_hermes_through_its_own_cli_before_init(container):
    result = container.run(container.setup, COLONYMIND_CONTAINER_MODEL_URL="http://model.example:8000/v1",
                           COLONYMIND_CONTAINER_MODEL="example-model", COLONYMIND_CONTAINER_MODEL_KEY="sk-example-0000")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = container.calls()
    assert calls[:5] == [
        "hermes config set model.provider custom as=hermes",
        "hermes config set model.base_url http://model.example:8000/v1 as=hermes",
        "hermes config set model.default example-model as=hermes",
        # The endpoint is OpenAI-compatible: Hermes speaks chat completions to it, whatever transport it had.
        "hermes config set model.api_mode chat_completions as=hermes",
        # The key stays in the environment: Hermes reads the reference, and neither a command line nor a file
        # carries the value.
        "hermes config set model.api_key ${COLONYMIND_CONTAINER_MODEL_KEY} as=hermes",
    ]
    assert calls[5].startswith("colonymind init --non-interactive as=hermes")
    assert "sk-example-0000" not in "\n".join(calls) + result.stdout + result.stderr


def test_init_is_given_the_same_endpoint_so_a_changed_one_is_followed(container):
    env = {"COLONYMIND_CONTAINER_MODEL_URL": "http://model.example:8000/v1", "COLONYMIND_CONTAINER_MODEL": "example-model"}
    assert container.run(container.setup, **env).returncode == 0
    container.instance.joinpath("colonymind.yaml").write_text("{}\n")
    moved = {**env, "COLONYMIND_CONTAINER_MODEL_URL": "http://other.example:9000/v1"}
    assert container.run(container.setup, **moved).returncode == 0
    assert container.endpoints() == ["http://model.example:8000/v1|example-model",
                                     "http://other.example:9000/v1|example-model",   # upgrade
                                     "http://other.example:9000/v1|example-model"]   # init
    # An explicit answer for ColonyMind still wins.
    explicit = container.run(container.setup, **env, COLONYMIND_INIT_MODEL_URL="http://explicit.example/v1")
    assert explicit.returncode == 0
    assert container.endpoints()[-1] == "http://explicit.example/v1|example-model"


#: Run by stock Hermes' interpreter with its home's .env loaded: the runtime Hermes resolves for its model, as a chat
#: request would use it. Nothing is sent anywhere.
_HERMES_RUNTIME = """
import json, sys
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home=sys.argv[1])
from hermes_cli.runtime_provider import resolve_runtime_provider
runtime = resolve_runtime_provider()
print(json.dumps({key: runtime.get(key) for key in ("provider", "api_mode", "base_url")}))
"""


def _hermes_runtime(container) -> dict[str, str]:
    """The provider, transport (``api_mode``) and base URL stock Hermes resolves from the container's Hermes home."""
    if not HERMES_PYTHON:
        pytest.skip("stock Hermes is not installed (COLONYMIND_TEST_HERMES_PYTHON names its interpreter)")
    result = subprocess.run([HERMES_PYTHON, "-c", _HERMES_RUNTIME, str(container.hermes)], capture_output=True,
                            text=True, timeout=120, check=False,
                            env={"PATH": os.environ["PATH"], "HOME": str(container.hermes),
                                 "HERMES_HOME": str(container.hermes)})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("earlier", [
    {"provider": "anthropic", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
    {"provider": "custom", "base_url": "http://model.example:8000/anthropic", "api_mode": "anthropic_messages"},
    {"provider": "bedrock", "api_mode": "bedrock_converse"},
], ids=["anthropic", "custom-messages", "bedrock"])
def test_hermes_speaks_chat_completions_to_the_containers_endpoint(container, earlier):
    """A reused Hermes home whose model had another transport: Hermes honors a persisted ``model.api_mode`` for a
    custom endpoint, so the setup sets the one the endpoint speaks. Asked of stock Hermes itself."""
    _hermes_model(container, default="claude-example", **earlier)
    result = container.run(container.setup, **ENDPOINT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hermes config set model.api_mode chat_completions as=hermes" in container.calls()
    assert _hermes_runtime(container) == {"provider": "custom", "api_mode": "chat_completions",
                                          "base_url": ENDPOINT["COLONYMIND_CONTAINER_MODEL_URL"]}


_IMAGE_PATHS = {"seed": "seed=/opt/hermes/cli-config.yaml.example", "python": "python=/opt/colonymind/.venv/bin/python",
                "endpoint": "endpoint=/opt/colonymind/docker/endpoint.py"}


def _setup_with(tmp_path: Path, **paths: str) -> Path:
    """The setup step with its image paths pointed at stand-ins: the sidecar's interpreter is this one and the
    endpoint helper the repository's (``docker/endpoint.py``); ``seed`` when given."""
    text = SETUP.read_text()
    for name, value in {"python": sys.executable, "endpoint": str(DOCKER / "endpoint.py"), **paths}.items():
        line = _IMAGE_PATHS[name]
        assert line in text
        text = text.replace(line, f"{name}={value}")
    script = tmp_path / "setup.sh"
    script.write_text(text)
    return script


def _with_seed(tmp_path: Path, seed: Path) -> Path:
    """The setup step with the path of the config Hermes seeds pointed at ``seed``."""
    return _setup_with(tmp_path, seed=str(seed))


def _router(container, **fields) -> Path:
    container.instance.mkdir(exist_ok=True)
    (container.instance / "colonymind.yaml").write_text("{}\n")
    path = container.instance / ".colonymind-llm-config.json"
    path.write_text(json.dumps({"provider": "local", "baseUrl": "http://model.example:8000/v1", "apiKey": "local-no-key",
                                "models": {"small": "example-model", "medium": "example-model",
                                           "large": "example-model"}, **fields}))
    return path


ENDPOINT = {"COLONYMIND_CONTAINER_MODEL_URL": "http://model.example:8000/v1", "COLONYMIND_CONTAINER_MODEL": "example-model"}


# A router file the owner tuned: every form of model entry the router supports, each with settings of its own, and
# settings beside the models.
TUNED = {"small": {"model": "example-model", "maxTokens": 4096},
         "medium": "openai/example-model",
         "large": {"model": "example-model", "usefulContextTokens": 65536, "extraBody": {"priority": 10},
                   "supportsVision": True}}


@pytest.mark.parametrize("models", [None, TUNED], ids=["strings", "tuned"])
def test_the_routers_own_file_is_kept_while_it_names_the_same_endpoint(container, models):
    router = _router(container, **({"models": models} if models else {}))
    before = router.read_text()
    assert container.run(container.setup, **ENDPOINT).returncode == 0
    # Given without /v1, the endpoint is the same one the router file records with it.
    result = container.run(container.setup, **{**ENDPOINT, "COLONYMIND_CONTAINER_MODEL_URL": "http://model.example:8000"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert router.read_text() == before and not (container.instance / "backups").exists()
    assert "router" not in result.stdout


@pytest.mark.parametrize("change", [{"COLONYMIND_CONTAINER_MODEL_URL": "http://other.example:9000/v1"},
                                    {"COLONYMIND_CONTAINER_MODEL": "other-model"},
                                    {"COLONYMIND_CONTAINER_MODEL_KEY": "sk-example-0001"}])
def test_a_changed_endpoint_points_the_routers_file_there_and_keeps_its_other_settings(container, monkeypatch, change):
    tuned = {**TUNED, "small": {**TUNED["small"], "baseUrl": "http://model.example:8000/v1", "apiKey": "sk-example-0000"}}
    router = _router(container, models=tuned, modelPool={"extra": {"model": "pool-model", "maxTokens": 2048}})
    before = json.loads(router.read_text())
    env = {**ENDPOINT, **change}
    result = container.run(container.setup, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    after = json.loads(router.read_text())
    url = env["COLONYMIND_CONTAINER_MODEL_URL"]
    model = env["COLONYMIND_CONTAINER_MODEL"]
    # The endpoint, its key and every tier's model are the container's ...
    assert after["baseUrl"] == url
    assert after["apiKey"] == env.get("COLONYMIND_CONTAINER_MODEL_KEY", "local-no-key")
    assert after["models"]["medium"] in {model, f"openai/{model}"}
    # ... and everything else stays: each tier's own settings (but a tier's own endpoint and key, since every tier is
    # served by the container's one endpoint), and what the file says beside the models.
    assert after["models"]["small"] == {"model": model, "maxTokens": 4096}
    assert after["models"]["large"] == {**TUNED["large"], "model": model}
    assert after["modelPool"] == before["modelPool"] and after["provider"] == before["provider"]
    # The version it replaced is kept, and init, which writes the file only when it is missing, keeps this one.
    [kept] = (container.instance / "backups").glob("router-*.json")
    assert json.loads(kept.read_text()) == before
    assert stat.S_IMODE(router.stat().st_mode) == 0o600 and stat.S_IMODE(kept.stat().st_mode) == 0o600
    assert "the router's file now names" in result.stdout
    assert "sk-example-0001" not in result.stdout + result.stderr
    assert container.calls()[-1].startswith("colonymind init --non-interactive")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    bearer = f"Bearer {env.get('COLONYMIND_CONTAINER_MODEL_KEY', 'local-no-key')}"
    assert _authorization(container)["router"] == {(url, bearer)}
    # Once pointed there, a restart changes nothing.
    again = container.run(container.setup, **env)
    assert again.returncode == 0 and json.loads(router.read_text()) == after
    assert len(list((container.instance / "backups").glob("router-*.json"))) == 1


def _routes(router: Path) -> dict[str, tuple[str, str, str]]:
    """The binding each function role the router's file names selects first, with the endpoint and key the router's
    routing snapshot pins for it (the file declares the hosts, so every binding is eligible)."""
    from colonymind.router.functions import build_snapshot, candidates
    from colonymind.router.tiers import build_tiers_from_host
    host = json.loads(router.read_text())
    snapshot = build_snapshot(host, build_tiers_from_host(host, configure_environment=False, discover=False))
    routes = {}
    for role in host["functionRoles"]:
        first = candidates(snapshot, role, {}, has_images=False, has_tools=False)[0]
        routes[role] = (first.name, first.config.base_url, first.config.api_key)
    return routes


@pytest.mark.parametrize("change", [{"COLONYMIND_CONTAINER_MODEL_URL": "http://other.example:9000/v1"}, {}],
                         ids=["moved", "same-server"])
def test_the_model_pools_bindings_keep_each_endpoint_with_its_key(container, change):
    """The model pool stays, and no binding in it pairs a server with a key given for another. A binding on a server
    of its own keeps that server and the key it was sent there, its own or the file's; a binding the file's endpoint
    serves follows it with the key given there, or with one of its own that was given for that same server."""
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model",
                  api_key="${COLONYMIND_CONTAINER_MODEL_KEY}")
    pool = {"elsewhere": {"model": "pool-model", "baseUrl": "http://pool.example:7000/v1"},
            "elsewhere-own-key": {"model": "pool-model", "baseUrl": "http://pool.example:7000/v1",
                                  "apiKey": "sk-example-0007"},
            "shared-own-key": {"model": "pool-model", "apiKey": "sk-example-0008"},
            "shared": {"model": "pool-model"}}
    roles = {"extraction": ["elsewhere"], "judging": ["elsewhere-own-key"], "planning": ["shared-own-key"],
             "coding": ["shared"]}
    router = _router(container, apiKey="sk-example-0000", modelPool=pool, functionRoles=roles,
                     localHosts=["model.example", "other.example", "pool.example"])
    before = _routes(router)
    assert before["extraction"] == ("elsewhere", "http://pool.example:7000/v1", "sk-example-0000")
    env = {**ENDPOINT, "COLONYMIND_CONTAINER_MODEL_KEY": "sk-example-0001", **change}
    result = container.run(container.setup, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    url = env["COLONYMIND_CONTAINER_MODEL_URL"]
    moved = url != ENDPOINT["COLONYMIND_CONTAINER_MODEL_URL"]
    after = _routes(router)
    assert after["extraction"] == before["extraction"]
    assert after["judging"] == before["judging"]
    assert after["coding"] == ("shared", url, "sk-example-0001")
    assert after["planning"] == ("shared-own-key", url, "sk-example-0001" if moved else "sk-example-0008")
    assert set(json.loads(router.read_text())["modelPool"]) == set(pool)
    assert "sk-example-0001" not in result.stdout + result.stderr
    # Once pointed there, a restart changes nothing.
    again = container.run(container.setup, **env)
    assert again.returncode == 0 and _routes(router) == after
    assert len(list((container.instance / "backups").glob("router-*.json"))) == 1


@pytest.mark.parametrize("text", ["{not json", "[]", '{"models": ["example-model"]}'],
                         ids=["not-json", "not-an-object", "models-not-an-object"])
def test_a_routers_file_setup_cannot_read_stops_setup_and_is_left_as_it_is(container, text):
    """What cannot be compared is not taken for a changed endpoint: the file is the owner's, so setup stops and says
    why instead of replacing it."""
    router = _router(container)
    router.write_text(text)
    result = container.run(container.setup, **ENDPOINT)
    assert result.returncode != 0
    assert str(router) in result.stdout + result.stderr and "move it away" in result.stdout + result.stderr
    assert router.read_text() == text and not (container.instance / "backups").exists()
    assert not [line for line in container.calls() if line.startswith("colonymind")]


def test_a_new_instance_waits_until_hermes_has_a_model(container, tmp_path):
    """The config Hermes seeds names a hosted placeholder no one chose; a new instance is not pointed at it."""
    seed = tmp_path / "cli-config.yaml.example"
    seed.write_text("model:\n  provider: auto\n  base_url: https://hosted.example/api/v1\n")
    (container.hermes / "config.yaml").write_text(seed.read_text())
    script = _with_seed(tmp_path, seed)
    result = container.run(script)
    assert result.returncode != 0
    assert "COLONYMIND_CONTAINER_MODEL_URL" in result.stdout and "hermes model" in result.stdout
    assert container.calls() == []
    # Once Hermes has a model of its own (its config changed), setup goes ahead and init follows Hermes.
    (container.hermes / "config.yaml").write_text("model:\n  provider: custom\n  base_url: http://local.example/v1\n")
    assert container.run(script).returncode == 0
    assert container.calls()[-1].startswith("colonymind init --non-interactive")


def test_an_existing_instance_or_an_explicit_endpoint_needs_no_hermes_model(container, tmp_path):
    seed = tmp_path / "cli-config.yaml.example"
    seed.write_text("model:\n  provider: auto\n")
    (container.hermes / "config.yaml").write_text(seed.read_text())
    script = _with_seed(tmp_path, seed)
    assert container.run(script, COLONYMIND_INIT_MODEL_URL="http://local.example/v1").returncode == 0
    container.instance.joinpath("colonymind.yaml").write_text("{}\n")
    assert container.run(script).returncode == 0


def _hermes_model(container, **model: str) -> None:
    """Hermes' config with the model block an earlier setup (or the owner, with ``hermes model``) left."""
    (container.hermes / "config.yaml").write_text(yaml.safe_dump({"model": model}))


def _init_key(container) -> str:
    """The key ``colonymind init`` selects for its model endpoint (the fixture's init is a stand-in), by init's own rule
    and in the environment the setup step ran it with: its override (``COLONYMIND_MODEL_API_KEY``), else the key Hermes'
    config resolves to (``model.api_key``, else ``OPENAI_API_KEY`` in Hermes' ``.env``). It is the key init writes into
    a router file it creates and sends its recall probe (``--embed-url auto``) with."""
    from colonymind.init import hermes_model_endpoint, read_hermes_config
    with mock.patch.dict(os.environ, container.init_environment(), clear=True):
        endpoint = hermes_model_endpoint(container.hermes, read_hermes_config(container.hermes / "config.yaml"))
        return os.environ.get("COLONYMIND_MODEL_API_KEY") or endpoint["api_key"]


def _init_writes_router(container) -> Path:
    """What ``colonymind init`` does with the router's own file, with init's own function: none there, it writes one for
    the endpoint the setup step told it, with the key it selects (``_init_key``)."""
    from colonymind.init import write_llm_config
    told = container.init_environment()
    write_llm_config(container.instance, base_url=told["COLONYMIND_INIT_MODEL_URL"], model=told["COLONYMIND_INIT_MODEL"],
                     api_key=_init_key(container))
    return container.instance / ".colonymind-llm-config.json"


def _authorization(container) -> dict[str, object]:
    """The Authorization header each side sends its model: Hermes from its ``model.api_key``, else its legacy
    ``model.api``, as Hermes reads them (the first that is set; a ``${NAME}`` in it read from ``os.environ``; no key, no
    header), and the router per tier, from the endpoint and key its routing snapshot pins for the tier (its client
    sends ``local-no-key`` when there is none)."""
    from colonymind.router.functions import build_snapshot
    from colonymind.router.tiers import build_tiers_from_host
    model = yaml.safe_load((container.hermes / "config.yaml").read_text())["model"]
    key = next((str(value).strip() for value in (model.get("api_key"), model.get("api"))
                if isinstance(value, str) and value.strip()), "")
    if key.startswith("${") and key.endswith("}"):
        key = os.environ.get(key[2:-1].removeprefix("env:"), key)
    host = json.loads((container.instance / ".colonymind-llm-config.json").read_text())
    snapshot = build_snapshot(host, build_tiers_from_host(host, configure_environment=False, discover=False))
    return {"hermes": (model.get("base_url"), f"Bearer {key}" if key else None),
            "router": {(binding.config.base_url, f"Bearer {binding.config.api_key or 'local-no-key'}")
                       for binding in snapshot.bindings.values()}}


MOVED = {"COLONYMIND_CONTAINER_MODEL_URL": "http://other.example:9000/v1", "COLONYMIND_CONTAINER_MODEL": "example-model"}


@pytest.mark.parametrize("previous", [
    # A key Hermes holds for the earlier endpoint: given as it is, or as a variable of its own ...
    {"provider": "custom", "base_url": "http://model.example:8000/v1", "api_key": "sk-example-0002"},
    {"provider": "custom", "base_url": "http://model.example:8000/v1", "api_key": "${EARLIER_MODEL_KEY}"},
    # ... the same host over another scheme or port is another endpoint ...
    {"provider": "custom", "base_url": "https://other.example:9000/v1", "api_key": "sk-example-0002"},
    # ... and so is a named provider's own endpoint.
    {"provider": "openrouter", "api_key": "sk-example-0002"},
])
def test_a_key_hermes_holds_for_another_endpoint_is_never_sent_to_a_new_one(container, monkeypatch, previous):
    _hermes_model(container, default="example-model", **previous)
    # The router's file init wrote for the earlier endpoint, with the key Hermes resolved for it.
    _router(container, apiKey="sk-example-0002")
    env = {**MOVED, "COLONYMIND_CONTAINER_MODEL_KEY": "", "EARLIER_MODEL_KEY": "sk-example-0002"}
    result = container.run(container.setup, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hermes config unset model.api_key as=hermes" in container.calls()
    assert "COLONYMIND_CONTAINER_MODEL_KEY" in result.stdout
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _init_writes_router(container)
    new = "http://other.example:9000/v1"
    assert _authorization(container) == {"hermes": (new, None), "router": {(new, "Bearer local-no-key")}}
    assert "sk-example-0002" not in result.stdout + result.stderr


@pytest.mark.parametrize("container_key", ["", "sk-example-0003"], ids=["no-key", "key-given"])
@pytest.mark.parametrize("previous", [
    {"provider": "custom", "base_url": "http://model.example:8000/v1", "api": "sk-example-0002"},
    {"provider": "custom", "base_url": "http://model.example:8000/v1", "api": "${EARLIER_MODEL_KEY}"},
    {"provider": "openrouter", "api": "sk-example-0002"},
    # Beside the current key name: Hermes reads model.api when model.api_key is empty or removed.
    {"provider": "custom", "base_url": "http://model.example:8000/v1", "api_key": "sk-example-0002",
     "api": "sk-example-0002"},
])
def test_a_legacy_key_hermes_holds_for_another_endpoint_is_never_sent_to_a_new_one(container, monkeypatch, previous,
                                                                                   container_key):
    """Hermes also reads its legacy key ``model.api`` (when ``model.api_key`` is empty) and clears both when it
    clears a credential. A key held there for the earlier endpoint is removed like ``model.api_key``, whether or not
    the variables give a key: one given only shadows it while ``model.api_key`` names the variable, and emptying the
    variable later would send the earlier key to the new server."""
    _hermes_model(container, default="example-model", **previous)
    _router(container, apiKey="sk-example-0002")
    env = {**MOVED, "COLONYMIND_CONTAINER_MODEL_KEY": container_key, "EARLIER_MODEL_KEY": "sk-example-0002"}
    result = container.run(container.setup, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hermes config unset model.api as=hermes" in container.calls()
    assert "api" not in yaml.safe_load((container.hermes / "config.yaml").read_text())["model"]
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _init_writes_router(container)
    new = "http://other.example:9000/v1"
    sent = f"Bearer {container_key}" if container_key else None
    assert _authorization(container)["hermes"] == (new, sent)
    assert "sk-example-0002" not in result.stdout + result.stderr
    if container_key:
        # The key given is emptied at a later start: nothing is left for Hermes to fall back on.
        env["COLONYMIND_CONTAINER_MODEL_KEY"] = ""
        monkeypatch.setenv("COLONYMIND_CONTAINER_MODEL_KEY", "")
        assert container.run(container.setup, **env).returncode == 0
        assert _authorization(container)["hermes"] == (new, None)


_HERMES_KEY = """
import json, sys
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home=sys.argv[1])
from hermes_cli.runtime_provider import resolve_runtime_provider
runtime = resolve_runtime_provider()
print(json.dumps({"base_url": runtime.get("base_url"), "earlier_key": runtime.get("api_key") == sys.argv[2]}))
"""


def test_stock_hermes_sends_no_legacy_key_to_the_new_endpoint(container):
    """Asked of stock Hermes itself: after the setup moved its model, the key it resolves for the new endpoint is not
    the one its legacy ``model.api`` held for the earlier one."""
    if not HERMES_PYTHON:
        pytest.skip("stock Hermes is not installed (COLONYMIND_TEST_HERMES_PYTHON names its interpreter)")
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model",
                  api="sk-example-0002")
    assert container.run(container.setup, **MOVED, COLONYMIND_CONTAINER_MODEL_KEY="").returncode == 0
    result = subprocess.run([HERMES_PYTHON, "-c", _HERMES_KEY, str(container.hermes), "sk-example-0002"],
                            capture_output=True, text=True, timeout=120, check=False,
                            env={"PATH": os.environ["PATH"], "HOME": str(container.hermes),
                                 "HERMES_HOME": str(container.hermes)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "base_url": MOVED["COLONYMIND_CONTAINER_MODEL_URL"], "earlier_key": False}


def test_a_legacy_key_hermes_holds_for_this_endpoint_stays(container):
    """A legacy key the owner gave Hermes for the same server is theirs for it: the path may change, the host does
    not, and nothing is removed."""
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model",
                  api="sk-example-0002")
    env = {**ENDPOINT, "COLONYMIND_CONTAINER_MODEL_URL": "http://model.example:8000", "COLONYMIND_CONTAINER_MODEL_KEY": ""}
    result = container.run(container.setup, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not [line for line in container.calls() if "unset" in line]
    assert _authorization_of_hermes(container) == ("http://model.example:8000", "Bearer sk-example-0002")


def _authorization_of_hermes(container) -> tuple:
    model = yaml.safe_load((container.hermes / "config.yaml").read_text())["model"]
    key = next((value for value in (model.get("api_key"), model.get("api")) if value), "")
    return model.get("base_url"), f"Bearer {key}" if key else None


#: A key in Hermes' ``.env``. Hermes itself sends ``OPENAI_API_KEY`` only to OpenAI's own hosts.
DOTENV_KEY = "sk-example-0005"


@pytest.mark.parametrize("instance", ["new", "existing"])
@pytest.mark.parametrize("dotenv", [f"OPENAI_API_KEY={DOTENV_KEY}\n",
                                    f"OPENAI_BASE_URL=http://model.example:8000/v1\nOPENAI_API_KEY={DOTENV_KEY}\n"],
                         ids=["key", "endpoint-and-key"])
def test_a_key_in_hermes_env_file_is_never_sent_to_the_containers_endpoint(container, monkeypatch, dotenv, instance):
    """A reused Hermes volume whose ``.env`` holds a key, alone or with the endpoint it was for. The variables name
    another endpoint and give no key, so the router is keyless there: on a new instance, where init writes the
    router's file and probes with the key it selects, and on an existing one, whose file the setup points there."""
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model")
    (container.hermes / ".env").write_text(dotenv)
    if instance == "existing":
        _router(container, apiKey=DOTENV_KEY)   # what init recorded for the earlier endpoint, from that .env
    env = {**MOVED, "COLONYMIND_CONTAINER_MODEL_KEY": ""}
    result = container.run(container.setup, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert _init_key(container) in {"", "local-no-key"}
    _init_writes_router(container)
    new = MOVED["COLONYMIND_CONTAINER_MODEL_URL"]
    assert _authorization(container) == {"hermes": (new, None), "router": {(new, "Bearer local-no-key")}}
    assert DOTENV_KEY not in result.stdout + result.stderr


def test_a_key_given_for_the_containers_endpoint_goes_to_it_alone(container, monkeypatch):
    """ColonyMind told another endpoint than Hermes' (an explicit ``COLONYMIND_INIT_MODEL_URL``) is not sent the key
    given for the container's."""
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model")
    elsewhere = "http://explicit.example:8100/v1"
    env = {**ENDPOINT, "COLONYMIND_CONTAINER_MODEL_KEY": "sk-example-0003", "COLONYMIND_INIT_MODEL_URL": elsewhere}
    result = container.run(container.setup, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert _init_key(container) in {"", "local-no-key"}
    _init_writes_router(container)
    assert _authorization(container) == {"hermes": (ENDPOINT["COLONYMIND_CONTAINER_MODEL_URL"], "Bearer sk-example-0003"),
                                         "router": {(elsewhere, "Bearer local-no-key")}}


def test_a_key_given_with_the_new_endpoint_is_the_one_both_send(container, monkeypatch):
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model",
                  api_key="sk-example-0002")
    _router(container, apiKey="sk-example-0002")
    env = {**MOVED, "COLONYMIND_CONTAINER_MODEL_KEY": "sk-example-0003"}
    assert container.run(container.setup, **env).returncode == 0
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _init_writes_router(container)
    new = "http://other.example:9000/v1"
    assert _authorization(container) == {"hermes": (new, "Bearer sk-example-0003"),
                                         "router": {(new, "Bearer sk-example-0003")}}


def test_without_a_key_hermes_keeps_its_own_while_the_endpoint_stays_on_its_host(container, monkeypatch):
    """A key the owner gave Hermes (``hermes model``) for this endpoint stays: the path may change, the host does not."""
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model",
                  api_key="sk-example-0002")
    env = {**ENDPOINT, "COLONYMIND_CONTAINER_MODEL_URL": "http://model.example:8000", "COLONYMIND_CONTAINER_MODEL_KEY": ""}
    result = container.run(container.setup, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not [line for line in container.calls() if "model.api_key" in line]
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _init_writes_router(container)
    assert _authorization(container) == {"hermes": ("http://model.example:8000", "Bearer sk-example-0002"),
                                         "router": {("http://model.example:8000/v1", "Bearer sk-example-0002")}}


@pytest.mark.parametrize("removed", [{"COLONYMIND_CONTAINER_MODEL_KEY": ""}, {}], ids=["empty", "unset"])
def test_a_removed_key_is_removed_from_hermes_and_the_router(container, monkeypatch, removed):
    """Started with a key, then with none: neither side keeps sending it (Hermes would otherwise send an empty key,
    or its ``${...}`` reference verbatim when the variable is gone, and the router the key it recorded)."""
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model",
                  api_key="${COLONYMIND_CONTAINER_MODEL_KEY}")
    router = _router(container, apiKey="sk-example-0000")   # what init recorded while the key was given
    result = container.run(container.setup, **ENDPOINT, **removed)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hermes config unset model.api_key as=hermes" in container.calls()
    assert "COLONYMIND_CONTAINER_MODEL_KEY" in result.stdout
    for name, value in {**ENDPOINT, **removed}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("COLONYMIND_CONTAINER_MODEL_KEY", raising=False)
    if router.exists():
        assert "sk-example-0000" not in router.read_text()
    _init_writes_router(container)
    assert _authorization(container) == {"hermes": (ENDPOINT["COLONYMIND_CONTAINER_MODEL_URL"], None),
                                         "router": {(ENDPOINT["COLONYMIND_CONTAINER_MODEL_URL"], "Bearer local-no-key")}}
    assert "sk-example-0000" not in result.stdout + result.stderr


def test_the_routers_file_is_kept_while_its_key_is_the_one_hermes_resolves(container, monkeypatch):
    """A key the owner gave Hermes for this endpoint is the one init records: a restart keeps that file as it is."""
    _hermes_model(container, provider="custom", base_url="http://model.example:8000/v1", default="example-model",
                  api_key="sk-example-0002")
    router = _router(container, apiKey="sk-example-0002")
    before = router.read_text()
    assert container.run(container.setup, **ENDPOINT).returncode == 0
    assert router.read_text() == before and not (container.instance / "backups").exists()
    # The router's own override (COLONYMIND_MODEL_API_KEY) is the key it records, and so the one compared.
    result = container.run(container.setup, **ENDPOINT, COLONYMIND_MODEL_API_KEY="sk-example-0004")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(router.read_text())["apiKey"] == "sk-example-0004"


def test_without_any_key_nothing_is_removed(container):
    result = container.run(container.setup, **ENDPOINT, COLONYMIND_CONTAINER_MODEL_KEY="")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not [line for line in container.calls() if "model.api_key" in line]


def test_a_model_endpoint_without_a_model_name_is_refused(container):
    result = container.run(container.setup, COLONYMIND_CONTAINER_MODEL_URL="http://model.example:8000/v1")
    assert result.returncode != 0
    assert "COLONYMIND_CONTAINER_MODEL" in result.stdout + result.stderr
    assert container.calls() == []


def test_no_model_endpoint_leaves_hermes_config_alone(container):
    assert container.run(container.setup).returncode == 0
    assert not [line for line in container.calls() if line.startswith("hermes")]


def test_a_fresh_instance_without_owner_handles_says_what_that_means(container):
    result = container.run(container.setup)
    assert result.returncode == 0
    assert "COLONYMIND_INIT_OWNER_HANDLES" in result.stdout
    container.instance.joinpath("colonymind.yaml").write_text("{}\n")
    again = container.run(container.setup)
    assert "COLONYMIND_INIT_OWNER_HANDLES" not in again.stdout
    handles = container.run(container.setup, COLONYMIND_INIT_OWNER_HANDLES="telegram=5550100")
    assert "COLONYMIND_INIT_OWNER_HANDLES" not in handles.stdout


def test_a_non_root_start_runs_without_switching_user(container):
    result = container.run(container.setup, FAKE_UID=str(os.getuid()))
    assert result.returncode == 0, result.stdout + result.stderr
    assert [line.split(" as=")[1].split()[0] for line in container.calls()] == ["root"]


def _shim(tmp_path: Path, container) -> Path:
    """The shim with its two absolute paths pointed at stand-ins."""
    real = tmp_path / "real-colonymind"
    real.write_text('#!/bin/sh\necho "real $* as=${FAKE_AS:-root} home=$HOME" >> "$FAKE_LOG"\n')
    real.chmod(0o755)
    suid = tmp_path / "bin" / "s6-setuidgid"
    text = (DOCKER / "colonymind-shim.sh").read_text()
    assert "REAL=/opt/colonymind/.venv/bin/colonymind" in text and "S6_SUID=/command/s6-setuidgid" in text
    shim = tmp_path / "shim.sh"
    shim.write_text(text.replace("REAL=/opt/colonymind/.venv/bin/colonymind", f"REAL={real}")
                    .replace("S6_SUID=/command/s6-setuidgid", f"S6_SUID={suid}"))
    return shim


def test_the_shim_runs_colonymind_as_the_hermes_user_when_called_as_root(container, tmp_path):
    shim = _shim(tmp_path, container)
    assert container.run(shim, "doctor", "--json").returncode == 0
    assert container.calls() == ["real doctor --json as=hermes home=/opt/data"]


def test_the_shim_runs_colonymind_directly_for_the_hermes_user(container, tmp_path):
    shim = _shim(tmp_path, container)
    assert container.run(shim, "status", FAKE_UID=str(os.getuid()), HOME="/elsewhere").returncode == 0
    assert container.calls() == ["real status as=root home=/elsewhere"]


# --- the compose example -------------------------------------------------------------------------------------------

def _compose() -> dict:
    return yaml.safe_load((DOCKER / "compose.yml").read_text())


def test_the_compose_example_runs_the_published_image_with_both_volumes():
    [service] = _compose()["services"].values()
    assert service["image"].startswith("ghcr.io/aevonix/colonymind:")
    targets = {volume.split(":")[1] for volume in service["volumes"]}
    assert targets == {HERMES_DATA, INSTANCE}
    assert service["command"] == ["gateway", "run"]
    assert "ports" not in service


def test_the_compose_example_requires_the_owners_handles_and_one_model_endpoint():
    [service] = _compose()["services"].values()
    environment = service["environment"]
    for required in ("COLONYMIND_INIT_OWNER_HANDLES", "COLONYMIND_CONTAINER_MODEL_URL", "COLONYMIND_CONTAINER_MODEL"):
        assert re.fullmatch(r"\$\{\w+:\?[^}]+\}", environment[required]), required
    # A key is optional (a local server needs none) and only ever comes from the caller's environment.
    assert re.fullmatch(r"\$\{\w+:-\}", environment["COLONYMIND_CONTAINER_MODEL_KEY"])


def test_the_compose_example_sets_only_names_the_image_reads():
    [service] = _compose()["services"].values()
    known = set(INIT_ENVIRONMENT.values()) | {"COLONYMIND_CONTAINER_MODEL_URL", "COLONYMIND_CONTAINER_MODEL",
                                              "COLONYMIND_CONTAINER_MODEL_KEY"}
    for name, value in service["environment"].items():
        if name.startswith("COLONYMIND_"):
            assert name in known, name
        # Every value is a variable reference: the example carries no literal answer, and no secret.
        assert str(value).startswith("${"), (name, value)
    setup = SETUP.read_text()
    for name in ("COLONYMIND_CONTAINER_MODEL_URL", "COLONYMIND_CONTAINER_MODEL", "COLONYMIND_CONTAINER_MODEL_KEY"):
        assert name in setup


# --- the workflows -------------------------------------------------------------------------------------------------

def test_ci_builds_the_quickstart_image_from_the_qualified_distributions_and_runs_the_smoke():
    job = _workflow("ci.yml")["jobs"]["docker"]
    assert job["needs"] == "artifacts"
    steps = job["steps"]
    [download] = [step for step in steps if str(step.get("uses", "")).startswith("actions/download-artifact@")]
    assert download["with"] == {"name": "release-distributions", "path": "dist/"}
    [build] = [step for step in steps if str(step.get("uses", "")).startswith("docker/build-push-action@")]
    assert build["with"]["file"] == "docker/Dockerfile" and build["with"]["context"] == "."
    assert build["with"]["load"] is True and build["with"]["push"] is False
    assert steps.index(download) < steps.index(build)
    [smoke] = [step for step in steps if "docker/smoke.sh" in str(step.get("run", ""))]
    assert steps.index(build) < steps.index(smoke)
    assert "tags" not in str(smoke.get("if", ""))


def test_the_smoke_checks_what_the_plan_asks_inside_the_container():
    smoke = (DOCKER / "smoke.sh").read_text()
    for needed in ("colonymind doctor", "colonymind demo", "outcome approved", "docker restart", "HERMES_UID"):
        assert needed in smoke, needed
    # Containers stop gracefully and nothing is removed by force.
    assert "--force" not in smoke and " -f " not in smoke


def test_the_release_publishes_the_image_the_docker_job_qualified():
    ci = _workflow("ci.yml")["jobs"]["docker"]["steps"]
    [keep] = [step for step in ci if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
    assert keep["if"] == "startsWith(github.ref, 'refs/tags/')"
    kept = keep["with"]["path"]
    assert "colonymind-image.tar" in kept and "colonymind-image-id.txt" in kept
    release = _workflow("release.yml")["jobs"]["docker"]
    script = "\n".join(str(step.get("run", "")) for step in release["steps"])
    assert "colonymind-image.tar" in script and "colonymind-image-id.txt" in script
    assert "ghcr.io/aevonix/colonymind" in script
    assert "sidecar-image" not in (WORKFLOWS / "release.yml").read_text() + (WORKFLOWS / "ci.yml").read_text()


def test_the_install_guide_has_the_container_section():
    guide = (ROOT / "docs" / "INSTALL.md").read_text()
    section = guide.split("\n## Container\n", 1)
    assert len(section) == 2, "docs/INSTALL.md needs a '## Container' section"
    body = section[1].split("\n## ", 1)[0]
    for needed in ("docker/compose.yml", "ghcr.io/aevonix/colonymind", "COLONYMIND_INIT_OWNER_HANDLES",
                   "COLONYMIND_CONTAINER_MODEL_URL", INSTANCE, "colonymind doctor"):
        assert needed in body, needed


def _shell_commands(text: str) -> list[str]:
    """The commands in a shell snippet, one per entry, continuation lines joined."""
    return [line.strip() for line in text.replace("\\\n", " ").splitlines() if line.strip()]


def _quickstarts() -> dict[str, list[str]]:
    """The quickstart commands the guide's Container section and the compose example's header give."""
    guide = (ROOT / "docs" / "INSTALL.md").read_text().split("\n## Container\n", 1)[1].split("\n## ", 1)[0]
    header = []
    for line in (DOCKER / "compose.yml").read_text().splitlines():
        if not line.startswith("#"):
            break
        header.append(line[1:].strip())
    return {"INSTALL.md": _shell_commands(re.search(r"```bash\n(.*?)```", guide, re.S).group(1)),
            "compose.yml": _shell_commands("\n".join(header).split("\n\n", 1)[1].split("\n\n", 1)[0])}


def test_the_quickstart_gives_every_compose_command_the_variables_it_requires():
    """Compose reads the whole file for every command (``exec`` and ``down`` as much as ``up``), and the example
    requires the owner's handles and the model endpoint (``${NAME:?...}``). Given as the prefix of one command, they
    are gone for the next one, which then fails before it starts."""
    required = set(re.findall(r"\$\{(\w+):\?", (DOCKER / "compose.yml").read_text()))
    assert required == {"OWNER_HANDLES", "MODEL_URL", "MODEL"}
    for where, commands in _quickstarts().items():
        compose = [index for index, command in enumerate(commands) if "docker compose" in command]
        assert compose, where
        for index in compose:
            assert commands[index].startswith("docker compose"), (where, commands[index])
        exported = {name for command in commands[:compose[0]] if command.startswith("export ")
                    for name in re.findall(r"(\w+)=", command.removeprefix("export "))}
        assert required <= exported, (where, commands)
