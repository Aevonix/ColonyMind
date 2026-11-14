"""ColonyMind cognition server for agent runtimes.

A standalone FastAPI server that agent hosts (Hermes plugin, MCP harnesses,
REST integrations) mount via the ``/v1/host`` API surface.
"""

from pathlib import Path
import os

# Local first: LiteLLM, which routes model requests, downloads its model price list from the internet when it is
# imported unless told to use the copy it ships. ColonyMind never reads prices from it, so the flag is set here,
# before anything of ColonyMind can import the library, and starting the sidecar makes no network request. An
# operator who sets it explicitly (to false, for the downloaded list) keeps that choice.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

try:  # single source of truth: the installed package metadata (pyproject)
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("colonymind")
except Exception:  # editable/unbuilt checkouts without installed metadata
    __version__ = "0.0.0+unknown"


def get_state_dir() -> Path:
    """Return the selected state directory without moving existing state.

    The instance directory (``$COLONYMIND_HOME``, default ``~/.colonymind``)
    holds the state once ``colonymind init`` has written ``colonymind.yaml``.
    Unconfigured library use defaults to ``~/.colonymind/data``.

    Creates the directory if it does not exist.
    """
    explicit = os.environ.get("COLONYMIND_STATE_DIR") or os.environ.get("COLONYMIND_HOME")
    if explicit:
        path = Path(explicit).expanduser()
    elif (Path.home() / ".colonymind" / "colonymind.yaml").is_file():
        path = Path.home() / ".colonymind"
    else:
        path = Path.home() / ".colonymind" / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path
