"""Registration/dispatcher fixtures whose HTTP calls stay in memory."""
import pytest

from conftest import PRELUDE, build_home, probe as hermes_probe


@pytest.fixture
def offline_home(tmp_path):
    return build_home(tmp_path, 1, "http://127.0.0.1:1")


MOCK_HTTP = '''
import httpx
_real_client = httpx.Client
def _offline_client(*args, **kwargs):
    kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(404, json={}))
    return _real_client(*args, **kwargs)
httpx.Client = _offline_client
'''


def probe(code, home, **kwargs):
    """Load the real plugins/dispatcher; replace only outbound HTTP transport."""
    return hermes_probe(code, home, prelude=MOCK_HTTP + PRELUDE, **kwargs)
