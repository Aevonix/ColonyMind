"""The chat models the memory benchmark calls after recall: the reader, and the judge that grades it.

A model is named by what can be published about it (``ModelSpec``: its public name and its pinned weights
revision). Where it is served and its key are read from the process environment only
(``COLONYMIND_BENCH_<ROLE>_BASE_URL`` and ``COLONYMIND_BENCH_<ROLE>_API_KEY``, the role being ``READER`` or ``JUDGE``)
and are never written anywhere: no row, report or error message carries them. A failed call names its HTTP status
alone, since a response body can repeat a request's key.

Every call is one ``POST <base>/chat/completions`` (the OpenAI-compatible chat route) with the options the caller
passes; nothing is retried here. What a caller does with a call that did not complete is its own rule (the reader
and the judge leave its row unwritten, so a resumed run asks again, and never ask a completed call twice).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping, Optional

ROLES = ('READER', 'JUDGE')
# How long one call may take (seconds): a harness choice, generous for a long-context reader on a busy endpoint.
TIMEOUT_SECONDS = 600.0


class ModelError(RuntimeError):
    """A call that did not complete, or a client that cannot be configured. The message names no address or key."""


@dataclass(frozen=True)
class ModelSpec:
    """A model as a result publishes it: its public name and the weights revision it was pinned to."""
    name: str
    revision: str

    def __post_init__(self):
        if not str(self.name).strip() or not str(self.revision).strip():
            raise ValueError('A model is named by its public name and a pinned weights revision')

    def identity(self) -> dict:
        return {'name': self.name, 'revision': self.revision}

    def same_model(self, identity: Mapping[str, Any]) -> bool:
        """Whether ``identity`` (another spec's ``identity``) names this model, whatever revision it was pinned to."""
        return str(identity.get('name') or '').strip().casefold() == self.name.strip().casefold()


@dataclass(frozen=True)
class Reply:
    content: str
    model: Optional[str] = None                          # the model id the server reported
    usage: Mapping[str, Any] = field(default_factory=dict)


def spec_from_environment(role: str, *, name: Optional[str] = None, revision: Optional[str] = None,
                          environ: Optional[Mapping[str, str]] = None) -> ModelSpec:
    """The model named on the command line, else by ``COLONYMIND_BENCH_<ROLE>_MODEL`` and ``_REVISION``."""
    env = os.environ if environ is None else environ
    return ModelSpec(name or env.get(f'COLONYMIND_BENCH_{role}_MODEL', ''),
                     revision or env.get(f'COLONYMIND_BENCH_{role}_REVISION', ''))


class OpenAICompatible:
    """An OpenAI-compatible chat endpoint for one role, configured from the environment."""

    def __init__(self, role: str, spec: ModelSpec, *, environ: Optional[Mapping[str, str]] = None,
                 transport=None, timeout: float = TIMEOUT_SECONDS):
        if role not in ROLES:
            raise ModelError(f'A benchmark model role is one of {", ".join(ROLES)}')
        env = os.environ if environ is None else environ
        base = str(env.get(f'COLONYMIND_BENCH_{role}_BASE_URL') or '').rstrip('/')
        if not base:
            raise ModelError(f'COLONYMIND_BENCH_{role}_BASE_URL is not set')
        self.role, self.spec = role, spec
        self._url = base + '/chat/completions'
        self._key = str(env.get(f'COLONYMIND_BENCH_{role}_API_KEY') or '')
        self._transport, self._timeout = transport, timeout

    async def complete(self, messages, *, temperature: float, max_tokens: int, response_format=None) -> Reply:
        import httpx
        body = {'model': self.spec.name, 'messages': list(messages), 'temperature': temperature,
                'max_tokens': int(max_tokens)}
        if response_format is not None:
            body['response_format'] = response_format
        headers = {'Authorization': f'Bearer {self._key}'} if self._key else {}
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
                response = await client.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ModelError(f'{self.role.lower()} call failed ({type(exc).__name__})') from None
        if response.status_code != 200:
            raise ModelError(f'{self.role.lower()} call failed (status {response.status_code})')
        try:
            data = response.json()
            content = data['choices'][0]['message']['content']
        except (ValueError, KeyError, IndexError, TypeError):
            raise ModelError(f'{self.role.lower()} call returned no chat completion') from None
        return Reply(content if isinstance(content, str) else '', model=data.get('model'),
                     usage=dict(data.get('usage') or {}))
