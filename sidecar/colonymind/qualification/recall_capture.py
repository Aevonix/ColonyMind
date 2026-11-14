"""Reranker capture shared by the recall benchmarks (``benchmarks/source_recall``, ``benchmarks/public_memory``).

Every call recall makes to the reranker is kept for offline re-scoring: the query, the documents and ``top_k`` it
was sent, and what came of it: the rows it returned, ``cancelled``, ``error`` (the exception's type only: an error
message or a response body can carry a secret) or, for a call recall stopped waiting for, ``abandoned``. Recall
never cancels a call on a missed deadline, so a call can still be running when its recall is done.

``RerankRecorder`` records around a rerank function; ``ProviderCapture`` stands in for a reranker provider inside a
served sidecar (``colonymind.api.routers.host.set_reranker``) and passes everything else, the calibration metadata
included, to the provider, so the calibration fingerprint is the provider's own. ``SelectionCapture`` is the
``selector`` a benchmark hands to ``colonymind.memory.search.select_memory``.
"""
import asyncio
from copy import deepcopy
import dataclasses
import os
import time

from colonymind.memory.recall import calibration_fingerprint
from colonymind.memory.selection import RecallSelector

ABANDONED = 'abandoned'


def plain(value):
    """A provider's rows as JSON values (a dataclass row becomes its fields)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return deepcopy(value)


def settled(observations):
    """The calls as a recall saw them when it finished: one still running is ``abandoned``."""
    return [dict(deepcopy(call), outcome=call.get('outcome', ABANDONED)) for call in observations]


class RerankRecorder:
    """Record every call to ``rerank_fn(query, documents, top_k=...)`` in ``observed``, and its timing in ``calls``
    (``{kind: rerank, ms, documents}``), returning what the function returns, unchanged."""

    def __init__(self, rerank_fn, calls=None):
        self.rerank_fn = rerank_fn
        self.calls = [] if calls is None else calls
        self.observed = []

    async def __call__(self, query, documents, top_k=10, **kwargs):
        observation = {'query': query, 'documents': list(documents), 'top_k': top_k}
        self.observed.append(observation)
        start = time.perf_counter()
        try:
            result = await self.rerank_fn(query, documents, top_k=top_k, **kwargs)
            observation.update(outcome='returned', results=plain(result))
            return result
        except asyncio.CancelledError:
            # A cancellation is a hold limit or shutdown, never a provider-reported timeout.
            observation['outcome'] = 'cancelled'
            raise
        except Exception as exc:
            # Error messages and HTTP response bodies can contain secrets.
            observation.update(outcome='error', error_type=type(exc).__name__)
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.calls.append({'kind': 'rerank', 'ms': elapsed, 'documents': len(documents)})

    def take(self):
        """The calls recorded since the last ``take``, settled, and a fresh record for the next recall."""
        taken, self.observed = settled(self.observed), []
        return taken


class ProviderCapture:
    """A reranker provider whose ``rerank`` calls are recorded (``recorder``); every other attribute is the
    provider's own."""

    def __init__(self, provider, calls=None):
        self._provider = provider
        self.recorder = RerankRecorder(provider.rerank, calls)

    async def rerank(self, query, documents, top_k=10, **kwargs):
        return await self.recorder(query, documents, top_k, **kwargs)

    def __getattr__(self, name):
        return getattr(self._provider, name)


class SelectionCapture:
    """Record this sequential benchmark's selector observations, not credentials.

    It is the ``selector`` the production ``select_memory`` calls: every request the
    reranker sees, its returned rows and the selected typed rows are kept for replay.
    """

    ENVIRONMENT_KEYS = ('COLONYMIND_RECALL_RERANK', 'COLONYMIND_RECALL_RERANK_MIN_SCORE',
                        'COLONYMIND_RECALL_RERANK_TIMEOUT_MS', 'COLONYMIND_RECALL_RERANK_CALIBRATION')
    CALIBRATION_KEYS = ('provider', 'model', 'prompt_style', 'format_version',
                        'weights_revision', 'embedding_identity', 'candidate_format',
                        'embedding_model', 'embedding_dimensions', 'index_generation')

    def __init__(self, rerank_fn, calibration, calls, *, ranking_format='grounded-quotation-bundles-v1'):
        self.rerank_fn, self.calibration, self.calls = rerank_fn, calibration, calls
        self.ranking_format = ranking_format
        self.recorder = RerankRecorder(rerank_fn, calls)
        self.last_replay = None
        self.selector = RecallSelector(self.rerank, calibration_metadata=lambda: calibration)

    @property
    def observed(self):
        return self.recorder.observed

    async def rerank(self, query, documents, top_k):
        return await self.recorder(query, documents, top_k)

    async def select_context(self, query, beliefs, quotations, *, limit=5, max_chars=6000,
                             current_work_available=False):
        """The selector contract ``select_memory`` uses; the replay is kept on ``last_replay``."""
        if self.ranking_format == 'verbose-claim-json':
            beliefs = [dict(row, ranking_text=row['content']) for row in beliefs]
        self.recorder.observed = []
        parameters = {'limit': limit, 'max_chars': max_chars, 'current_work_available': current_work_available}
        replay = {'version': 1, 'query': query, 'beliefs': deepcopy(beliefs),
                  'quotations': deepcopy(quotations), 'parameters': parameters,
                  'environment': {key: os.environ.get(key) for key in self.ENVIRONMENT_KEYS},
                  'calibration': {key: self.calibration[key] for key in self.CALIBRATION_KEYS
                                  if key in self.calibration},
                  'calibration_fingerprint': calibration_fingerprint(self.calibration)}
        selected, context = await self.selector.select_context(query, beliefs, quotations, **parameters)
        # A call recall stopped waiting for is still running: to this selection it was abandoned.
        replay.update(rerank_calls=self.recorder.take(), selected=deepcopy(selected))
        self.last_replay = replay
        return selected, context

    async def select(self, query, beliefs, quotations, *, limit=5, max_chars=6000):
        selected, context = await self.select_context(query, beliefs, quotations, limit=limit, max_chars=max_chars)
        return selected, context, self.last_replay


__all__ = ['ABANDONED', 'ProviderCapture', 'RerankRecorder', 'SelectionCapture', 'plain', 'settled']
