"""ColonyMind Vector Store — embedding providers and pipeline.

Provides hardware-agnostic embedding via swappable backends:
  - CUDAEmbeddingProvider     (NVIDIA GPU via sentence-transformers)
  - CPUEmbeddingProvider      (CPU via sentence-transformers)
  - MLXEmbeddingProvider      (Apple Silicon via PyTorch MPS — legacy)
  - NativeMLXEmbeddingProvider (Apple Silicon via native MLX framework)

EmbeddingPipeline wraps a provider and adds LRU caching + latency monitoring.

Every provider's model calls leave a usage row each (``colonymind.usage``): a remote endpoint's per attempt
(``openai_provider``), a local model's per call (``EmbeddingProvider._local``), with its latency and no token count,
since a local model reports none.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from colonymind import usage

# A live recall query, its instruction prefix included, is kept within COLONYMIND_EMBED_QUERY_MAX_TOKENS of the
# embedding model's input, counted conservatively in UTF-8 bytes: a tokenizer emits at most one token per byte of its
# input (byte-level BPE builds every token from whole bytes; SentencePiece and WordPiece tokens cover at least one
# character, with byte fallback for the rest), plus a few markers of its own (QUERY_SPECIAL_TOKENS). A character
# count gives no such bound: one character can be four bytes and more than one token. A query past the model's
# context is refused (HTTP 400) and recall loses its semantic hits. The default of 4,096 tokens fits any endpoint that
# accepts that many and keeps about the first 4,000 characters of an ASCII query (the keyword search's own bound,
# ``recall.lexical_terms``); a deployment sets its endpoint's own limit.
QUERY_TOKENS_ENV = "COLONYMIND_EMBED_QUERY_MAX_TOKENS"
DEFAULT_QUERY_TOKENS = 4096
QUERY_SPECIAL_TOKENS = 16
BYTES_PER_TOKEN_FLOOR = 1
# Live queries carry changing context and rarely repeat: a small cache of their own, least recently used out first.
QUERY_CACHE_ENV = "COLONYMIND_EMBED_QUERY_CACHE_SIZE"
DEFAULT_QUERY_CACHE = 128


def _setting(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return min(max(value, low), high)


def bounded_query(instruction: str, query: str) -> str:
    """``instruction`` and as much of ``query`` as keeps them within the query token limit, counted in UTF-8 bytes
    (``BYTES_PER_TOKEN_FLOOR`` per token after ``QUERY_SPECIAL_TOKENS``). The cut never splits a character."""
    tokens = _setting(QUERY_TOKENS_ENV, DEFAULT_QUERY_TOKENS, 64, 1_000_000)
    room = (tokens - QUERY_SPECIAL_TOKENS) * BYTES_PER_TOKEN_FLOOR - len(instruction.encode("utf-8", "surrogatepass"))
    encoded = str(query).encode("utf-8", "surrogatepass")
    if len(encoded) <= room:
        return instruction + str(query)
    # A character cut in two leaves an incomplete sequence at the end, which is dropped.
    return instruction + encoded[:max(room, 0)].decode("utf-8", "ignore")

from colonymind.vector.config import EmbeddingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Abstract interface for all embedding backends."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.  Returns float32 vector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.  Returns list of float32 vectors."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a live query: a remote provider makes one attempt (the turn cannot wait for retries)."""
        with usage.embedding_task(usage.embedding_task_of("query_embedding")):
            return await self.embed(text)

    @staticmethod
    def _recorded(default_task: str) -> usage.Call:
        """A local model call's usage row: the task the caller named, else ``default_task``; no token count (a local
        model reports none), never a zero."""
        return usage.call("embedder", task=usage.embedding_task_of(default_task), role="embedding", endpoint="embed")

    async def _local(self, default_task: str, work, /, *args, **kwargs):
        """``work(*args, **kwargs)`` off the event loop, as one local model call (``_recorded``)."""
        loop = asyncio.get_running_loop()
        with self._recorded(default_task):
            return await loop.run_in_executor(None, functools.partial(work, *args, **kwargs))

    @abstractmethod
    async def warmup(self) -> None:
        """Load model weights, allocate device memory, run one dummy batch."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        ...


# ---------------------------------------------------------------------------
# CUDA provider
# ---------------------------------------------------------------------------

class CUDAEmbeddingProvider(EmbeddingProvider):
    """NVIDIA GPU embedding via sentence-transformers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model = None

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        # Warm up with a dummy encode
        await self._local("embedding_check", self._model.encode, ["warmup"])

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        device = self._config.device or "cuda"
        kwargs: dict = {"device": device}
        if self._config.cache_dir:
            kwargs["cache_folder"] = self._config.cache_dir
        model = SentenceTransformer(self._config.model_id, **kwargs)
        return model

    async def embed(self, text: str) -> list[float]:
        result = await self._local("text_embedding", self._model.encode, [text], normalize_embeddings=True)
        return result[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = await self._local("text_embedding", self._model.encode, texts,
                                   batch_size=self._config.max_batch_size, normalize_embeddings=True)
        return result.tolist()

    async def close(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# CPU provider
# ---------------------------------------------------------------------------

class CPUEmbeddingProvider(EmbeddingProvider):
    """CPU-only embedding via sentence-transformers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model = None

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        await self._local("embedding_check", self._model.encode, ["warmup"])

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        kwargs: dict = {"device": "cpu"}
        if self._config.cache_dir:
            kwargs["cache_folder"] = self._config.cache_dir
        model = SentenceTransformer(self._config.model_id, **kwargs)
        return model

    async def embed(self, text: str) -> list[float]:
        result = await self._local("text_embedding", self._model.encode, [text], normalize_embeddings=True)
        return result[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = await self._local("text_embedding", self._model.encode, texts,
                                   batch_size=self._config.max_batch_size, normalize_embeddings=True)
        return result.tolist()

    async def close(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# MLX provider (Apple Silicon)
# ---------------------------------------------------------------------------

class MLXEmbeddingProvider(EmbeddingProvider):
    """Apple Silicon embedding via sentence-transformers with PyTorch MPS backend.

    NOTE: This is the legacy provider. For true native MLX performance,
    use ``NativeMLXEmbeddingProvider`` (provider="native_mlx").
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model = None

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    async def warmup(self) -> None:
        # PyTorch MPS backend deadlocks when model initialization runs in
        # asyncio's ThreadPoolExecutor during FastAPI lifespan startup.
        # Load synchronously in the main thread — startup is not serving
        # requests yet, so brief blocking is acceptable.
        self._model = self._load_model()
        with self._recorded("embedding_check"):
            self._model.encode(["warmup"])

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        # sentence-transformers auto-detects MPS on Apple Silicon
        device = self._config.device or "mps"
        kwargs: dict = {"device": device}
        if self._config.cache_dir:
            kwargs["cache_folder"] = self._config.cache_dir
        model = SentenceTransformer(self._config.model_id, **kwargs)
        return model

    async def embed(self, text: str) -> list[float]:
        result = await self._local("text_embedding", self._model.encode, [text], normalize_embeddings=True)
        return result[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = await self._local("text_embedding", self._model.encode, texts,
                                   batch_size=self._config.max_batch_size, normalize_embeddings=True)
        return result.tolist()

    async def close(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Native MLX provider (Apple Silicon via true MLX framework)
# ---------------------------------------------------------------------------

class NativeMLXEmbeddingProvider(EmbeddingProvider):
    """Apple Silicon embedding via the native MLX framework (mlx-embeddings).

    Uses ``mlx_embeddings`` to load models directly into MLX arrays, avoiding
    the PyTorch MPS overhead. Supports both original HuggingFace models
    (converted on-the-fly) and pre-converted ``mlx-community`` checkpoints.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    async def warmup(self) -> None:
        # MLX model loading can trigger lazy weight evaluation; keep it
        # synchronous in the main thread just like the MPS provider.
        self._model, self._tokenizer = self._load_model()
        with self._recorded("embedding_check"):
            self._warmup_inference()

    def _load_model(self):
        from mlx_embeddings import load

        kwargs: dict = {"lazy": False}
        if self._config.cache_dir:
            kwargs["cache_dir"] = self._config.cache_dir
        model, tokenizer = load(self._config.model_id, **kwargs)
        return model, tokenizer

    def _warmup_inference(self):
        from mlx_embeddings import generate

        generate(self._model, self._tokenizer, "warmup", max_length=512)

    async def embed(self, text: str) -> list[float]:
        result = await self._local("text_embedding", self._infer, [text], max_length=512)
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._local("text_embedding", self._infer, texts, max_length=512)

    def _infer(self, texts, max_length: int = 512):
        from mlx_embeddings import generate

        outputs = generate(self._model, self._tokenizer, texts, max_length=max_length)
        # text_embeds is an mx.array of shape (batch, dims)
        return outputs.text_embeds.tolist()

    async def close(self) -> None:
        self._model = None
        self._tokenizer = None


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def make_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Instantiate the correct provider for the given config.

    Supported providers: cuda, cpu, mlx, native_mlx, openai_api.
    """
    from colonymind.vector.openai_provider import OpenAIAPIEmbeddingProvider

    providers = {
        "cuda": CUDAEmbeddingProvider,
        "cpu": CPUEmbeddingProvider,
        "mlx": MLXEmbeddingProvider,
        "native_mlx": NativeMLXEmbeddingProvider,
        "openai_api": OpenAIAPIEmbeddingProvider,
    }
    provider_cls = providers.get(config.provider)
    if provider_cls is None:
        if config.provider == "skip":
            logger.info("Embedding provider set to 'skip' — embeddings disabled")
            return None
        raise ValueError(
            f"Unknown embedding provider {config.provider!r}. "
            f"Choose from: {', '.join(providers)}, skip"
        )
    return provider_cls(config)


# ---------------------------------------------------------------------------
# Embedding pipeline (caching + latency monitoring)
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    """Wraps an EmbeddingProvider with caching and latency monitoring.

    Supports both text-only and multimodal embedding. When multimodal is
    enabled, the pipeline holds both a text provider and a multimodal provider.

    Live queries (``embed_query``) have a small LRU cache of their own and are cut
    below the model's context; single texts (``embed``) share the general cache;
    batches (background indexing, each text embedded once) are never cached.
    """

    LATENCY_WARN_MS = 500.0  # Raised for Qwen3-Embedding-8B on CUDA

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_size: int = 4096,
        multimodal_provider: Any = None,  # MultimodalEmbeddingProvider
        image_store: Any = None,  # LocalImageStore or EmbedOnlyStore
        query_transform: str = "identity",
    ) -> None:
        self._provider = provider
        self._multimodal_provider = multimodal_provider
        self._image_store = image_store
        self._cache: dict[str, list[float]] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._multimodal_enabled = multimodal_provider is not None
        self.query_instruction = os.environ.get('COLONYMIND_EMBED_QUERY_INSTRUCTION',
            'Instruct: Given a search query, retrieve relevant memories that answer it\nQuery: ')
        self._index_identity = None
        from colonymind.vector.indexes import query_transform_identity
        self._query_transform = query_transform_identity(query_transform)

    @property
    def is_multimodal(self) -> bool:
        """True when multimodal embedding is active."""
        return self._multimodal_enabled

    @property
    def modalities(self) -> list[str]:
        """Supported modalities."""
        if self._multimodal_enabled and self._multimodal_provider:
            return self._multimodal_provider.modalities
        return ["text"]

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    @property
    def query_transform(self) -> str:
        """Revision of the client query transform, for recall calibration and lesson bounds."""
        return self._query_transform

    @property
    def document_identity(self):
        return self.index_identity.document_identity

    @property
    def index_identity(self):
        from colonymind.vector.indexes import EmbeddingIdentity
        return self._index_identity or EmbeddingIdentity.from_pipeline(self)

    async def warmup(self) -> None:
        """Delegate warmup to the underlying provider. Its model calls are checks (``embedding_check``), never a
        memory lane's cost."""
        with usage.embedding_task("embedding_check"):
            await self._provider.warmup()
        from colonymind.vector.indexes import EmbeddingIdentity
        self._index_identity = EmbeddingIdentity.from_pipeline(self)

    def _check_identity(self):
        if self._index_identity is not None:
            from colonymind.vector.indexes import EmbeddingIdentity, IncompatibleIndex
            if EmbeddingIdentity.from_pipeline(self) != self._index_identity:
                raise IncompatibleIndex('Embedding configuration changed; use a new pipeline and index generation')

    async def embed_query(self, query: str) -> list[float]:
        """A live recall query, cut with its instruction to what COLONYMIND_EMBED_QUERY_MAX_TOKENS admits (default
        4,096, counted in UTF-8 bytes: ``bounded_query``), one attempt on a remote provider, and a small LRU cache
        of its own (COLONYMIND_EMBED_QUERY_CACHE_SIZE, default 128)."""
        self._check_identity()
        text = bounded_query(self.query_instruction, query)
        cached = self._query_cache.get(text)
        if cached is not None:
            self._query_cache.move_to_end(text)
            return cached
        started = time.monotonic()
        with usage.embedding_task(usage.embedding_task_of("query_embedding")):
            vector = await getattr(self._provider, "embed_query", self._provider.embed)(text)
        elapsed_ms = (time.monotonic() - started) * 1000
        if elapsed_ms > self.LATENCY_WARN_MS:
            logger.warning("Query embedding latency %.0fms exceeds %.0fms threshold", elapsed_ms, self.LATENCY_WARN_MS)
        size = _setting(QUERY_CACHE_ENV, DEFAULT_QUERY_CACHE, 0, 100_000)
        if size:
            self._query_cache[text] = vector
            self._query_cache.move_to_end(text)
            while len(self._query_cache) > size:
                self._query_cache.popitem(last=False)
        return vector

    async def health_check(self) -> dict[str, Any]:
        """Verify the embedder is loaded and producing valid output.

        Returns a dict with keys:
          provider, model, dims, latency_ms, status ("ok"|"error"),
          and optionally error (str) if status is "error".
        """
        import math

        model_id = ""
        provider_name = ""
        dims = 0
        if hasattr(self._provider, "_config"):
            model_id = self._provider._config.model_id
            provider_name = self._provider._config.provider
            dims = self._provider._config.dimensions

        try:
            if self._provider is None:
                return {"provider": provider_name, "model": model_id, "dims": dims, "latency_ms": 0, "status": "error", "error": "provider not initialized"}

            t0 = time.monotonic()
            with usage.embedding_task("embedding_check"):        # a check, never a memory lane's cost
                vector = await self._provider.embed("colonymind health check")
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Validate vector
            if not vector:
                return {"provider": provider_name, "model": model_id, "dims": dims, "latency_ms": elapsed_ms, "status": "error", "error": "empty vector returned"}
            if len(vector) != dims:
                return {"provider": provider_name, "model": model_id, "dims": dims, "latency_ms": elapsed_ms, "status": "error", "error": f"dimension mismatch: expected {dims}, got {len(vector)}"}
            if any(math.isnan(v) or math.isinf(v) for v in vector):
                return {"provider": provider_name, "model": model_id, "dims": dims, "latency_ms": elapsed_ms, "status": "error", "error": "vector contains NaN or Inf"}
            if all(v == 0.0 for v in vector):
                return {"provider": provider_name, "model": model_id, "dims": dims, "latency_ms": elapsed_ms, "status": "error", "error": "vector is all zeros"}

            return {"provider": provider_name, "model": model_id, "dims": dims, "latency_ms": round(elapsed_ms, 1), "status": "ok"}
        except Exception as exc:
            return {"provider": provider_name, "model": model_id, "dims": dims, "latency_ms": 0, "status": "error", "error": str(exc)}

    async def embed(self, text: str) -> list[float]:
        """Single embed with LRU caching and latency monitoring."""
        self._check_identity()
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        t0 = time.monotonic()
        vector = await self._provider.embed(text)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if elapsed_ms > self.LATENCY_WARN_MS:
            logger.warning(
                "Embedding latency %.0fms exceeds %.0fms threshold — "
                "check COLONYMIND_EMBED_PROVIDER and model config",
                elapsed_ms,
                self.LATENCY_WARN_MS,
            )

        self._put_cache(text, vector)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Explicit batch embed, never cached: a batch is indexing work, each text embedded once (keeping every
        background batch's vectors only held memory: no live query hits them)."""
        self._check_identity()
        if not texts:
            return []
        return await self._provider.embed_batch(list(texts))

    # ------------------------------------------------------------------
    # Multimodal methods
    # ------------------------------------------------------------------

    async def embed_image(self, source: str | bytes, mime_type: str = "", caption: str = "") -> tuple[list[float], dict[str, Any]]:
        """Embed a single image. Returns (vector, metadata).

        Parameters
        ----------
        source : str | bytes
            File path, URL, base64 string, or raw bytes.
        mime_type : str
            Optional MIME type hint.
        caption : str
            Optional user-provided caption.

        Returns
        -------
        tuple[list[float], dict]
            Vector and metadata dict with image_hash, image_ref, caption,
            thumbnail_ref, width, height, exif.
        """
        if not self._multimodal_enabled or not self._multimodal_provider:
            raise ValueError("Multimodal not enabled. Set COLONYMIND_MULTIMODAL=true and restart.")

        from colonymind.vector.image_preprocess import (
            compute_image_hash, extract_exif, load_image, resize_image,
            strip_gps_exif, validate_image,
        )
        from colonymind.vector.safety_image import check_image, ImageCheckLevel

        # Load image
        data, detected_mime = await load_image(source, mime_type)
        mime = detected_mime or "image/jpeg"

        # Validate
        errors = validate_image(data)
        if errors:
            raise ValueError(f"Invalid image: {'; '.join(errors)}")

        # Safety check
        check_level = ImageCheckLevel(os.environ.get("COLONYMIND_IMAGE_CHECK", "basic"))
        result = await check_image(data, mime, level=check_level)
        if not result.safe:
            raise ValueError(f"Image check failed: {result.reason}")

        # Resize if needed
        data, width, height = resize_image(data)

        # Strip GPS from stored image if configured
        strip_gps = os.environ.get("COLONYMIND_STRIP_EXIF_GPS", "true").lower() == "true"
        if strip_gps:
            strip_gps_exif(data)

        # Extract EXIF before stripping
        exif = extract_exif(data)

        # Compute hash
        img_hash = compute_image_hash(data)

        # Build ImageInput
        from colonymind.vector.multimodal_types import ImageInput
        image_input = ImageInput(
            data=data, mime_type=mime, width=width, height=height,
            original_path=source if isinstance(source, str) and not source.startswith("data:") else "",
            image_hash=img_hash, exif=exif, caption=caption,
        )

        # Store image
        image_ref = ""
        thumbnail_ref = ""
        if self._image_store:
            stored = await self._image_store.store(image_input)
            image_ref = stored.path
            thumbnail_ref = stored.thumbnail_path

        # Auto-caption if no caption provided
        if not caption:
            from colonymind.vector.caption import caption_image
            llm_cfg = getattr(self, "_llm_config", None)
            caption = await caption_image(image_input, llm_config=llm_cfg)

        # Embed
        vector = await self._multimodal_provider.embed_image(image_input)

        metadata = {
            "modality": "image",
            "image_hash": img_hash,
            "image_ref": image_ref,
            "thumbnail_ref": thumbnail_ref,
            "caption": caption,
            "width": width,
            "height": height,
            "model_id": self._multimodal_provider.model_id,
            "embedded_at": time.time(),
        }
        # Add EXIF data
        if exif.get("captured_at"):
            metadata["captured_at"] = exif["captured_at"]
        if exif.get("gps_lat") and exif.get("gps_lon"):
            metadata["gps_lat"] = exif["gps_lat"]
            metadata["gps_lon"] = exif["gps_lon"]

        return vector, metadata

    def set_llm_config(self, config: dict[str, Any]) -> None:
        """Set LLM config for auto-captioning."""
        self._llm_config = config

    async def close(self) -> None:
        """Release underlying provider resources."""
        await self._provider.close()
        if self._multimodal_provider:
            await self._multimodal_provider.close()
        self._cache.clear()
        self._cache_order.clear()
        self._query_cache.clear()

    def _put_cache(self, key: str, value: list[float]) -> None:
        if key in self._cache:
            return
        if len(self._cache_order) >= self._cache_size:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        self._cache[key] = value
        self._cache_order.append(key)
