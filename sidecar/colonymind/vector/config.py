"""ColonyMind Vector Store — configuration dataclasses.

All embedding and vector store configuration is driven by environment
variables.  No model names, device strings, or hardware assumptions
appear in source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from colonymind.removed_settings import reject_removed_arguments


@reject_removed_arguments({"quantization": "EmbeddingConfig.quantization"})
@dataclass
class EmbeddingConfig:
    """Resolved at runtime from env/settings.  Never hardcoded."""

    provider: str  # "cuda" | "cpu" | "mlx" | "native_mlx"
    model_id: str  # HuggingFace model ID
    dimensions: int  # output embedding dimension
    max_batch_size: int = 64
    device: Optional[str] = None  # auto-detected if None
    cache_dir: Optional[str] = None  # local model weights directory
    revision: Optional[str] = None  # operator-declared weights/deployment revision; not verification
    base_url: Optional[str] = field(default=None, repr=False)
    api_key: Optional[str] = field(default=None, repr=False)
    request_dimensions: Optional[int] = None  # explicit text API option; dimensions still validates output

    def __post_init__(self) -> None:
        if self.request_dimensions is None:
            return
        if type(self.request_dimensions) is not int or self.request_dimensions <= 0:
            raise ValueError("request_dimensions must be a positive integer")
        if self.provider != "openai_api":
            raise ValueError("request_dimensions requires the openai_api text embedding provider")
        if self.request_dimensions != self.dimensions:
            raise ValueError("request_dimensions must equal the configured output dimensions")

    @classmethod
    def from_env(cls) -> Optional[EmbeddingConfig]:
        """Build config from COLONYMIND_EMBED_* environment variables.

        Returns ``None`` if the required variables (MODEL, DIMS) are not set.
        """
        from colonymind.removed_settings import reject_removed_environment
        reject_removed_environment(os.environ)
        model_id = os.environ.get("COLONYMIND_EMBED_MODEL")
        dims_str = os.environ.get("COLONYMIND_EMBED_DIMS")
        if not model_id or not dims_str:
            return None
        return cls(
            provider=os.environ.get("COLONYMIND_EMBED_PROVIDER", "cpu"),
            model_id=model_id,
            dimensions=int(dims_str),
            max_batch_size=int(os.environ.get("COLONYMIND_EMBED_BATCH_SIZE", "64")),
            device=os.environ.get("COLONYMIND_EMBED_DEVICE") or None,
            cache_dir=os.environ.get("COLONYMIND_EMBED_CACHE_DIR") or None,
            revision=os.environ.get("COLONYMIND_EMBED_REVISION") or None,
            request_dimensions=(int(os.environ["COLONYMIND_EMBED_REQUEST_DIMS"])
                if os.environ.get("COLONYMIND_EMBED_REQUEST_DIMS") else None),
        )


@reject_removed_arguments({"cache_size": "VectorStoreConfig.cache_size", "batch_window_ms": "VectorStoreConfig.batch_window_ms"})
@dataclass
class VectorStoreConfig:
    """Configuration for the LanceDB vector store."""

    data_dir: str = ""  # path to Lance files directory

    @classmethod
    def from_env(cls) -> VectorStoreConfig:
        """Build from COLONYMIND_VECTOR_* environment variables."""
        from colonymind.removed_settings import reject_removed_environment
        reject_removed_environment(os.environ)
        colonymind_home = os.path.expanduser(
            os.environ.get("COLONYMIND_HOME", os.path.join(os.path.expanduser("~"), ".colonymind"))
        )
        return cls(
            data_dir=os.environ.get(
                "COLONYMIND_VECTOR_STORE_PATH",
                os.path.join(colonymind_home, "vector"),
            ),
        )
