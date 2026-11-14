"""Tests for colonymind.vector.reranker — active reranker providers."""
from secrets import token_hex

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from colonymind.vector.reranker import RerankResult, RerankerProvider, OpenAIAPIRerankerProvider


class TestRerankResult:
    def test_fields(self):
        r = RerankResult(index=0, score=0.95, text="hello")
        assert r.index == 0
        assert r.score == 0.95


class TestOpenAIAPIRerankerProvider:
    def test_configure(self):
        key = token_hex(24)
        provider = OpenAIAPIRerankerProvider("test-model")
        provider.configure("https://api.example.com", key)
        assert provider._base_url == "https://api.example.com"
        assert provider._api_key == key

    @pytest.mark.asyncio
    async def test_warmup_is_noop(self):
        provider = OpenAIAPIRerankerProvider("test-model")
        await provider.warmup()  # Should not raise
