"""Actual provider metadata, reordered arrival, restart and no inferred delivery."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import time

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host, transport
from colonymind.contacts.comms import CommsLog
from colonymind.commitments.store import CommitmentStore
from colonymind.initiatives.temporal_followup import TemporalFollowups
from colonymind.turns import TurnIdempotencyLedger, canonical_turn_digest
from onekey import KEY, _principal, _write_keyring


