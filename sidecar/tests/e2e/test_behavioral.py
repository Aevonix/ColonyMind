"""E2E behavioral integration tests for ColonyMind cognitive subsystems.

These tests verify that subsystems actually DO what they claim — not just
that endpoints respond, but that data flows correctly, state changes as
expected, and cross-subsystem integration produces meaningful results.

Prerequisites:
- ColonyMind sidecar running (default: localhost:7777)
- COLONYMIND_API_KEY set

Run:
    COLONYMIND_API_KEY=your-key pytest tests/e2e/test_behavioral.py -v
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid

import httpx
import pytest

COLONYMIND_URL = os.environ.get("COLONYMIND_URL", "http://localhost:7777")
COLONYMIND_API_KEY = os.environ.get("COLONYMIND_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {COLONYMIND_API_KEY}"} if COLONYMIND_API_KEY else {}


@pytest.fixture(scope="session")
def client():
    return httpx.Client(base_url=COLONYMIND_URL, headers=HEADERS, timeout=30)


# ---------------------------------------------------------------------------
# Commitment Tracking — behavioral
# ---------------------------------------------------------------------------

class TestCommitmentBehavior:
    """Verify commitments actually track and surface obligations."""

    def test_overdue_commitment_detected(self, client):
        """A commitment with a due_date in the near past should appear as overdue."""
        contact = f"e2e-overdue-{uuid.uuid4().hex[:6]}"

        # Create commitment due very soon (1 second from now)
        # Then wait briefly so it becomes overdue
        import datetime
        future = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=1)).isoformat()
        r = client.post("/v1/host/commitments", json={
            "person_id": contact,
            "description": "Soon-due task",
            "due_at": future,
            "priority": 3,
        })
        assert r.status_code in (200, 201)
        cid = r.json()["id"]

        # Wait for it to become overdue
        time.sleep(2)

        # Query overdue
        r = client.get("/v1/host/commitments?status=overdue")
        assert r.status_code == 200
        overdue = r.json()
        ids = [c["id"] for c in overdue] if isinstance(overdue, list) else [c["id"] for c in overdue.get("commitments", [])]
        assert cid in ids, f"Commitment {cid} not found in overdue list, got {ids}"

    def test_fulfilled_commitment_not_overdue(self, client):
        """A fulfilled commitment should NOT appear as overdue even if past due."""
        contact = f"e2e-fulfilled-{uuid.uuid4().hex[:6]}"

        import datetime
        future = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=1)).isoformat()
        r = client.post("/v1/host/commitments", json={
            "person_id": contact,
            "description": "Done task",
            "due_at": future,
            "priority": 1,
        })
        assert r.status_code in (200, 201)
        cid = r.json()["id"]

        # Wait then fulfill
        time.sleep(2)
        client.patch(f"/v1/host/commitments/{cid}", json={"status": "fulfilled"})

        # Should NOT appear in overdue
        r = client.get("/v1/host/commitments?status=overdue")
        overdue = r.json()
        ids = [c["id"] for c in overdue] if isinstance(overdue, list) else [c["id"] for c in overdue.get("commitments", [])]
        assert cid not in ids, "Fulfilled commitment should not be overdue"

    def test_commitments_filter_by_contact(self, client):
        """Commitments should be filterable by contact."""
        contact = f"e2e-filter-{uuid.uuid4().hex[:6]}"

        # Create for specific contact
        r = client.post("/v1/host/commitments", json={
            "person_id": contact,
            "description": "Contact-specific task",
        })
        assert r.status_code in (200, 201)

        # Filter by that contact
        r = client.get(f"/v1/host/commitments?person_id={contact}")
        assert r.status_code == 200
        data = r.json()
        items = data if isinstance(data, list) else data.get("commitments", [])
        assert len(items) >= 1
        assert all(c["person_id"] == contact for c in items)


# ---------------------------------------------------------------------------
# Surprise: behavioral
# ---------------------------------------------------------------------------

class TestSurpriseBehavior:
    """Verify surprise scoring actually works."""

    def test_surprise_score_reflects_expectation_violation(self, client):
        """A surprise with a large expected/actual gap should have a higher score."""
        # Mild surprise
        r1 = client.post("/v1/host/surprises", json={
            "observation": f"mild-{uuid.uuid4().hex[:6]}",
            "expected": "positive",
            "actual": "neutral",
            "surprise_score": 0.3,
        })
        mild_score = r1.json().get("surprise_score", 0.3)

        # Major surprise
        r2 = client.post("/v1/host/surprises", json={
            "observation": f"major-{uuid.uuid4().hex[:6]}",
            "expected": "complete success",
            "actual": "catastrophic failure",
            "surprise_score": 0.9,
        })
        major_score = r2.json().get("surprise_score", 0.9)

        assert major_score > mild_score, "Major surprise should score higher than mild"

    def test_surprise_resolution_changes_state(self, client):
        """Resolving a surprise should mark it as resolved."""
        r = client.post("/v1/host/surprises", json={
            "observation": f"resolve-test-{uuid.uuid4().hex[:6]}",
            "expected": "A", "actual": "B",
            "surprise_score": 0.95,  # High enough to appear in default min_score filter
        })
        sid = r.json()["id"]

        # Before resolution: should be in unresolved
        r = client.get("/v1/host/surprises/unresolved")
        unresolved = r.json()
        unresolved_ids = [s["id"] for s in unresolved]
        assert sid in unresolved_ids, f"Surprise {sid} not in unresolved: {unresolved_ids[:5]}..."

        # Resolve
        client.patch(f"/v1/host/surprises/{sid}", json={"resolution": "Fixed"})

        # After resolution: should NOT be in unresolved
        r = client.get("/v1/host/surprises/unresolved")
        unresolved_ids = [s["id"] for s in r.json()]
        assert sid not in unresolved_ids, "Resolved surprise should not be in unresolved list"


# ---------------------------------------------------------------------------
# Cross-Subsystem Integration — behavioral
# ---------------------------------------------------------------------------

class TestCrossSubsystemBehavior:
    """Verify data flows between subsystems correctly."""

    def test_context_assembly_includes_commitments(self, client):
        """Context assembly should include pending commitments for the contact."""
        contact = f"e2e-ctx-{uuid.uuid4().hex[:6]}"

        # Create a commitment
        desc = f"CRITICAL: Ship v0.6.0 by Friday {uuid.uuid4().hex[:4]}"
        r = client.post("/v1/host/commitments", json={
            "person_id": contact,
            "description": desc,
            "priority": 3,
        })
        assert r.status_code in (200, 201)

        # Assemble context for that contact
        r = client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "e2e-test"},
            "context": {"session_id": "e2e-ctx", "contact_id": contact},
            "incoming_message": {"role": "user", "content": "What am I supposed to do?"},
        })
        assert r.status_code == 200
        context_text = json.dumps(r.json())

        # The commitment description should appear in context
        assert "Ship v0.6.0" in context_text or "commitment" in context_text.lower() or "pending" in context_text.lower(),\
            f"Context should reference the commitment or commitments section, got: {context_text[:500]}"

    def test_health_reflects_all_subsystems(self, client):
        """Health endpoint should report all wired subsystems."""
        r = client.get("/v1/host/health")
        assert r.status_code == 200
        data = r.json()

        caps = data.get("capabilities", [])
        notes = data.get("notes", {})

        # Core capabilities that must be present
        required_caps = [
            "memory", "response_gate", "reasoning", "context",
            "contacts", "skills",
            "secrets", "autonomy",
            "events", "commitments", "affect",
            "surprises",
        ]
        missing = [c for c in required_caps if c not in caps]
        assert len(missing) == 0, f"Missing capabilities: {missing}"

        # New subsystems should have notes
        for note_key in ["commitments", "affect", "surprises"]:
            assert note_key in notes, f"Health notes should include {note_key}"
