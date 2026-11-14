"""New claims record the extraction version that formed them: v19 adds subjects named in an earlier message."""
from contextlib import closing
import json

import pytest

from colonymind.beliefs.source_claims import EXTRACTION_VERSION
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.turns import TurnIdempotencyLedger
from test_source_claim_projection import Model, claim


@pytest.mark.asyncio
async def test_new_claims_and_their_job_record_v19(tmp_path):
    assert EXTRACTION_VERSION == "source-claims-v19"
    text = "My office is in River."
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("t", contact_id="contact-a", session_id="s", occurred_at="2026-03-01T09:00:00+00:00",
                         messages=[{"role": "user", "content": text}])
    assert await SourceClaimProjection(ledger).process_one(Model({text: claim(text, "River")}))
    with closing(ledger._connect()) as conn:
        data, = [json.loads(row[0]) for row in conn.execute("SELECT data_json FROM source_claims")]
        job, = conn.execute("SELECT extraction_version FROM source_claim_jobs").fetchall()
    assert data["extraction_version"] == job[0] == "source-claims-v19"
