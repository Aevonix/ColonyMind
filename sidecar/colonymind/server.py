"""ColonyMind sidecar FastAPI server.

Intelligence sidecar server mounted by agent frameworks (OpenClaw, Hermes,
etc.) as a plugin via the ``/v1/host`` API surface.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

# CLI and direct service starts use the same selected private instance.
from colonymind.util.instance import load_environment
from colonymind.util.temporal import now_utc
load_environment()

from fastapi import FastAPI

from colonymind.api.routers.host import (
    router as host_router,
    v2_router as host_v2_router,
    set_llm_router,
    set_embedder,
    set_reranker,
    set_contacts_store,
    set_secrets_vault,
    set_commitment_store,
    set_affect_store,
    set_comms_log,
    set_preference_learner,
    set_initiative_store,
    set_telemetry,
    supported_capabilities,
)

from colonymind import get_state_dir

logger = logging.getLogger(__name__)


def _state_dir() -> Path:
    """Resolve the ColonyMind state directory (wrapper for get_state_dir)."""
    return get_state_dir()


async def _initialize_contacts_store():
    """Open the canonical contact store without graph backfill or pruning.

    A merge moves the person's ledger sources, comms and affect with their
    handles whoever starts it: the people router passes these itself; the
    owner confirming a link (which folds the shadow contact that held the
    handle) goes through the store's defaults, set here. The comms log and
    affect store are opened before the contacts.
    """
    from colonymind.api.routers import people as people_router
    from colonymind.contacts.config import ContactsConfig
    from colonymind.contacts.store import SQLiteContactStore

    config = ContactsConfig.from_env()
    store = SQLiteContactStore(config=config, sources_of=people_router.person_sources,
                               reattribute=people_router.reattribute_hooks())
    await store.connect()
    set_contacts_store(store)
    logger.info("ContactsStore initialized (path=%s)", config.sqlite_path)
    return store


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize subsystems on startup, tear down on shutdown."""
    # Before any store opens: the vector store needs more open files than a user
    # process starts with on macOS, whether or not a service unit asked for them.
    from colonymind.resources import raise_open_file_limit
    raise_open_file_limit()
    state_dir = _state_dir()
    from colonymind.session_archive.store import ArchiveStore
    from colonymind.api.routers.host import set_session_archive_store, set_session_archive_engine
    from colonymind.api.archive_queue import ArchiveQueueClient
    from colonymind.util.temporal import now_utc
    archive_store = ArchiveStore(state_dir / 'colonymind-session-archive.db', clock=lambda: now_utc().timestamp())
    set_session_archive_store(archive_store)
    set_session_archive_engine(ArchiveQueueClient(archive_store))
    # Recover durable work admissions without blocking startup or any request.
    from colonymind.work import schedule_work
    await asyncio.to_thread(schedule_work)

    # --- 1. LLM Router ---
    llm_router = None
    try:
        from colonymind.router.router import LLMRouter
        from colonymind.router.tiers import build_tiers_from_host
        import json as _json

        config_path = state_dir / ".colonymind-llm-config.json"
        if config_path.exists():
            try:
                host_llm_config = _json.loads(config_path.read_text())
                llm_router = LLMRouter(tiers={})
                llm_router.configure(host_llm_config, config_path=config_path)
                logger.info(
                    "LLMRouter initialized from persisted host config (provider=%s)",
                    host_llm_config.get("provider", "unknown"),
                )
            except Exception as cfg_exc:
                logger.warning("Persisted LLM config unavailable: %s", type(cfg_exc).__name__)
                llm_router = LLMRouter(tiers={})
                llm_router.watch_config(config_path)
        else:
            llm_router = LLMRouter(tiers={})
            llm_router.watch_config(config_path)
            logger.info("LLMRouter awaiting explicit host config")
    except Exception as exc:
        logger.warning("LLMRouter init failed — model calls will not be available: %s", exc)

    if llm_router is not None:
        set_llm_router(llm_router)

    # --- 6. Embedding pipeline ---
    from colonymind.removed_settings import reject_removed_environment
    reject_removed_environment(os.environ)
    embed_provider = os.environ.get("COLONYMIND_EMBED_PROVIDER", "")
    embed_model = os.environ.get("COLONYMIND_EMBED_MODEL", "")
    embed_dims = os.environ.get("COLONYMIND_EMBED_DIMS", "")
    reranker_model = os.environ.get("COLONYMIND_RERANKER_MODEL", "")

    # Auto-detect tier if not explicitly configured
    if embed_provider != "skip" and (not embed_provider or not embed_model):
        try:
            from colonymind.vector.scanner import scan
            from colonymind.vector.tiers import get_tier_by_memory
            hw = scan()
            tier = get_tier_by_memory(hw.vram_gb, hw.ram_gb)
            spec = tier.text_embedder
            if spec:
                if hw.gpu_type == "cuda":
                    embed_provider = embed_provider or "cuda"
                elif hw.gpu_type == "mlx":
                    # Prefer native MLX when the package is available
                    try:
                        import mlx_embeddings  # noqa: F401
                        embed_provider = embed_provider or "native_mlx"
                    except ImportError:
                        embed_provider = embed_provider or "mlx"
                else:
                    embed_provider = embed_provider or "cpu"
                embed_model = embed_model or spec.model_id
                embed_dims = embed_dims or str(spec.dims)
                reranker_model = reranker_model or (tier.text_reranker.model_id if tier.text_reranker else "")
                logger.info(
                    "Auto-detected embedding tier: %s (GPU=%s %dGB, RAM=%dGB) -> %s",
                    tier.label, hw.gpu_name, hw.vram_gb, hw.ram_gb, spec.model_id,
                )
        except Exception as exc:
            logger.warning("Hardware scan failed, using defaults: %s", exc)
            embed_provider = embed_provider or "cpu"
            embed_model = embed_model or "sentence-transformers/all-MiniLM-L6-v2"
            embed_dims = embed_dims or "384"

    from colonymind.api.routers.host import set_embed_failure
    set_embed_failure(None)
    try:
        from colonymind.vector.embedder import EmbeddingPipeline
        from colonymind.vector.config import EmbeddingConfig
        request_dims = (int(os.environ["COLONYMIND_EMBED_REQUEST_DIMS"])
                        if os.environ.get("COLONYMIND_EMBED_REQUEST_DIMS") else None)
        if embed_dims:
            declared_dims = int(embed_dims)
        elif request_dims:
            declared_dims = request_dims
        elif embed_provider == "openai_api":
            declared_dims = 0   # learned from the endpoint's first embedding
        else:
            declared_dims = 384
        embed_config = EmbeddingConfig(
            provider=embed_provider,
            model_id=embed_model,
            dimensions=declared_dims,
            revision=os.environ.get("COLONYMIND_EMBED_REVISION") or None,
            request_dimensions=request_dims,
        )
        from colonymind.vector.embedder import make_provider
        provider = make_provider(embed_config)
        if provider is not None and embed_provider == "openai_api" and hasattr(provider, "configure"):
            # openai_api needs explicit endpoint config (the text-only path
            # never called configure(); only the multimodal branch passed these).
            provider.configure(
                os.environ.get("COLONYMIND_EMBED_BASE_URL", ""),
                os.environ.get("COLONYMIND_EMBED_API_KEY", ""),
            )
        if provider is None:
            # 'skip' provider — embeddings disabled entirely
            logger.info("EmbeddingPipeline skipped (provider=skip) — embeddings disabled")
        else:
            pipeline = EmbeddingPipeline(provider)

            # Wire up multimodal if enabled
            multimodal_enabled = os.environ.get("COLONYMIND_MULTIMODAL", "false").lower() == "true"
            if multimodal_enabled:
                try:
                    from colonymind.vector.multimodal_provider import make_multimodal_provider
                    from colonymind.vector.image_store import make_image_store

                    mm_config = EmbeddingConfig(
                        provider=embed_provider,
                        model_id=embed_model,
                        dimensions=int(embed_dims) if embed_dims else 1024,
                        base_url=os.environ.get("COLONYMIND_EMBED_BASE_URL"),
                        api_key=os.environ.get("COLONYMIND_EMBED_API_KEY"),
                        request_dimensions=embed_config.request_dimensions,
                    )
                    mm_provider = make_multimodal_provider(mm_config)
                    img_store = make_image_store(
                        mode=os.environ.get("COLONYMIND_IMAGE_STORAGE", "local"),
                        state_dir=os.environ.get("COLONYMIND_STATE_DIR", "."),
                    )
                    pipeline = EmbeddingPipeline(
                        provider=provider,
                        multimodal_provider=mm_provider,
                        image_store=img_store,
                    )
                    logger.info("Multimodal enabled (model=%s, storage=%s)", embed_model, os.environ.get("COLONYMIND_IMAGE_STORAGE", "local"))
                except Exception as exc:
                    logger.warning("Multimodal init failed, falling back to text-only: %s", exc)

            await pipeline.warmup()
            set_embedder(pipeline)
            logger.info("EmbeddingPipeline initialized (provider=%s model=%s)", embed_provider, embed_model)

            # Open the vector store the source projections and semantic recall share
            try:
                from colonymind.vector.store import VectorStore
                from colonymind.vector.indexes import IndexCatalog
                from colonymind.turns import get_turn_idempotency_ledger
                from colonymind.vector import set_store, set_pipeline
                vector_db_path = os.path.join(state_dir, "lancedb")
                vs = VectorStore(data_dir=vector_db_path, identity=pipeline.index_identity,
                    catalog=IndexCatalog(get_turn_idempotency_ledger(state_dir)))
                embed_dims = int(os.environ.get("COLONYMIND_EMBED_DIMS") or pipeline.dimensions or 384)
                await vs.connect(dimensions=embed_dims)
                await vs.ensure_collections(dimensions=embed_dims)
                set_store(vs)
                set_pipeline(pipeline)
                # Old versions go past a threshold and each night, never inside a request.
                vs.compaction.start()
                logger.info("Vector store wired for semantic recall (path=%s)", vector_db_path)
            except Exception as vexc:
                logger.warning("Vector store wiring failed (recall will use keyword fallback): %s", vexc)
                set_embed_failure(f"the vector store did not open: {type(vexc).__name__}: {vexc}")

            # Pass LLM config to pipeline for auto-captioning
            llm_config_path = Path(os.environ.get("COLONYMIND_STATE_DIR", ".")) / ".colonymind-llm-config.json"
            if llm_config_path.exists() and hasattr(pipeline, "set_llm_config"):
                try:
                    llm_cfg = _json.loads(llm_config_path.read_text())
                    pipeline.set_llm_config(llm_cfg)
                    logger.info("LLM config passed to EmbeddingPipeline for auto-captioning")
                except Exception as exc:
                    logger.debug("Could not pass LLM config to pipeline: %s", exc)

            # Health check + model mismatch detection
            try:
                hc = await pipeline.health_check()
                if hc.get("status") != "ok":
                    logger.warning("Embedder health check failed: %s", hc.get("error", "unknown"))
                else:
                    logger.info("Embedder health check passed (latency=%.1fms)", hc.get("latency_ms", 0))
            except Exception as exc:
                logger.warning("Embedder health check exception: %s", exc)
    except Exception as exc:
        logger.warning("EmbeddingPipeline init failed: %s", exc)
        # The sidecar still serves, so nothing else would say it: health carries the
        # reason in words and stays degraded until the embedder comes up.
        set_embed_failure(f"the embedder (provider={embed_provider}, model={embed_model or 'unset'}) "
                          f"did not initialise: {type(exc).__name__}: {exc}")

    # --- 6b. Reranker pipeline ---
    reranker_provider_name = os.environ.get("COLONYMIND_RERANKER_PROVIDER", "")
    if reranker_model and reranker_model.lower() not in ("none", "", "null"):
        try:
            from colonymind.vector.reranker import (
                OpenAIAPIRerankerProvider,
                NativeMLXRerankerProvider,
                MLXRerankerProvider,
                CPURerankerProvider,
                CUDARerankerProvider,
            )
            reranker_base_url = os.environ.get("COLONYMIND_RERANKER_BASE_URL", "")
            reranker_api_key = os.environ.get("COLONYMIND_RERANKER_API_KEY", "")
            if reranker_provider_name == "openai_api" or reranker_base_url:
                # Remote reranker over an OpenAI/Jina-compatible /v1/rerank
                # endpoint, mirroring the embedder's openai_api path so the
                # model stays off-box instead of loading in-process.
                # COLONYMIND_RERANKER_PROMPT_STYLE=qwen3 applies the Qwen3-Reranker
                # instruction template, without which its scores are noise.
                reranker_provider = OpenAIAPIRerankerProvider(reranker_model)
                reranker_provider.configure(
                    reranker_base_url,
                    reranker_api_key,
                    os.environ.get("COLONYMIND_RERANKER_PROMPT_STYLE", ""),
                )
            else:
                from colonymind.vector.scanner import scan
                hw = scan()
                if hw.gpu_type == "mlx":
                    # Prefer native MLX when the package is available
                    try:
                        import mlx_lm  # noqa: F401
                        reranker_provider = NativeMLXRerankerProvider(reranker_model)
                    except ImportError:
                        reranker_provider = MLXRerankerProvider(reranker_model)
                elif hw.gpu_type == "cuda":
                    reranker_provider = CUDARerankerProvider(reranker_model)
                else:
                    reranker_provider = CPURerankerProvider(reranker_model)
            await reranker_provider.warmup()
            set_reranker(reranker_provider)
            logger.info(
                "Reranker initialized (provider=%s model=%s)",
                reranker_provider_name or "local", reranker_model,
            )
        except Exception as exc:
            logger.warning("Reranker init failed: %s", exc)
    else:
        logger.info("No reranker configured for this tier")

    # --- 7b. Commitment Store ---
    try:
        from colonymind.commitments.store import CommitmentStore

        commitments_db = state_dir / "colonymind-commitments.db"
        commitment_store = CommitmentStore(db_path=commitments_db)
        commitment_readiness = (
            commitment_store.resolution_recovery_readiness()
        )
        if commitment_readiness.get("ready") is not True:
            raise RuntimeError(
                "CommitmentStore resolution recovery is unavailable"
            )

        set_commitment_store(commitment_store)
        logger.info(
            "CommitmentStore initialized (db=%s, capability=%s)",
            commitments_db, commitment_readiness["capability"],
        )
    except Exception as exc:
        set_commitment_store(None)
        logger.error("CommitmentStore initialization failed")
        raise RuntimeError("CommitmentStore initialization failed") from exc

    # --- 7c. The contacts' affect, the comms ledger and the owner's preferences ---
    try:
        from colonymind.contacts.affect import AffectStore

        from colonymind.turns import get_turn_idempotency_ledger
        source_ledger = get_turn_idempotency_ledger(state_dir)
        affect_db = state_dir / "colonymind-affect.db"
        affect_store = AffectStore(db_path=affect_db, source_ledger=source_ledger)
        affect_store.purge_erased_sources()
        set_affect_store(affect_store)
        logger.info("AffectStore initialized (db=%s)", affect_db)

        from colonymind.contacts.comms import CommsLog
        comms_log = CommsLog(db_path=state_dir / "colonymind-comms.db", source_ledger=source_ledger)
        comms_log.purge_erased_sources()
        set_comms_log(comms_log)
        logger.info("CommsLog initialized")

        # Owner preference learner — captures the owner's *explicit* directives
        # about how to communicate ("be concise", "use bullets", "no emoji") at
        # high confidence.
        from colonymind.intelligence.components.preference_learner import PreferenceLearner
        from colonymind.self_model.perspective import SelfPerspective
        from colonymind.turns import get_turn_idempotency_ledger
        from colonymind.identity import get_owner_contact_id
        preference_learner = PreferenceLearner(db_path=str(state_dir / "colonymind-preferences.db"),
            perspective=SelfPerspective(get_turn_idempotency_ledger(state_dir), owner_id=get_owner_contact_id()))
        set_preference_learner(preference_learner)
        logger.info("PreferenceLearner initialized (db=%s)", state_dir / "colonymind-preferences.db")
    except Exception as exc:
        logger.warning("Contacts' affect, comms ledger or preference learner init failed: %s", exc)

    # --- Type feedback store (outcome-driven priority decay/boost) ---
    try:
        from colonymind.feedback import TypeFeedbackStore
        from colonymind.api.routers.host import set_feedback_store
        set_feedback_store(TypeFeedbackStore(db_path=str(state_dir / "colonymind-feedback.db")))
        logger.info("TypeFeedbackStore initialized (db=%s)", state_dir / "colonymind-feedback.db")
    except Exception as exc:
        logger.warning("TypeFeedbackStore init failed: %s", exc)

    # --- The action journal and its competence store ---
    _action_journal = None
    try:
        from colonymind.api.routers.host import set_self_model
        from colonymind.self_model.journal import ActionJournal
        from colonymind.self_model.store import CompetenceStore, SelfModel
        _action_journal = ActionJournal(db_path=str(state_dir / "colonymind-action-journal.db"))
        _self_model = SelfModel(CompetenceStore(db_path=str(state_dir / "colonymind-self-model.db")),
                                journal=_action_journal)
        _self_model.perspective = getattr(locals().get("preference_learner"), "perspective", None)
        set_self_model(_self_model)
        logger.info("SelfModel and action journal ready (%s)", state_dir)
    except Exception as exc:
        logger.warning("SelfModel or action journal init failed: %s", exc)

    # --- Expectation engine (Mind M3a): predictions + surprise + calibration ---
    try:
        from colonymind.self_model.expectations import (
            ExpectationEngine, ExpectationStore, expectations_enabled,
            expectations_mode,
        )
        from colonymind.api.routers.host import set_expectations
        if expectations_enabled():
            _exp_store = ExpectationStore(
                db_path=str(state_dir / "colonymind-expectations.db"))
            # workspace wired later; set_expectations stores the engine and the
            # autonomy phase links the workspace ref at runtime.
            _expectations = ExpectationEngine(_exp_store, journal=_action_journal)
            set_expectations(_expectations)
            logger.info("Expectation engine ready (mode=%s, db=%s)",
                        expectations_mode(),
                        state_dir / "colonymind-expectations.db")
        else:
            logger.info("Expectation engine disabled (COLONYMIND_EXPECTATIONS=off)")
    except Exception as exc:
        logger.warning("Expectation engine init failed: %s", exc)

    # --- 8. Contacts ---
    contacts_store = None
    try:
        contacts_store = await _initialize_contacts_store()
    except Exception as exc:
        logger.warning("ContactsStore init failed: %s", exc)

    # --- 18. Secrets: the sealed store (docs/SECRETS.md) ---
    # Turn capture seals detected secrets into it; the first start seals what was stored before (once).
    try:
        from colonymind.api.routers.secrets import set_vault as set_secrets_route_vault
        from colonymind.secrets.config import current_settings as secrets_settings
        from colonymind.secrets.migrate import migrate_on_start
        from colonymind.secrets.vault import get_vault
        secrets_vault = get_vault(state_dir, key_store=secrets_settings().key_store)
        set_secrets_vault(secrets_vault)
        set_secrets_route_vault(secrets_vault)
        logger.info("sealed store initialized (%s)",
                    "key available" if secrets_vault.available else "key unavailable: detected values are withheld")
        try:
            from colonymind.config import load_config as _secrets_config
            _hermes_home = _secrets_config().hermes_home
        except Exception:
            _hermes_home = None
        migrate_on_start(state_dir, hermes_home=_hermes_home)
    except Exception as exc:
        logger.warning("sealed store init failed: %s", exc)

    # --- 20. The mind's intention store ---
    try:
        from colonymind.initiatives.store import InitiativeStore
        initiative_store = InitiativeStore(state_dir=state_dir)
        set_initiative_store(initiative_store)
        logger.info("InitiativeStore initialized (state_dir=%s)", state_dir)
    except Exception as exc:
        logger.warning("InitiativeStore init failed: %s", exc)

    # Temporal telemetry: the mind's tick beats into it, turns/sync and context/assemble
    # touch it, and /v1/host/health reads it. Built before the mind so the tick can report.
    from colonymind.telemetry import TelemetryStore
    telemetry = TelemetryStore()
    telemetry.load()  # restore last_*_at across restart (v0.21.0)
    telemetry.started_at = now_utc()
    app.state.telemetry = telemetry
    set_telemetry(telemetry)
    logger.info("TelemetryStore initialized")

    # --- 21. The mind tick (architecture 3.2) ---
    # Replaces the autonomy loop and its scheduler: health_check became the
    # tick's upkeep probes, digest_flush became the outbox digest, and the
    # other scheduled jobs went with their modules.
    mind = None
    try:
        from colonymind.mind.factory import build_mind
        from colonymind.api.routers.mind import set_mind
        from colonymind.api.routers import host as _host_for_mind
        from colonymind.config import load_config, load_identity, update_config
        from colonymind.identity import get_owner_contact_id
        from colonymind.turns import get_turn_idempotency_ledger

        _mind_store = locals().get("initiative_store")
        if _mind_store is None:
            raise RuntimeError("the initiative store is not wired")
        _mind_cfg = load_config()
        _mind_identity = load_identity(_mind_cfg.home) if _mind_cfg.exists else {}
        _mind_interests = [str(item) for item in (_mind_identity.get("agent") or {}).get("interests") or []]

        def _persist_mind_setting(changes: dict) -> None:
            try:
                update_config(changes)
            except Exception:
                logger.warning("mind setting not written to colonymind.yaml", exc_info=True)

        # One factory for this Mind and the benchmark arm's (colonymind.mind.factory): every store the
        # faculties read is taken from the host router module the stores above were set on.
        mind = build_mind(
            _host_for_mind, config=_mind_cfg.get("mind") or {}, store=_mind_store, state_dir=state_dir,
            ledger=get_turn_idempotency_ledger(state_dir), owner_id=get_owner_contact_id(),
            feedback=_host_for_mind._feedback_store, expectations=_host_for_mind._expectations,
            interests=_mind_interests, persist=_persist_mind_setting,
            heartbeat=lambda: telemetry.touch("last_tick_at"))
        set_mind(mind)
        mind.start()
        logger.info("Mind tick started (autonomy=%s, enabled=%s, owner=%s)",
                    mind.level, mind.enabled, "set" if mind.owner_id else "unset")
    except Exception as exc:
        logger.warning("Mind init failed: %s", exc)

    from colonymind.api.routers import curated_memory
    from colonymind.turns import get_turn_idempotency_ledger
    try:
        curated_memory.bind_ledger(app, get_turn_idempotency_ledger(state_dir))
    except Exception:
        curated_memory.bind_ledger(app, None)
        logger.warning("Curated memory unavailable")

    logger.info("Sidecar capabilities: %s", supported_capabilities())

    source_claim_task = None
    claims_enabled = os.environ.get("COLONYMIND_SOURCE_CLAIMS", "on").strip().lower() in {"on", "1", "true"}
    from colonymind.vector import get_store as source_vector_store
    if claims_enabled or source_vector_store() is not None:
        from colonymind.turns import get_turn_idempotency_ledger
        from colonymind.beliefs.source_projection import run_source_claim_worker
        from colonymind.api.routers import host as source_claim_host
        source_claim_task = asyncio.create_task(run_source_claim_worker(
            get_turn_idempotency_ledger(state_dir), lambda: source_claim_host._llm_router,
            claims_enabled=claims_enabled,
            commitments_provider=lambda: source_claim_host._commitment_store))
    # The keyword indexes' segment merges (deleted entries otherwise slow every keyword search).
    from colonymind.turns import get_turn_idempotency_ledger as keyword_ledger
    from colonymind.turns.idempotency import keyword_index_maintenance
    keyword_maintenance = asyncio.create_task(keyword_index_maintenance(lambda: keyword_ledger(state_dir)))
    # Instinct's daily maintenance (the decision log's retention).
    from colonymind import instinct as instinct_core
    instinct_maintenance = asyncio.create_task(instinct_core.maintenance(state_dir))
    from colonymind.config import load_config as plasticity_config
    from colonymind.plasticity.nightly import lifespan as plasticity_lifespan
    from colonymind.context_card.lifecycle import context_card_lifespan
    async with context_card_lifespan(app, state_dir=state_dir):
        async with plasticity_lifespan(plasticity_config(), data_dir=state_dir, mind=mind) as plasticity:
            app.state.plasticity = plasticity
            yield

    keyword_maintenance.cancel()
    try:
        await keyword_maintenance
    except asyncio.CancelledError:
        pass
    instinct_maintenance.cancel()
    try:
        await instinct_maintenance
    except asyncio.CancelledError:
        pass
    await instinct_core.aclose()

    if source_claim_task is not None:
        source_claim_task.cancel()
        try:
            await source_claim_task
        except asyncio.CancelledError:
            pass
    open_vector_store = source_vector_store()
    if open_vector_store is not None and getattr(open_vector_store, 'compaction', None) is not None:
        await open_vector_store.compaction.close()

    # Shutdown — close connections
    # Stop the mind tick before any store it uses is closed: its in-flight
    # tick must finish.
    try:
        from colonymind.api.routers.mind import get_mind as _get_mind_for_shutdown, set_mind as _set_mind
        _running_mind = _get_mind_for_shutdown()
        if _running_mind is not None:
            await _running_mind.stop()
        _set_mind(None)
    except Exception:
        logger.warning("Mind shutdown failed")
    set_llm_router(None)
    # The remote model providers keep pooled connections (``vector.http_client``); close them.
    from colonymind.api.routers import host as _model_host
    for _model in (_model_host._embedder, _model_host._reranker):
        try:
            if getattr(_model, "close", None) is not None:
                await _model.close()
        except Exception:
            logger.debug("model client close failed", exc_info=True)
    set_embedder(None)
    set_reranker(None)
    # The pipeline recall reads (``vector.get_pipeline``) is the embedder closed above, and the store was opened for
    # it: clear both, so nothing later in this process embeds a query through a closed provider.
    from colonymind.vector import set_pipeline as _set_pipeline, set_store as _set_store
    _set_store(None)
    _set_pipeline(None)
    if contacts_store is not None:
        try:
            await contacts_store.close()
        except Exception:
            logger.debug("ContactStore close failed", exc_info=True)
    set_contacts_store(None)
    set_session_archive_store(None)
    set_session_archive_engine(None)
    try:
        from colonymind.api.routers.host import set_self_model as _set_self_model
        _set_self_model(None)
    except Exception:
        logger.debug("self-model shutdown failed", exc_info=True)
    set_commitment_store(None)
    set_affect_store(None)
    set_secrets_vault(None)
    try:
        from colonymind.api.routers.secrets import set_vault as _unset_secrets_route_vault
        _unset_secrets_route_vault(None)
    except Exception:
        pass
    set_initiative_store(None)
    # Every other host singleton the startup set: an app that has closed leaves nothing wired for the next
    # one, or for a handler called directly in the same process.
    set_comms_log(None)
    set_preference_learner(None)
    set_telemetry(None)
    from colonymind.api.routers.host import (
        set_embed_failure as _clear_embed_failure,
        set_expectations as _clear_expectations,
        set_feedback_store as _clear_feedback_store,
    )
    _clear_feedback_store(None)
    _clear_expectations(None)
    _clear_embed_failure(None)
    logger.info("Sidecar shutdown complete")


@wraps(_lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Hold the lease through startup, all requests and shutdown. An offline
    # restore must never leave a serving reader with a warm pre-restore cache.
    from colonymind.util.state_access import state_access
    from colonymind.loop_lag import loop_lag_lifespan
    async with loop_lag_lifespan():
        with state_access(_state_dir()):
            async with _lifespan(app):
                yield


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    from colonymind.runtime_logging import RequestLogTiming, configure_runtime_logging
    if os.environ.get('COLONYMIND_RUNTIME_LOGGING') == '1':
        configure_runtime_logging(redirect_stdio=True)
    from colonymind import __version__
    app = FastAPI(
        title="ColonyMind",
        version=__version__,
        lifespan=lifespan,
    )

    # API authentication (skips health/docs; loopback dev mode without a key).
    from colonymind.api.auth import configured_api_key
    from colonymind.api.middleware import ApiKeyMiddleware, BodySizeLimitMiddleware

    # Body-size cap runs before auth so oversized payloads are rejected with
    # 413 regardless of the auth state.
    try:
        max_body = int(os.environ.get("COLONYMIND_MAX_BODY_BYTES", "") or 10 * 1024 * 1024)
    except ValueError:
        max_body = 10 * 1024 * 1024
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body)

    api_key = configured_api_key()
    # Who each credential may write as: read once, here, so a malformed grant fails the start.
    from colonymind.api.credentials import CredentialTable, describe
    from colonymind.owner_goals.authority import get_action_verifier
    from colonymind.owner_goals.hub import bind_companion_owner_audience, get_companion_action_verifier
    credentials = CredentialTable.load()
    app.add_middleware(ApiKeyMiddleware, api_key=api_key, credentials=credentials,
                       owner_goal_audience_binder=bind_companion_owner_audience)
    app.dependency_overrides[get_action_verifier] = get_companion_action_verifier
    if api_key:
        logger.info("API key authentication enabled; %s", "; ".join(
            f"{grant.name} {describe(grant)}" for grant in credentials.grants()))
    else:
        logger.warning("No API key configured; serving loopback clients only (dev mode)")

    from colonymind.api.errors import install_exception_handlers
    install_exception_handlers(app)
    from colonymind.api.routers.watch import router as watch_router
    app.include_router(watch_router)
    app.include_router(host_router)
    app.include_router(host_v2_router)
    from colonymind.api.routers.context_card import router as context_card_router
    app.include_router(context_card_router)
    from colonymind.api.routers import devices as devices_router
    app.include_router(devices_router.router)
    from colonymind.api.routers.agent_mailbox import router as agent_mailbox_router
    app.include_router(agent_mailbox_router)
    from colonymind.api.routers.agent_mailbox import agent_router as agent_mailbox_agent_router
    app.include_router(agent_mailbox_agent_router)
    from colonymind.api.routers import work as work_router
    app.include_router(work_router.router)
    from colonymind.mind.factory import mind_routers
    for mind_route in mind_routers():   # the same list the benchmark's mind arm mounts
        app.include_router(mind_route)
    from colonymind.api.routers import executions as executions_router
    app.include_router(executions_router.router)
    from colonymind.api.routers import social_state as social_state_router
    app.include_router(social_state_router.router)
    from colonymind.api.routers import temporal_followups as temporal_followups_router
    app.include_router(temporal_followups_router.router)
    from colonymind.api.routers import transport as transport_router
    app.include_router(transport_router.router)
    from colonymind.api.routers import secrets as secrets_router
    app.include_router(secrets_router.router)
    from colonymind.api.routers import claims as claims_router
    app.include_router(claims_router.router)
    from colonymind.api.routers import branches as branches_router
    app.include_router(branches_router.router)
    from colonymind.api.routers import inventory as inventory_router   # what memory holds about one person
    app.include_router(inventory_router.router)
    from colonymind.api.routers import curated_memory
    app.include_router(curated_memory.router)
    from colonymind.api.routers import instinct as instinct_router   # the Hermes gates' external points
    app.include_router(instinct_router.router)
    from colonymind.api.routers import recorder as recorder_router   # the recorder for callers in another process
    app.include_router(recorder_router.router)
    from colonymind.api.routers import hands as hands_router
    app.include_router(hands_router.runtime_router)
    app.include_router(hands_router.router)
    from colonymind.api import mcp as mcp_endpoint   # the read-first MCP endpoint an issued credential reaches
    app.include_router(mcp_endpoint.router)

    app.add_middleware(RequestLogTiming)
    return app


# Uvicorn entry point
app = create_app()
