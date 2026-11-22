"""Command-line setup, operation and diagnostics for ColonyMind."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from colonymind.util.temporal import now_utc


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="colonymind",
        description="ColonyMind intelligence sidecar server",
    )
    parser.add_argument("--instance", help="Instance directory (default: $COLONYMIND_HOME or ~/.colonymind)")
    sub = parser.add_subparsers(dest="command")
    from colonymind.qualification.cli import add_parser as add_model_parser
    add_model_parser(sub)

    from colonymind.jobwatch.cli import add_parser as add_watch_parser
    add_watch_parser(sub)

    # --- init and upgrade ---
    from colonymind.init import add_parsers as add_init_parsers
    add_init_parsers(sub)
    from colonymind.credentials_cli import add_parser as add_credentials_parser
    add_credentials_parser(sub)

    # --- twin (live replica on a second machine) ---
    from colonymind.twin.cli import add_parser as add_twin_parser
    add_twin_parser(sub)

    # --- start ---
    start_p = sub.add_parser("start", help="Start the sidecar server")
    start_p.add_argument("--host", default=None, help="Override listen host")
    start_p.add_argument("--port", type=int, default=None, help="Override listen port")
    start_p.add_argument("--detach", "-d", action="store_true", help="Run in background (daemon mode)")
    start_p.add_argument("--force", "-f", action="store_true", help="Kill existing process on port if needed")

    # --- stop ---
    sub.add_parser("stop", help="Stop the running sidecar")

    # --- status ---
    sub.add_parser("status", help="Check sidecar health and pipeline status")

    # --- service ---
    service_p = sub.add_parser("service", help="Manage the selected private instance user service")
    service_sub = service_p.add_subparsers(dest="service_command")
    service_sub.add_parser("install", help="Install and enable autostart (does not start the process)")
    service_sub.add_parser("uninstall", help="Remove the selected service; retain private data")
    service_sub.add_parser("start", help="Start the selected service and wait for HTTP readiness")
    service_sub.add_parser("stop", help="Stop the selected service")
    service_sub.add_parser("restart", help="Restart the selected service")
    service_sub.add_parser("status", help="Show manager state and HTTP readiness")

    # --- validate ---
    val_p = sub.add_parser("validate", help="Run end-to-end pipeline validation (uses LLM credits)")
    val_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    # --- doctor ---
    doctor_p = sub.add_parser("doctor", help="Check the install: Hermes version, keys, adapter, service, plugin")
    doctor_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    doctor_p.add_argument("--url", default=None, help="Sidecar URL (default: from colonymind.yaml)")
    doctor_p.add_argument("--api-key", default=None, help="API key (default: api.key)")
    doctor_p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds (default: 10)")
    doctor_p.add_argument("--fix", action="store_true",
                          help="Apply safe automatic fixes (LLM config baseUrl/apiKey), then re-check")

    # --- generate-types ---
    sub.add_parser("generate-types", help="Export OpenAPI spec (for TypeScript generation)")

    # --- backfill ---
    backfill_p = sub.add_parser("backfill", help="Re-embed all vectors with current model")
    backfill_p.add_argument("--collection", default=None, help="Specific collection to backfill (default: all)")
    backfill_p.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding")

    # --- migrate-tier ---
    migrate_p = sub.add_parser("migrate-tier", help="Migrate vectors from old model to current")
    migrate_p.add_argument("--old-model", default=None, help="Old model ID to migrate from (default: all)")
    migrate_p.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding")
    migrate_p.add_argument("--wait-seconds", type=float, default=600,
                           help="Maximum time to wait for completion; the server job continues (default: 600)")

    # --- activate-multimodal ---
    mm_p = sub.add_parser("activate-multimodal", help="Enable multimodal embeddings and rerank")
    mm_p.add_argument("--model", default=None, help="Multimodal model ID (default: auto-detect from tier)")
    mm_p.add_argument("--storage", default="local", choices=["local", "embed_only"], help="Image storage mode")

    # --- backup ---
    backup_p = sub.add_parser("backup", help="Export the instance state (databases, identity, config, vectors) as a portable archive")
    backup_p.add_argument("--full", action="store_true", help="Accepted for compatibility: every backup is a full-state backup")
    backup_p.add_argument("--output", "-o", default=None, help="Output directory (default ~/colonymind-backups)")
    backup_p.add_argument("--passphrase", default=None, help="Encrypt backup with this passphrase (prompted if --encrypt)")
    backup_p.add_argument("--encrypt", action="store_true", help="Encrypt backup (prompts for passphrase)")
    backup_p.add_argument("--no-vectors", action="store_true", help="Skip LanceDB vector store")

    # --- restore ---
    restore_p = sub.add_parser("restore", help="Restore ColonyMind from a backup")
    restore_mode = restore_p.add_mutually_exclusive_group()
    restore_mode.add_argument("--full", action="store_true", help="Reconstruct full archive state; current authority and erasures still require reconciliation")
    restore_mode.add_argument("--memory-only", action="store_true", help="Recover canonical memory using a surviving current source ledger")
    restore_p.add_argument("--input", "-i", default=None, help="Backup file path (default: prompts for it)")
    restore_p.add_argument("--current-state", default=None, help="Surviving authoritative source state (--memory-only)")
    restore_p.add_argument("--output", "-o", default=None, help="Fresh memory bundle destination (--memory-only)")
    restore_p.add_argument("--passphrase", default=None, help="Passphrase to decrypt (default: prompts for it)")
    mm_p.add_argument("--safety", default="basic", choices=["off", "basic", "strict"], help="Image safety level")
    mm_p.add_argument("--skip-download", action="store_true", help="Skip model download")

    # --- secrets: the sealed store (list, show, use, release, forget, migrate) ---
    from colonymind.secrets.cli import add_parser as add_secrets_parser
    add_secrets_parser(sub)

    # --- mind ---
    from colonymind.mind.cli import add_parser as add_mind_parser
    add_mind_parser(sub)
    from colonymind.hands.cli import add_parser as add_hands_parser
    add_hands_parser(sub)
    from colonymind.contacts.cli import add_parser as add_people_parser
    add_people_parser(sub)
    from colonymind.atlas.cli import add_parser as add_atlas_parser
    add_atlas_parser(sub)
    from colonymind.reports.cli import add_parser as add_report_parser
    add_report_parser(sub)
    from colonymind.recall_cli import add_parser as add_recall_parser
    add_recall_parser(sub)
    from colonymind.demo import add_parser as add_demo_parser
    add_demo_parser(sub)
    from colonymind.instinct.cli import add_parser as add_instinct_parser
    add_instinct_parser(sub)
    from colonymind.plasticity.cli import add_parser as add_plasticity_parser
    add_plasticity_parser(sub)
    wm_p = sub.add_parser("wm", help="Manage working memory branches and consults")
    wm_sub = wm_p.add_subparsers(dest="wm_command", required=True)
    wm_sub.add_parser("reset", help="Clear automatic withdrawal of the consult tool")
    from colonymind.memory_cli import add_parser as add_memory_parser   # the memory inventory and claim history
    add_memory_parser(sub)
    # ``memory reextract`` joins the ``memory`` group the inventory made.
    from colonymind.beliefs.reextract_cli import add_parser as add_reextract_parser
    add_reextract_parser(sub)
    from colonymind.console.cli import add_parser as add_console_parser   # the owner console on loopback
    add_console_parser(sub)

    # --- feeds ---
    feeds_p = sub.add_parser("feeds", help="Manage spec-driven intelligence feeds")
    feeds_p.add_argument("feeds_args", nargs=argparse.REMAINDER,
                         help="Feeds subcommand (create/validate/list/status/pause/resume/run/delete)")

    # --- initiative ---
    init_p = sub.add_parser("initiative", help="Manage initiatives")
    init_sub = init_p.add_subparsers(dest="initiative_command")

    init_list = init_sub.add_parser("list", help="List initiatives")
    init_list.add_argument("--status", default=None, help="Filter by status (pending, assigned, acknowledged, completed, failed, cancelled)")
    init_list.add_argument("--agent", default=None, help="Filter by assigned agent")
    init_list.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")

    init_show = init_sub.add_parser("show", help="Show initiative details")
    init_show.add_argument("initiative_id", help="Initiative ID")

    init_cancel = init_sub.add_parser("cancel", help="Cancel an initiative")
    init_cancel.add_argument("initiative_id", help="Initiative ID")
    init_cancel.add_argument("--reason", default=None, help="Reason for cancellation")

    args = parser.parse_args()
    if args.instance:
        os.environ["COLONYMIND_INSTANCE_SELECTED"] = "1"
        selected = str(Path(args.instance).expanduser().resolve())
        os.environ["COLONYMIND_HOME"] = selected
        os.environ["COLONYMIND_STATE_DIR"] = selected

    if args.command == "watch":
        from colonymind.jobwatch.cli import main as run_watch
        raise SystemExit(run_watch(args))

    if args.command == "models":
        from colonymind.qualification.cli import run
        try:
            code = run(args)
        except (ValueError, OSError, KeyError) as exc:
            print(f"Model qualification failed: {type(exc).__name__}", file=sys.stderr)
            raise SystemExit(2) from None
        raise SystemExit(code)

    if args.command == "twin":
        from colonymind.twin.cli import run as run_twin
        sys.exit(run_twin(args))

    if args.command == "init":
        from colonymind.init import run_init
        code = run_init(args)
        if code != 0 or args.uninstall:
            sys.exit(code)

    elif args.command == "upgrade":
        from colonymind.init import run_upgrade
        sys.exit(run_upgrade(args))

    elif args.command == "credentials":
        from colonymind.credentials_cli import run as run_credentials
        sys.exit(run_credentials(args))

    elif args.command == "start":
        _load_dotenv()
        host = args.host or os.environ.get("COLONYMIND_SIDECAR_HOST", "127.0.0.1")
        port = args.port or int(os.environ.get("COLONYMIND_SIDECAR_PORT", "7777"))

        # Fail closed: never serve an unauthenticated API on a public interface.
        _guard_bind_auth(host)

        # Service-aware: error out if launchd is managing the sidecar
        if _is_service_loaded():
            print("❌ A user service is managing this sidecar.")
            print("  Use 'colonymind service stop' and 'colonymind service start' instead,")
            sys.exit(1)

        if args.detach:
            _cmd_start_daemon(host, port, args.force)
        else:
            # Foreground mode — check port first
            existing_pid = _find_pid_on_port(port)
            if existing_pid:
                if _configured_instance():
                    print("Selected port is already in use; no process was stopped.")
                    raise SystemExit(1)
                if args.force:
                    print(f"Killing existing process {existing_pid} on port {port}...")
                    try:
                        os.kill(existing_pid, 15)  # SIGTERM
                        time.sleep(2)
                        if _find_pid_on_port(port):
                            os.kill(existing_pid, 9)  # SIGKILL
                            time.sleep(1)
                        print("Process killed.")
                    except ProcessLookupError:
                        pass
                else:
                    print(f"Error: Port {port} is already in use (PID {existing_pid})")
                    print("Use --force to kill existing process, or stop it first with: colonymind stop")
                    sys.exit(1)
            
            import uvicorn
            try:
                ws_max_size = int(
                    os.environ.get("COLONYMIND_MAX_WS_FRAME_BYTES", "") or 1 * 1024 * 1024
                )
            except ValueError:
                ws_max_size = 1 * 1024 * 1024
            from colonymind.runtime_logging import configure_runtime_logging
            configure_runtime_logging(redirect_stdio=bool(os.environ.get('COLONYMIND_INSTANCE_SERVICE')))
            uvicorn.run(
                "colonymind.server:app",
                host=host,
                port=port,
                log_level=os.environ.get("LOG_LEVEL", "info").lower(),
                log_config=None,
                ws_max_size=ws_max_size,
            )

    elif args.command == "stop":
        _load_dotenv()
        # Service-aware: error out if launchd is managing the sidecar
        if _is_service_loaded():
            print("❌ Sidecar is managed by a user service.")
            print("  Use 'colonymind service stop' instead.")
            sys.exit(1)
        _cmd_stop()

    elif args.command == "status":
        _load_dotenv()
        _cmd_status()

    elif args.command == "service":
        if not hasattr(args, "service_command") or not args.service_command:
            print("❌ No service subcommand given")
            print("  Usage: colonymind service {install|uninstall|start|stop|restart|status}")
            sys.exit(1)
        from colonymind.services.instance import manage, ServiceError
        try:
            manage(args.service_command)
        except (ServiceError, ValueError, OSError) as exc:
            print(f"Service operation failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from None

    elif args.command == "generate-types":
        _load_dotenv()
        import json
        from colonymind.server import create_app
        app = create_app()
        spec = app.openapi()
        out = os.environ.get("COLONYMIND_OPENAPI_OUT", "openapi.json")
        with open(out, "w") as f:
            json.dump(spec, f, indent=2)
        n = len(spec.get("components", {}).get("schemas", {}))
        p = len(spec.get("paths", {}))
        print(f"✅ OpenAPI spec written to {out} ({n} schemas, {p} paths)")

    elif args.command == "backfill":
        _load_dotenv()
        import httpx
        host = os.environ.get("COLONYMIND_SIDECAR_HOST", "127.0.0.1")
        port = os.environ.get("COLONYMIND_SIDECAR_PORT", "7777")
        api_key = os.environ.get("COLONYMIND_API_KEY", "")
        try:
            resp = httpx.post(
                f"http://{host}:{port}/v1/host/memory/backfill",
                json={"identity": {"host_id": "cli"}, "collection": args.collection, "batch_size": args.batch_size},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                task_id = data.get("task_id", "")
                print(f"Backfill started (task_id={task_id})")
                while True:
                    time.sleep(2)
                    status_resp = httpx.get(
                        f"http://{host}:{port}/v1/host/memory/backfill/{task_id}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10,
                    )
                    if status_resp.status_code == 200:
                        sd = status_resp.json()
                        if sd.get("status") == "completed":
                            print(f"Backfill complete: {sd.get('processed', 0)} processed, {sd.get('skipped', 0)} skipped, {sd.get('failed', 0)} failed")
                            break
                        elif sd.get("status") == "failed":
                            print(f"Backfill failed: {sd.get('errors', [])}")
                            break
                        else:
                            print(f"  ... {sd.get('processed', 0)} processed so far")
            else:
                print(f"Backfill failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"Could not connect to sidecar: {e}")

    elif args.command == "migrate-tier":
        if not 0 < args.wait_seconds < float("inf"):
            parser.error("--wait-seconds must be finite and greater than zero")
        _load_dotenv()
        import httpx
        host = os.environ.get("COLONYMIND_SIDECAR_HOST", "127.0.0.1")
        port = os.environ.get("COLONYMIND_SIDECAR_PORT", "7777")
        api_key = os.environ.get("COLONYMIND_API_KEY", "")
        try:
            resp = httpx.post(
                f"http://{host}:{port}/v1/host/memory/migrate",
                json={"identity": {"host_id": "cli"}, "old_model_id": args.old_model, "batch_size": args.batch_size},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                task_id = data.get("task_id", "")
                print(f"Migration started (task_id={task_id})")
                deadline = time.monotonic() + args.wait_seconds
                status_url = f"http://{host}:{port}/v1/host/memory/migrate/{task_id}"
                while time.monotonic() < deadline:
                    time.sleep(min(2, max(0, deadline - time.monotonic())))
                    status_resp = httpx.get(
                        status_url,
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10,
                    )
                    if status_resp.status_code == 200:
                        sd = status_resp.json()
                        if sd.get("status") == "completed":
                            print(f"Migration complete: {sd.get('vectors_migrated', 0)} vectors migrated, {sd.get('collections_migrated', 0)} collections")
                            break
                        elif sd.get("status") == "failed":
                            print(f"Migration failed: {sd.get('errors', [])}")
                            print("Re-run colonymind migrate-tier with the same settings to resume staged work.")
                            raise SystemExit(1)
                        elif sd.get("status") == "resumable":
                            print("Migration is interrupted, not complete. Re-run colonymind migrate-tier with the same settings to resume staged work.")
                            raise SystemExit(2)
                        else:
                            print(f"  ... {sd.get('vectors_migrated', 0)} vectors migrated so far")
                    else:
                        print(f"Migration status unavailable (HTTP {status_resp.status_code}); completion is not confirmed.")
                        print("After checking the sidecar, re-run colonymind migrate-tier with the same settings to resume staged work.")
                        raise SystemExit(2)
                else:
                    print(f"Stopped waiting; the server migration may still be running. Check GET {status_url}.")
                    print("After an interruption, re-run colonymind migrate-tier with the same settings to resume staged work.")
                    raise SystemExit(2)
            else:
                print(f"Migration failed: {resp.status_code} {resp.text}")
                raise SystemExit(1)
        except Exception as e:
            print(f"Could not connect to sidecar: {e}")
            print("Completion is not confirmed. Check the sidecar before resuming colonymind migrate-tier.")
            raise SystemExit(1)

    elif args.command == "activate-multimodal":
        _load_dotenv()
        env_path = Path(os.environ.get("COLONYMIND_STATE_DIR", ".")) / ".env"
        if not env_path.exists():
            # Try sidecar directory
            env_path = Path(__file__).parent / ".env"
        if not env_path.exists():
            print("No .env file found. Run 'colonymind init' first.")
            return

        # Determine multimodal model from tier
        model = args.model
        reranker_model = ""
        dims = 0

        if not model:
            try:
                from colonymind.vector.tiers import TIER_TABLE
                from colonymind.vector.scanner import HardwareScanner
                scanner = HardwareScanner()
                scan = scanner.scan()
                tier = scanner.recommend_tier(scan)
                if tier and tier.multimodal_embedder:
                    model = tier.multimodal_embedder.model_id
                    dims = tier.multimodal_embedder.dims
                    if tier.multimodal_reranker:
                        reranker_model = tier.multimodal_reranker.model_id
                else:
                    print("Your hardware tier does not support multimodal embeddings.")
                    print("Available from Tier 1 (4GB+) with jina-clip-v2.")
                    return
            except Exception as exc:
                print(f"Could not auto-detect tier: {exc}")
                print("Use --model to specify a multimodal model ID.")
                return

        print(f"Activating multimodal embeddings:")
        print(f"  Model: {model}")
        print(f"  Dims: {dims}")
        if reranker_model:
            print(f"  Reranker: {reranker_model}")
        print(f"  Storage: {args.storage}")
        print(f"  Safety: {args.safety}")
        print()

        answer = input("Continue? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled.")
            return

        # Update .env
        lines = env_path.read_text().splitlines()
        updates = {
            "COLONYMIND_MULTIMODAL": "true",
            "COLONYMIND_EMBED_MODEL": model,
            "COLONYMIND_IMAGE_STORAGE": args.storage,
            "COLONYMIND_IMAGE_SAFETY": args.safety,
            "COLONYMIND_STRIP_EXIF_GPS": "true",
        }
        if dims:
            updates["COLONYMIND_EMBED_DIMS"] = str(dims)
        if reranker_model:
            updates["COLONYMIND_RERANKER_MODEL"] = reranker_model

        existing_keys = set()
        for i, line in enumerate(lines):
            if "=" in line and not line.strip().startswith("#"):
                key = line.split("=", 1)[0].strip()
                existing_keys.add(key)
                if key in updates:
                    lines[i] = f"{key}={updates[key]}"

        # Add new keys not yet in .env
        for key, value in updates.items():
            if key not in existing_keys:
                lines.append(f"{key}={value}")

        env_path.write_text("\n".join(lines) + "\n")
        print(f"\n✅ .env updated with multimodal config")

        # Download model
        if not args.skip_download:
            print(f"Downloading multimodal model {model}...")
            try:
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(model)
                print(f"✅ Model downloaded and cached")
            except Exception as exc:
                print(f"⚠️ Model download failed: {exc}")
                print("The model will download on first start instead.")

        print()
        print("Restart the sidecar to activate multimodal: colonymind start")
        print("If you have existing text vectors, run: colonymind migrate-tier")

    elif args.command == "validate":
        _load_dotenv()
        _cmd_validate(args)

    elif args.command == "doctor":
        _cmd_doctor(args)

    elif args.command == "backup":
        _cmd_backup(args)

    elif args.command == "restore":
        _cmd_restore(args)

    elif args.command == "initiative":
        _cmd_initiative(args)

    elif args.command == "secrets":
        _load_dotenv()
        from colonymind.secrets import cli as _secrets_cli
        _handler = getattr(_secrets_cli, f"cmd_secrets_{args.secrets_cmd}")
        _handler(args)

    elif args.command == "mind":
        _load_dotenv()
        from colonymind.mind.cli import run as run_mind_command
        sys.exit(run_mind_command(args))

    elif args.command == "hands":
        sys.exit(args.hands_handler(args))

    elif args.command == "people":
        _load_dotenv()
        from colonymind.contacts.cli import run as run_people_command
        sys.exit(run_people_command(args))

    elif args.command == "atlas":
        _load_dotenv()
        from colonymind.atlas.cli import run as run_atlas_command
        sys.exit(run_atlas_command(args))

    elif args.command == "report":
        from colonymind.reports.cli import run as run_report_command
        sys.exit(run_report_command(args))

    elif args.command == "recall":
        _load_dotenv()
        from colonymind.recall_cli import run as run_recall_command
        sys.exit(run_recall_command(args))

    elif args.command == "recall-questions":
        _load_dotenv()
        from colonymind.recall_cli import run_questions as run_recall_questions_command
        sys.exit(run_recall_questions_command(args))

    elif args.command == "memory" and getattr(args, "memory_command", None) == "reextract":
        _load_dotenv()
        from colonymind.beliefs.reextract_cli import run as run_reextract_command
        sys.exit(run_reextract_command(args))

    elif args.command == "demo":
        from colonymind.demo import run as run_demo
        sys.exit(run_demo(args))

    elif args.command == "instinct":
        from colonymind.instinct.cli import run as run_instinct_command
        sys.exit(run_instinct_command(args))

    elif args.command == "plasticity":
        from colonymind.plasticity.cli import run as run_plasticity_command
        sys.exit(run_plasticity_command(args))

    elif args.command == "wm":
        from colonymind.config import load_config
        from colonymind.memory.branches import get_store
        try:
            get_store(load_config()).reset()
        except (ValueError, OSError) as error:
            print(f"colonymind wm reset: {error}", file=sys.stderr)
            sys.exit(2)
        print("Working memory consult withdrawal cleared.")
        sys.exit(0)

    elif args.command == "memory":
        _load_dotenv()
        from colonymind.memory_cli import run as run_memory_command
        sys.exit(run_memory_command(args))

    elif args.command == "console":
        _load_dotenv()
        from colonymind.console.cli import run as run_console
        sys.exit(run_console(args))

    elif args.command == "feeds":
        from colonymind.feeds.cli import main as feeds_main
        feeds_main(args.feeds_args)

    else:
        parser.print_help()


def _cmd_backup(args) -> None:
    """Export the instance state as a portable archive."""
    _load_dotenv()
    state_dir = os.environ.get("COLONYMIND_STATE_DIR", os.getcwd())

    passphrase = None
    if args.passphrase:
        passphrase = args.passphrase.encode()
    elif args.encrypt:
        import getpass
        passphrase = getpass.getpass("Backup passphrase: ").encode()

    from colonymind.backup import create_full_backup
    output_dir = args.output or os.path.expanduser("~/colonymind-backups")
    try:
        archive = create_full_backup(
            state_dir, output_dir,
            passphrase=passphrase,
            include_vectors=not getattr(args, "no_vectors", False),
        )
        print(f"  Full backup saved to {archive}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Run 'colonymind init' first.")
        raise SystemExit(1)


def _cmd_restore(args) -> None:
    """Restore ColonyMind from a backup -- interactive by default."""
    _load_dotenv()
    state_dir = os.environ.get("COLONYMIND_STATE_DIR", os.getcwd())
    memory_only = getattr(args, "memory_only", False)
    if memory_only and (not getattr(args, "current_state", None)
                        or not getattr(args, "output", None)):
        print("  Memory recovery requires --current-state and a fresh --output.")
        raise SystemExit(2)
    if not memory_only and (getattr(args, "current_state", None) or getattr(args, "output", None)):
        print("  --current-state and --output require --memory-only.")
        raise SystemExit(2)

    if args.input:
        backup_path = args.input
    else:
        backup_path = input("  Backup file path: ").strip()

    if not backup_path or not Path(backup_path).exists():
        print(f"  Error: File not found: {backup_path}")
        raise SystemExit(1)

    passphrase = None
    if args.passphrase:
        passphrase = args.passphrase.encode()
    if passphrase is None and backup_path.endswith(".enc"):
        import getpass
        passphrase = getpass.getpass("  Backup passphrase: ").encode()

    from colonymind.backup import restore_full_backup, restore_source_memory
    try:
        if memory_only:
            summary = restore_source_memory(
                backup_path, args.output, current_state=args.current_state,
                passphrase=passphrase,
            )
            print(f"\n  Current source memory recovered to {args.output}")
            print(f"  Sources: {summary['source_count']}; original images: {summary['source_images']}")
            print("  Files: turn-idempotency.db, owned images and source-memory-recovery.json.")
            print("  Install only these memory files with writers stopped and separately current runtime bindings.")
            print("  The bundle does not restore runtime authority or completed-effect state.")
            return
        summary = restore_full_backup(backup_path, state_dir, passphrase=passphrase)
        print(f"\n  Archive reconstructed: instance {summary['instance_id']}")
        print(f"  Databases: {', '.join(summary.get('databases', []))}")
        if summary.get("plasticity"):
            print("  Learned weights restored (plasticity/).")
        if summary.get("plasticity_key_missing"):
            print("  Learned weights remain usable; old erasure and source hashes no longer match unless this is the instance that made the archive. Learners create a comparator key if needed.")
        if "vector_jobs_queued" in summary:
            print(f"  Vector jobs queued for rebuild: {summary['vector_jobs_queued']}")
        for name in summary.get("not_in_archive", []):
            print(f"  Not in the archive: {name}")
        for name in summary.get("not_restored", []):
            print(f"  Not restored: {name}")
        for error in summary.get("errors", []):
            print(f"  {error}")
        print("  Reconcile current authority, erasures and completed effects before starting services.")
        print("  Use --memory-only for bounded recovery from a surviving current source ledger.")
    except ValueError as e:
        print(f"  Error: {e}")
        raise SystemExit(1)


def _cmd_initiative(args) -> None:
    """Handle initiative subcommands."""
    _load_dotenv()

    import httpx
    host = os.environ.get("COLONYMIND_SIDECAR_HOST", "127.0.0.1")
    port = os.environ.get("COLONYMIND_SIDECAR_PORT", "7777")
    api_key = os.environ.get("COLONYMIND_API_KEY", "")
    base_url = f"http://{host}:{port}/v1/host"
    headers = {"Authorization": f"Bearer {api_key}"}

    if args.initiative_command == "list":
        params = {"limit": args.limit}
        if args.status:
            params["status"] = args.status
        if args.agent:
            params["agent_id"] = args.agent
        resp = httpx.get(f"{base_url}/initiatives", params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            raise SystemExit(1)
        data = resp.json()
        initiatives = data.get("initiatives", [])
        if not initiatives:
            print("No initiatives found.")
            return
        print(f"{'ID':<36} {'Type':<15} {'Status':<12} {'Priority':<8} {'Description'}")
        print("-" * 120)
        for i in initiatives:
            desc = i['description'][:60] + "..." if len(i['description']) > 60 else i['description']
            print(f"{i['id']:<36} {i['initiative_type']:<15} {i['status']:<12} {i['priority']:<8} {desc}")

    elif args.initiative_command == "show":
        resp = httpx.get(f"{base_url}/initiatives/{args.initiative_id}", headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            raise SystemExit(1)
        i = resp.json()
        print(f"Initiative ID: {i['id']}")
        print(f"Type: {i['initiative_type']}")
        print(f"Status: {i['status']}")
        print(f"Priority: {i['priority']}")
        print(f"Description: {i['description']}")
        if i.get('assigned_agent_id'):
            print(f"Assigned to: {i['assigned_agent_id']}")
        if i.get('result'):
            print(f"Result: {i['result']}")
        if i.get('error_message'):
            print(f"Error: {i['error_message']}")
        print(f"Created: {i['created_at']}")
        if i.get('completed_at'):
            print(f"Completed: {i['completed_at']}")
        if i.get('failed_at'):
            print(f"Failed: {i['failed_at']}")

    elif args.initiative_command == "cancel":
        resp = httpx.post(
            f"{base_url}/initiatives/{args.initiative_id}/cancel",
            json={"reason": args.reason},
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            raise SystemExit(1)
        print(f"Initiative {args.initiative_id} cancelled.")

    else:
        print("Usage: colonymind initiative [list|show|cancel]")


def _is_loopback_host(host: str) -> bool:
    """True if the bind host only accepts local connections."""
    from colonymind.api.auth import is_loopback_host
    return is_loopback_host(host)


def _guard_bind_auth(host: str) -> None:
    """Refuse to start an unauthenticated sidecar on a non-loopback interface.

    Without a key the API runs in dev mode and ApiKeyMiddleware serves loopback
    callers only, so a non-loopback bind would answer every remote request
    with 403. Fail at startup with the fix instead.
    """
    from colonymind.api.auth import configured_api_key
    if _is_loopback_host(host):
        return
    if configured_api_key():
        return
    print(
        f"Refusing to start: binding to {host!r} (non-loopback) with no API key; "
        "without a key the API only serves loopback clients.\n"
        "  Fix one of:\n"
        "    - write the key to api.key in the instance directory (colonymind init "
        "does this) or set COLONYMIND_API_KEY=<secret>, or\n"
        "    - bind to 127.0.0.1 (default) and reach it via SSH/proxy.",
        file=sys.stderr,
    )
    sys.exit(2)


def _find_pid_on_port(port: int) -> int | None:
    """Find the PID of a process listening on the given port.

    NOTE: This only finds LISTEN sockets, not client connections.
    """
    pids = _find_pids_on_port(port)
    return pids[0] if pids else None


def _find_pids_on_port(port: int) -> list[int]:
    """Find ALL PIDs in LISTEN state on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        return [int(p) for p in result.stdout.strip().splitlines() if p.isdigit()]
    except Exception:
        return []


def _is_service_loaded() -> bool:
    """Check whether the selected instance's user service manages this process."""
    if _configured_instance():
        from colonymind.services.instance import InstanceService
        service = InstanceService.selected()
        if os.environ.get("COLONYMIND_INSTANCE_SERVICE") == service.label:
            return False  # This foreground invocation belongs to that manager.
        if not service.link.exists() and not service.link.is_symlink():
            return False
        state = service.status()
        return (state["loaded"] if service.platform == "darwin" else
                state.get("state") in {"active", "activating", "deactivating"})
    return False


def _find_orphan_processes() -> list[int]:
    """Find orphaned colonymind sidecar processes (parent died).
    
    Returns list of PIDs that are:
    - Running uvicorn/colonymind
    - Have parent PID 1 (init) or parent doesn't exist
    """
    orphans = []
    try:
        # Find all python processes running uvicorn or colonymind
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return orphans
        
        for line in result.stdout.splitlines():
            # Look for colonymind sidecar processes
            if "uvicorn" in line and "colonymind" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        ppid = int(parts[2]) if len(parts) > 2 else 0
                        # Parent PID 1 = orphan (reparented to init)
                        # Also check if parent exists
                        if ppid == 1:
                            orphans.append(pid)
                        elif ppid > 1:
                            try:
                                os.kill(ppid, 0)  # Check if parent exists
                            except OSError:
                                # Parent doesn't exist, this is an orphan
                                orphans.append(pid)
                    except (ValueError, OSError):
                        pass
    except Exception:
        pass
    return orphans


def _cleanup_orphans(kill: bool = False) -> int:
    """Find and optionally kill orphaned colonymind processes.
    
    Args:
        kill: If True, kill the orphans; if False, just report them
    
    Returns:
        Number of orphans found
    """
    orphans = _find_orphan_processes()
    if not orphans:
        return 0
    
    if kill:
        for pid in orphans:
            try:
                os.kill(pid, 15)  # SIGTERM
                print(f"  Killed orphan process {pid}")
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"  Failed to kill {pid}: {e}")
        time.sleep(1)
        # Check if any survived
        for pid in orphans:
            try:
                os.kill(pid, 0)
                os.kill(pid, 9)  # SIGKILL
                print(f"  Force-killed stubborn process {pid}")
            except ProcessLookupError:
                pass
    
    return len(orphans)


def _cmd_start_daemon(host: str, port: int, force: bool) -> None:
    """Start the sidecar as a background daemon."""
    # Clean up any orphaned processes first
    local_instance = _configured_instance()
    if local_instance:
        from colonymind.setup import _check_port
        if _check_port(port):
            print("Selected port is already in use; no process was stopped.")
            raise SystemExit(1)
    orphan_count = 0 if local_instance else _cleanup_orphans(kill=True)
    if orphan_count:
        print(f"  Cleaned up {orphan_count} orphan process(es)")

    # Check if port is already in use
    existing_pids = _find_pids_on_port(port)
    if existing_pids:
        if local_instance:
            print("Selected port is already in use; no process was stopped.")
            raise SystemExit(1)
        if force:
            for pid in existing_pids:
                print(f"  Killing existing process {pid} on port {port}...")
                try:
                    os.kill(pid, 15)  # SIGTERM
                except ProcessLookupError:
                    pass
            # Wait up to 5s for all to die
            for _ in range(10):
                if not _find_pids_on_port(port):
                    break
                time.sleep(0.5)
            # Escalate to SIGKILL for any survivors
            for pid in _find_pids_on_port(port):
                try:
                    os.kill(pid, 9)  # SIGKILL
                except ProcessLookupError:
                    pass
            time.sleep(0.5)
            print(f"  ✅ Process(es) killed")
        else:
            print(f"  ⚠️ Port {port} is already in use (PIDs {existing_pids})")
            try:
                answer = input("  Kill existing process and restart? [Y/n] ").strip().lower()
            except EOFError:
                answer = "y"  # Default to yes when no stdin (e.g. scripts)
            if answer in ("n", "no"):
                print("  Cancelled.")
                return
            for pid in existing_pids:
                try:
                    os.kill(pid, 15)
                except ProcessLookupError:
                    pass
            for _ in range(10):
                if not _find_pids_on_port(port):
                    break
                time.sleep(0.5)
            for pid in _find_pids_on_port(port):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
            time.sleep(0.5)
            print(f"  ✅ Process(es) killed")

    # Build env from .env values
    env = {**os.environ}
    env["COLONYMIND_SIDECAR_HOST"] = host
    env["COLONYMIND_SIDECAR_PORT"] = str(port)

    # Start uvicorn
    from colonymind.runtime_logging import runtime_log_directory
    log_path = runtime_log_directory() / "sidecar.log"
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    env['COLONYMIND_LOG_PATH'] = str(log_path)
    env['COLONYMIND_RUNTIME_LOGGING'] = '1'
    print(f"  Starting ColonyMind sidecar on {host}:{port}...")
    print(f"  Log: {log_path}")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "colonymind.server:app",
         "--host", host,
         "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

    # Write PID file
    pid_path = Path(os.environ.get("COLONYMIND_STATE_DIR", ".")) / "sidecar.pid"
    pid_path.write_text(str(proc.pid))
    # Wait for health check
    _load_dotenv()
    api_key = os.environ.get("COLONYMIND_CLIENT_API_KEY") or os.environ.get("COLONYMIND_API_KEY", "dev-mode-no-key")
    try:
        healthy = _wait_for_sidecar(host, port, api_key, timeout=20.0)
    except BaseException:
        if local_instance and proc.poll() is None:
            proc.terminate()
        raise
    finally:
        if local_instance and proc.poll() is None:
            # macOS Python launchers exec the framework interpreter during startup.
            # Retain the serving child, including after an interrupted wait.
            record = pid_path.with_name("sidecar-process.json")
            record.write_text(json.dumps({"pid": proc.pid, "signature": _process_signature(proc.pid)}))
            record.chmod(0o600)
    if healthy:
        import httpx
        try:
            r = httpx.get(
                f"http://{host}:{port}/v1/host/health",
                headers={"X-API-Key": api_key},
                timeout=2,
            )
            data = r.json()
            caps = len(data.get("capabilities", []))
            print(f"  ✅ Sidecar running (PID {proc.pid}, {caps} capabilities)")
            # Check E2E validation status
            stamp = Path(os.environ.get("COLONYMIND_STATE_DIR", ".")) / ".colonymind-e2e-validated"
            if not stamp.exists():
                print(f"  ⚠️ E2E pipeline not validated — run 'colonymind validate' to test")
            return
        except Exception:
            pass
    else:
        print(f"  ❌ Sidecar didn't become healthy within 20s")
        # Print log tail
        if log_path.exists():
            try:
                lines = log_path.read_text().splitlines()
                tail = lines[-20:] if len(lines) > 20 else lines
                print(f"\n  Last log lines:")
                for line in tail:
                    print(f"    {line}")
            except Exception:
                pass
        print(f"  PID: {proc.pid}")
        sys.exit(1)


def _wait_for_sidecar(host: str, port: int, api_key: str, timeout: float = 10.0) -> bool:
    """Poll /health until the sidecar responds or timeout."""
    import httpx
    headers = {"X-API-Key": api_key}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(
                f"http://{host}:{port}/v1/host/health",
                headers=headers,
                timeout=2,
            )
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _process_signature(pid: int) -> str:
    """PID creation time and command prevent a stale file targeting PID reuse."""
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                            capture_output=True, text=True, timeout=5)
    return result.stdout.strip() if result.returncode == 0 else ""


def _cmd_stop() -> None:
    """Stop the running sidecar."""
    # Try PID file first
    pid_path = Path(os.environ.get("COLONYMIND_STATE_DIR", ".")) / "sidecar.pid"
    port = int(os.environ.get("COLONYMIND_SIDECAR_PORT", "7777"))

    pid = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pass

    # New private instances never infer process ownership from a port alone.
    if not pid and _configured_instance():
        print("No recorded process for this instance.")
        return
    if _configured_instance():
        try:
            record_path = pid_path.with_name("sidecar-process.json")
            record = json.loads(record_path.read_text())
            expected = record.get("signature")
            if record.get("pid") != pid or not expected or _process_signature(pid) != expected:
                raise ValueError()
        except (OSError, ValueError, subprocess.SubprocessError):
            print("Recorded process is gone or changed; no process was stopped.")
            return
        os.kill(pid, 15)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _process_signature(pid) == expected:
            time.sleep(.1)
        if _process_signature(pid) == expected:
            print("Sidecar has not stopped yet; inspect this instance's log before retrying.")
            raise SystemExit(1)
        pid_path.unlink(missing_ok=True)
        record_path.unlink(missing_ok=True)
        print("Private instance stopped.")
        return
    # Fallback: find by port
    if not pid:
        pid = _find_pid_on_port(port)

    if not pid:
        print(f"  No sidecar process found on port {port}")
        return

    try:
        os.kill(pid, 15)  # SIGTERM
        print(f"  Stopping sidecar (PID {pid})...")
        time.sleep(2)

        # Check if still alive
        if _find_pid_on_port(port):
            print(f"  Process didn't stop gracefully, killing...")
            os.kill(pid, 9)  # SIGKILL
            time.sleep(1)

        print(f"  ✅ Sidecar stopped")

        # Clean up PID file
        if pid_path.exists():
            pid_path.unlink()

    except ProcessLookupError:
        print(f"  Process {pid} already gone")
        if pid_path.exists():
            pid_path.unlink()
        
        # Check for orphan cleanup
        orphan_count = _cleanup_orphans(kill=True)
        if orphan_count:
            print(f"  Cleaned up {orphan_count} orphan process(es)")


def _cmd_status() -> None:
    """Check sidecar health and pipeline status."""
    import httpx

    host = os.environ.get("COLONYMIND_SIDECAR_HOST", "127.0.0.1")
    port = os.environ.get("COLONYMIND_SIDECAR_PORT", "7777")
    url = f"http://{host}:{port}"
    api_key = os.environ.get("COLONYMIND_CLIENT_API_KEY") or os.environ.get("COLONYMIND_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        resp = httpx.get(f"{url}/v1/host/health", headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        caps = data.get("capabilities", [])
        notes = data.get("notes", {})
        if _configured_instance():
            print(f"Instance: {os.environ.get('COLONYMIND_STATE_DIR', '')}")
            print(f"Sidecar HTTP status: {status}; endpoint: {url}")
            print("Verify recollection by telling Hermes a fact, then asking in a new session.")
            return

        # Status icon
        icon = "🟢" if status == "ok" else "🔴"
        print(f"{icon} ColonyMind Sidecar — {status}")
        print(f"  URL: {url}")
        print(f"  Capabilities: {len(caps)}")

        # Show notable notes
        for k, v in notes.items():
            if "fail" in str(v).lower() or "error" in str(v).lower() or "not wired" in str(v).lower():
                print(f"  ⚠️  {k}: {v}")

        # Check E2E validation stamp
        stamp = Path(os.environ.get("COLONYMIND_STATE_DIR", ".")) / ".colonymind-e2e-validated"
        if stamp.exists():
            stamp_data = json.loads(stamp.read_text())
            validated_at = stamp_data.get("validated_at", "unknown")
            print(f"  ✅ E2E validated: {validated_at}")
        else:
            print(f"  ⚠️  E2E pipeline not validated")
            print(f"     Run 'colonymind validate' to test the full pipeline")

    except Exception as exc:
        print(f"🔴 Sidecar not reachable: {exc}")
        # Check if process exists
        pid = _find_pid_on_port(int(port))
        if pid:
            print(f"  Process {pid} is on port {port} but not responding")
        else:
            print(f"  No process on port {port}")
            print(f"  Start with: colonymind start")
        sys.exit(1)


def _cmd_validate(args) -> None:
    """Run end-to-end pipeline validation."""
    import httpx

    host = os.environ.get("COLONYMIND_SIDECAR_HOST", "127.0.0.1")
    port = os.environ.get("COLONYMIND_SIDECAR_PORT", "7777")
    url = f"http://{host}:{port}"
    api_key = os.environ.get("COLONYMIND_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    state_dir = os.environ.get("COLONYMIND_STATE_DIR", ".")

    print("🧪 ColonyMind E2E Pipeline Validation")
    print("=" * 40)
    print()

    # Step 0: Confirm LLM usage
    if not args.yes:
        print("This test will send one prompt through the LLM to verify the full pipeline.")
        print("It uses a small amount of LLM API credits.")
        try:
            answer = input("Continue? [y/N] ").strip().lower()
        except EOFError:
            print("  \u26a0\ufe0f No stdin — use --yes to skip confirmation")
            return
        if answer not in ("y", "yes"):
            print("Cancelled.")
            return

    # Step 1: Check sidecar is running
    print("\n[1/5] Checking sidecar...")
    try:
        resp = httpx.get(f"{url}/v1/host/health", headers=headers, timeout=5)
        data = resp.json()
        if data.get("status") != "ok":
            print(f"  ❌ Sidecar not healthy: {data.get('status')}")
            return
        print(f"  ✅ Sidecar running ({len(data.get('capabilities', []))} capabilities)")
    except Exception as e:
        print(f"  ❌ Sidecar not reachable: {e}")
        print("  Start with: colonymind start")
        return

    # Step 2: Seed test data
    print("\n[2/5] Seeding test data...")
    test_contact = f"validate-{uuid.uuid4().hex[:6]}"

    r = httpx.post(f"{url}/v1/host/commitments", headers=headers,
        json={"person_id": test_contact, "description": "Validate E2E pipeline", "priority": 2}, timeout=5)
    if r.status_code not in (200, 201):
        print(f"  ❌ Could not create commitment: {r.status_code}")
        return
    cid = r.json().get("id")
    print(f"  ✅ Test commitment created")

    # Step 3: Context assembly
    print("\n[3/5] Testing context assembly...")
    r = httpx.post(f"{url}/v1/host/context/assemble", headers=headers,
        json={"identity": {"host_id": "validate"}, "context": {"session_id": "validate", "contact_id": test_contact},
              "incoming_message": {"role": "user", "content": "What am I working on?"}}, timeout=10)
    if r.status_code != 200:
        print(f"  ❌ Context assembly failed: {r.status_code}")
        return

    sections = r.json().get("sections", [])
    section_ids = [s["id"] for s in sections]
    expected = ["colonymind-commitments"]
    found = [e for e in expected if e in section_ids]
    print(f"  ✅ Context assembly: {len(sections)} sections, {len(found)}/{len(expected)} cognitive sections present")

    # Step 4: Check LLM is configured (live-fire the sidecar's own router —
    # this is the pipeline ColonyMind actually reasons with, harness-independent)
    print("\n[4/5] Checking LLM configuration...")
    llm_ok = False
    llm_latency = None
    try:
        r = httpx.get(f"{url}/v1/host/health/llm", headers=headers, timeout=30)
        if r.status_code == 200 and (r.json() or {}).get("ok"):
            body = r.json()
            llm_ok = True
            llm_latency = body.get("latency_ms")
            print(f"  ✅ LLM router answered (tier={body.get('tier', '?')}, "
                  f"latency={llm_latency}ms)")
        else:
            detail = ""
            try:
                detail = (r.json() or {}).get("error") or ""
            except Exception:
                pass
            print(f"  ⚠️ LLM router check failed (HTTP {r.status_code}) {detail}")
            print("  Configure models via POST /v1/host/configure or re-run "
                  "'colonymind init' (llm-config step)")
    except Exception as exc:
        print(f"  ⚠️ Could not reach the LLM health endpoint: {exc}")

    # Step 5: Full pipeline — context assembly reached the cognitive sections
    # (step 3) and the router answers (step 4); that IS the full sidecar
    # pipeline. A connected harness exercises it end-to-end on its next turn.
    print("\n[5/5] Pipeline verdict...")
    if llm_ok and found == expected:
        print("  ✅ Full sidecar pipeline working — context assembly + LLM router live")
    elif llm_ok:
        print("  ⚠️ LLM live but cognitive context sections are thin — run a few "
              "ordinary turns and check their retained evidence")
    else:
        print("  ⚪ LLM not verified — context pipeline validated, reasoning not")

    # Cleanup: delete test commitment
    if cid:
        httpx.delete(f"{url}/v1/host/commitments/{cid}", headers=headers, timeout=5)

    # Write validation stamp
    stamp_path = Path(state_dir) / ".colonymind-e2e-validated"
    stamp_data = {
        "validated_at": now_utc().isoformat(),
        "context_sections": len(sections),
        "cognitive_sections": len(found),
        "llm_tested": llm_ok,
    }
    stamp_path.write_text(json.dumps(stamp_data, indent=2))

    # Summary
    print()
    all_ok = found == expected  # the seeded commitment comes back in the assembled context
    if all_ok:
        print("🟢 Pipeline validation passed")
        print(f"  Context assembly: {len(sections)} sections")
        print(f"  Cognitive sections: {', '.join(found)}")
        if not llm_ok:
            print(f"  ⚠️ LLM pipeline not tested — configure LLM and re-run 'colonymind validate'")
    else:
        print("🔴 Pipeline validation incomplete")
        print(f"  Missing sections: {set(expected) - set(found)}")
        print(f"  Check sidecar logs and configuration")


def _configured_instance() -> bool:
    """True once ``colonymind init`` has written this instance's ``colonymind.yaml``."""
    from colonymind.config import CONFIG_FILE, instance_home
    return (instance_home() / CONFIG_FILE).is_file()


def _cmd_doctor(args) -> None:
    """Check the install: Hermes version, keys, adapter, pip check, sidecar, plugin."""
    from colonymind.config import ConfigError
    try:
        _load_dotenv()
    except ConfigError:
        pass  # check_config reports it
    from colonymind.doctor import (
        default_api_key,
        default_colonymind_url,
        exit_code,
        format_report,
        results_to_json,
        run_doctor,
    )

    url = args.url or default_colonymind_url()
    api_key = args.api_key if args.api_key is not None else default_api_key()

    if getattr(args, "fix", False):
        # Safe, idempotent config repairs only — everything else gets a
        # printed remedy instead of an automatic change.
        try:
            from colonymind.setup import repair_persisted_llm_config

            fixed = repair_persisted_llm_config()
            if fixed:
                print("🔧 Applied LLM config fixes: " + ", ".join(fixed) + "\n")
            else:
                print("🔧 No automatic fixes applicable\n")
        except Exception as exc:
            print(f"🔧 Automatic fixes unavailable: {exc}\n")

    results = run_doctor(colonymind_url=url, api_key=api_key, timeout=args.timeout)

    if args.json:
        print(json.dumps(results_to_json(results), indent=2))
    else:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        print(format_report(results, colonymind_url=url, color=color))

    sys.exit(exit_code(results))


def _load_dotenv() -> None:
    from colonymind.util.instance import load_environment
    load_environment()


if __name__ == "__main__":
    main()
