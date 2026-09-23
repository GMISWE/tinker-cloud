"""
TinkerCloud training API: FastAPI app factory.

create_app(config) wires storage, the selected TrainingBackend, the services and
the routers onto app.state; `app` is the environment-configured instance for
`uvicorn training.api:app`. The CLI lives in training/__main__.py.
"""
import asyncio
import logging
import os
import signal
from typing import Optional

# CRITICAL: Disable Ray auto-init BEFORE importing ray
os.environ["RAY_DISABLE_AUTO_INIT"] = "1"

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Import modules
from .storage import FuturesStorage, MetadataStorage, SampleFutureStore, SessionStorage
from .checkpoints import CheckpointError, CheckpointStore
from .storage.futures import DuplicateSeqId
from .config import get_config, TrainingConfig
from .utils import APIKeyAuth
from .core.runtime_state import TrainingRuntimeState
# Configure logging
logger = logging.getLogger(__name__)

def create_app(config: Optional[TrainingConfig] = None) -> FastAPI:
    """Application factory for the training API."""
    application = FastAPI(title="TinkerCloud training API", version="3.1.0")

    # Store config/runtime scaffolding for later use
    application.state.config = config or get_config()
    application.state.runtime = TrainingRuntimeState()

    # Import routers lazily to avoid circular imports during module load
    from .routers import training as training_router_module
    from .routers import health as health_router_module
    from .routers import futures as futures_router_module
    from .routers import models as models_router_module
    from .routers import checkpoints as checkpoints_router_module
    from .routers import sampling as sampling_router_module
    from .routers import session as session_router_module

    # Include all routers
    application.include_router(training_router_module.router)
    application.include_router(health_router_module.router)
    application.include_router(futures_router_module.router)
    application.include_router(models_router_module.router)
    application.include_router(checkpoints_router_module.router)
    application.include_router(sampling_router_module.router)
    application.include_router(session_router_module.router)
    logger.info("✅ Modular routers integrated: training, health, futures, models, checkpoints, sampling, session")

    @application.on_event("startup")
    async def startup_event():
        """Initialize storage, Ray, and inject dependencies into routers"""
        config_obj: TrainingConfig = application.state.config

        logger.info(f"Loaded configuration - Log level: {config_obj.server.log_level}")
        logger.info(
            "Access logging: %s (set KGATEWAY_ACCESS_LOG=true to enable)",
            "enabled" if config_obj.server.access_log else "disabled",
        )

        # Set up logging based on config
        # force=True ensures this takes effect even if uvicorn already configured logging
        logging.basicConfig(
            level=config_obj.server.log_level,
            format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

        # Suppress noisy third-party loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)

        # Storage roots (a fresh pod has none of them; SQLite does not create directories)
        for d in (config_obj.storage.metadata_dir, config_obj.storage.training_runs_dir,
                  config_obj.storage.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)
        futures_storage = FuturesStorage(config_obj.storage.futures_db_path)
        metadata_storage = MetadataStorage(config_obj.storage.metadata_dir)
        config_obj.storage.checkpoint_base.mkdir(parents=True, exist_ok=True)
        checkpoint_store = CheckpointStore(config_obj.storage.checkpoint_base, metadata_storage)
        swept = checkpoint_store.sweep_pending()
        if swept:
            logger.warning("Marked %d checkpoint save(s) interrupted by the last shutdown as failed", swept)
        logger.info("Checkpoint base: %s", config_obj.storage.checkpoint_base)

        # Initialize session storage (separate DB file for sessions/samplers)
        session_db_path = config_obj.storage.metadata_dir / "sessions.db"
        session_storage = SessionStorage(session_db_path)
        logger.info("Initialized storage modules - DB: %s, Sessions: %s",
                    config_obj.storage.futures_db_path, session_db_path)

        # Clean up stale futures from previous runs (all async tasks are lost on restart)
        removed_count = futures_storage.cleanup_old_futures(max_age_hours=0)
        logger.info("Cleaned up %s stale futures from previous runs", removed_count)

        # Sample futures are in memory only (storage.sample_futures): rollout
        # results are not read back across a restart, which frees every model.
        sample_futures = SampleFutureStore()
        application.state.sample_futures = sample_futures
        application.state.sample_futures_sweeper = asyncio.create_task(sample_futures.sweep_forever())

        # Clean up stale sessions BEFORE loading into SessionService
        removed_sessions, removed_session_ids = session_storage.cleanup_stale_sessions(max_age_hours=24)
        if removed_sessions > 0:
            logger.info("Cleaned up %s stale sessions before loading", removed_sessions)

        # Initialize backend
        backend_type = config_obj.backend.backend_type
        if backend_type == "miles":
            logger.info("Backend selected: miles (default)")
        else:
            logger.info("Backend selected: %s", backend_type)

        from .backends.factory import BackendFactory
        backend = BackendFactory.create(
            backend_type=backend_type,
            backend_overrides=config_obj.backend.backend_overrides,
        )
        logger.info("Backend initialized: %s", backend_type)
        backend_cfg = backend.config
        if backend_cfg is not None:
            logger.info("Effective %s backend config (value, source):\n%s", backend_type, backend_cfg.describe())

        auth = APIKeyAuth(
            api_key=config_obj.auth.api_key,
            enabled=config_obj.auth.enabled,
        )

        # Store in app state for dependency injection
        application.state.futures_storage = futures_storage
        application.state.metadata_storage = metadata_storage
        application.state.checkpoint_store = checkpoint_store
        application.state.session_storage = session_storage
        application.state.auth = auth
        application.state.backend = backend
        application.state.backend_type = backend_type

        # Initialize service singletons (services that depend on backend get it injected)
        from .services.model_service import ModelService
        from .services.training_service import TrainingService
        from .services.checkpoint_service import CheckpointService
        from .services.sampling_service import SamplingService
        from .services.session_service import SessionService

        application.state.model_service = ModelService(
            backend=backend, store=checkpoint_store, sample_futures=sample_futures,
        )

        application.state.training_service = TrainingService(backend=backend)
        application.state.checkpoint_service = CheckpointService(backend=backend, store=checkpoint_store)
        application.state.sampling_service = SamplingService(backend=backend)
        # Initialize SessionService with storage for persistence
        application.state.session_service = SessionService(storage=session_storage)

        logger.info("✅ Dependency providers registered on app state")

        # Session reaper: a session silent longer than session_timeout_s is
        # expired and its models freed (the SDK heartbeats every 10 s).
        application.state.session_reaper = None
        if config_obj.session_timeout_s >= 0:
            application.state.session_reaper = asyncio.create_task(
                _reap_sessions_forever(application, config_obj.session_timeout_s,
                                       config_obj.session_reap_interval_s)
            )
            logger.info("Session reaper enabled: timeout=%ss interval=%ss",
                        config_obj.session_timeout_s, config_obj.session_reap_interval_s)

        # Initialize Ray
        if not backend.needs_ray:
            logger.info("Backend %s does not use Ray; skipping ray.init", backend_type)
        elif not ray.is_initialized():
            logger.info("Initializing Ray with address=%s", config_obj.ray.address)
            # ray.init installs a SIGTERM handler that sys.exit()s, pre-empting
            # uvicorn's graceful shutdown (and with it the model teardown above).
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            try:
                ray.init(
                    address=config_obj.ray.address,
                    namespace=config_obj.ray.namespace,
                    ignore_reinit_error=True,
                )
                logger.info("Ray initialized successfully")
            finally:
                if signal.getsignal(signal.SIGTERM) is not prev_sigterm:
                    signal.signal(signal.SIGTERM, prev_sigterm)

    @application.on_event("shutdown")
    async def shutdown_event():
        """Free every live model (actors, placement groups) before the process exits."""
        reaper = application.state.session_reaper
        if reaper is not None:
            reaper.cancel()
        application.state.sample_futures_sweeper.cancel()
        runtime: TrainingRuntimeState = application.state.runtime
        for model_id in list(runtime.training_clients):
            await _free_model(application, model_id, reason="shutdown")
        await application.state.backend.close()
        if ray.is_initialized():
            ray.shutdown()
        application.state.futures_storage.close()

    @application.exception_handler(DuplicateSeqId)
    async def duplicate_seq_id_handler(request: Request, exc: DuplicateSeqId):
        """A reused (model_id, seq_id) with a different request is a client bug.

        400 rather than 409: the SDK retries 408/409/429/5xx for hours, and a
        conflict can never succeed on retry."""
        return JSONResponse(status_code=400, content={"error": str(exc), "status_code": 400,
                                                      "path": str(request.url.path)})

    @application.exception_handler(CheckpointError)
    async def checkpoint_error_handler(request: Request, exc: CheckpointError):
        """Store verdicts on a checkpoint identity: 400 malformed / wrong kind,
        404 unknown, 425 still being written, 500 the save failed."""
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc), "status_code": exc.status_code,
                                                                  "path": str(request.url.path)})

    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured responses"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )

    @application.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if os.getenv("DEBUG") else None
            }
        )

    return application


async def _free_model(application: FastAPI, model_id: str, reason: str) -> None:
    """Release one model's backend resources and its session link; never raises."""
    runtime: TrainingRuntimeState = application.state.runtime
    try:
        logger.info("Freeing model %s (%s)", model_id, reason)
        await application.state.model_service.delete_model(
            model_id=model_id,
            training_clients=runtime.training_clients,
            metadata_storage=application.state.metadata_storage,
        )
    except Exception:  # pylint: disable=broad-except
        logger.exception("Failed to free model %s (%s)", model_id, reason)
    try:
        application.state.session_service.remove_model(model_id)
    except Exception:  # pylint: disable=broad-except
        logger.exception("Failed to unlink model %s from its session", model_id)


async def _reap_sessions_forever(application: FastAPI, timeout_s: float, interval_s: float) -> None:
    session_service = application.state.session_service
    while True:
        await asyncio.sleep(interval_s)
        try:
            for session_id, model_ids in session_service.reap_stale_sessions(timeout_s):
                logger.warning("Session %s silent > %ss: expired, freeing %d model(s)",
                               session_id, timeout_s, len(model_ids))
                for model_id in model_ids:
                    if model_id in application.state.runtime.training_clients:
                        await _free_model(application, model_id, reason=f"session {session_id} expired")
        except Exception:  # pylint: disable=broad-except
            logger.exception("Session reaper tick failed")


app = create_app()

if __name__ == "__main__":  # `python -m training.api` == `python -m training`
    from .__main__ import main
    main()
