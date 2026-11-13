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
from .storage import FuturesStorage, MetadataStorage, SampleFutureStore
from .checkpoints import CheckpointError, CheckpointStore
from .storage.futures import DuplicateSeqId
from .config import get_config, TrainingConfig
from .utils import APIKeyAuth
from .core.runtime_state import TrainingRuntimeState
from .gateway.reaper import ModelReaper
from .utils.helpers import set_engine_name
# Configure logging
logger = logging.getLogger(__name__)

def create_app(config: Optional[TrainingConfig] = None) -> FastAPI:
    """Application factory for the training API."""
    application = FastAPI(title="TinkerCloud training API", version="3.1.0")

    # Store config/runtime scaffolding for later use
    application.state.config = config or get_config()
    application.state.runtime = TrainingRuntimeState()
    # the engine's client to its gateway (register, heartbeat, samplers); None until the entry point sets it
    application.state.gateway_client = None
    # ids carry the engine name; the single process is the engine named after its backend
    cfg: TrainingConfig = application.state.config
    set_engine_name(cfg.engine.name if cfg.engine.name is not None else cfg.backend.backend_type)

    # Import routers lazily to avoid circular imports during module load
    from .routers import training as training_router_module
    from .routers import health as health_router_module
    from .routers import futures as futures_router_module
    from .routers import models as models_router_module
    from .routers import checkpoints as checkpoints_router_module
    from .routers import sampling as sampling_router_module
    from .routers import billing as billing_router_module

    # Include all routers
    application.include_router(training_router_module.router)
    application.include_router(health_router_module.router)
    application.include_router(futures_router_module.router)
    application.include_router(models_router_module.router)
    application.include_router(checkpoints_router_module.router)
    application.include_router(sampling_router_module.router)
    application.include_router(billing_router_module.router)
    logger.info("✅ Modular routers integrated: training, health, futures, models, checkpoints, sampling, billing")

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
        metadata_storage.sweep_temp_files()   # before any writer runs
        swept = checkpoint_store.sweep_pending()
        if swept:
            logger.warning("Marked %d checkpoint save(s) interrupted by the last shutdown as failed", swept)
        application.state.checkpoint_sweeper = asyncio.create_task(checkpoint_store.sweep_forever())
        logger.info("Checkpoint base: %s", config_obj.storage.checkpoint_base)

        logger.info("Initialized storage modules - DB: %s", config_obj.storage.futures_db_path)

        # Clean up stale futures from previous runs (all async tasks are lost on restart)
        removed_count = futures_storage.purge_all()
        logger.info("Cleaned up %s stale futures from previous runs", removed_count)

        # Sample futures are in memory only (storage.sample_futures): rollout
        # results are not read back across a restart, which frees every model.
        sample_futures = SampleFutureStore()
        application.state.sample_futures = sample_futures
        application.state.sample_futures_sweeper = asyncio.create_task(sample_futures.sweep_forever())

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

        auth = APIKeyAuth(config_obj.auth)

        # Store in app state for dependency injection
        application.state.futures_storage = futures_storage
        application.state.metadata_storage = metadata_storage
        application.state.checkpoint_store = checkpoint_store
        application.state.auth = auth
        application.state.backend = backend
        application.state.backend_type = backend_type

        # Initialize service singletons (services that depend on backend get it injected)
        from .services.model_service import ModelService
        from .services.training_service import TrainingService
        from .services.checkpoint_service import CheckpointService
        from .services.sampling_service import SamplingService

        application.state.model_service = ModelService(
            backend=backend, store=checkpoint_store, sample_futures=sample_futures,
        )

        application.state.training_service = TrainingService(backend=backend)
        application.state.checkpoint_service = CheckpointService(backend=backend, store=checkpoint_store)
        application.state.sampling_service = SamplingService(backend=backend)

        logger.info("✅ Dependency providers registered on app state")

        # Sessions and their reaper live in the gateway; this engine frees a model on the
        # gateway's delete, and every live model at shutdown, each in its own task.
        application.state.model_reaper = ModelReaper(lambda model_id, reason: _free_model(application, model_id, reason))

        # Initialize Ray
        if not backend.needs_ray:
            logger.info("Backend %s does not use Ray; skipping ray.init", backend_type)
        elif not ray.is_initialized():
            # ray.init installs a SIGTERM handler that sys.exit()s, pre-empting
            # uvicorn's graceful shutdown (and with it the model teardown above).
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            try:
                if config_obj.engine.ray_local:
                    # "local" starts a head here even when RAY_ADDRESS names a shared one
                    ctx = ray.init(
                        address="local",
                        namespace=config_obj.ray.namespace,
                        num_gpus=config_obj.engine.num_gpus,
                        _temp_dir=config_obj.engine.ray_temp_dir,
                        ignore_reinit_error=True,
                    )
                    logger.info("Ray head started in-process: gcs=%s temp_dir=%s",
                                ctx.address_info["gcs_address"], config_obj.engine.ray_temp_dir)
                else:
                    logger.info("Initializing Ray with address=%s", config_obj.ray.address)
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
        application.state.checkpoint_sweeper.cancel()
        application.state.sample_futures_sweeper.cancel()
        runtime: TrainingRuntimeState = application.state.runtime
        # a free stuck on a hung op is cancelled first, then every live model is freed at once
        await application.state.model_reaper.shutdown(list(runtime.training_clients))
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
        404 unknown, 425 still being written, 500 the save or a delete failed."""
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
    """Release one model's backend resources; never raises."""
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


app = create_app()

if __name__ == "__main__":  # `python -m training.api` == `python -m training`
    from .__main__ import main
    main()
