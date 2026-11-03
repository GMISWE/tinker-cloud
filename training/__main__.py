"""TinkerCloud server entrypoint: `python -m training [--backend B] [--host H] [--port P]`.

One process serving today's deployments: the gateway on the TCP port and the engine, named after
its backend, on a unix socket under METADATA_DIR, registered with the gateway over a second
unix socket there. The split form is `python -m training.gateway` plus `python -m training.engine`.

Backend selection is explicit (Constitution P3): the flag, else TINKERCLOUD_BACKEND,
else the config default. Everything else comes from the environment via
`training.config.TrainingConfig` and the per-backend config models under
`training/backends/<backend>/config.py`.
"""
import argparse
import asyncio
import os
import sys

import uvicorn

from .backends.factory import SUPPORTED_BACKENDS
from .config import TrainingConfig, get_config, set_config
from .engine.__main__ import serve as serve_engine
from .gateway.__main__ import KEEPALIVE_S
from .gateway.app import build_app
from .gateway.sockets import SocketBusy, bind_tcp_socket, bind_unix_socket


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m training", description="TinkerCloud training API")
    parser.add_argument("--backend", choices=list(SUPPORTED_BACKENDS), default=None,
                        help="training backend (default: TINKERCLOUD_BACKEND, else miles)")
    parser.add_argument("--host", default=None, help="bind host (default: TRAINING_HOST or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: TRAINING_PORT or 8000)")
    args = parser.parse_args(argv)

    config = get_config()
    if args.backend:
        config.backend.backend_type = args.backend
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    metadata_dir = config.storage.metadata_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    config.engine.name = config.engine.name or config.backend.backend_type
    config.engine.socket = os.path.abspath(config.engine.socket or str(metadata_dir / "engine.sock"))
    config.gateway.engines_socket = os.path.abspath(config.engine.gateway_socket or str(metadata_dir / "gateway.sock"))
    config.engine.gateway_socket = config.gateway.engines_socket
    config.gateway.workers = 1
    set_config(config)

    try:
        engines = bind_unix_socket(config.gateway.engines_socket)
        engine = bind_unix_socket(config.engine.socket)
    except SocketBusy as exc:
        print(f"training: {exc}", file=sys.stderr)
        sys.exit(1)
    sdk = bind_tcp_socket(config.server.host, config.server.port)
    asyncio.run(serve_single(config, sdk, engines, engine))


async def serve_single(config: TrainingConfig, sdk, engines, engine) -> None:
    """The gateway and the engine in one event loop; a signal stops the engine first (it unregisters),
    then the gateway."""
    gateway = uvicorn.Server(uvicorn.Config(
        build_app(config), log_level=config.server.log_level.lower(), access_log=config.server.access_log,
        timeout_keep_alive=KEEPALIVE_S,
    ))
    serving = asyncio.create_task(gateway.serve(sockets=[sdk, engines]))
    try:
        await serve_engine(config, engine, also_stop=[gateway])
    finally:
        gateway.should_exit = True
        await serving


if __name__ == "__main__":
    main()
