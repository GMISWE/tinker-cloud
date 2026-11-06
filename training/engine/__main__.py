"""One engine: `python -m training.engine --name N --backend B [--socket P] [--gateway-socket G] [--num-gpus K]`.

The engine serves on its unix socket only, starts its own Ray head when the backend
needs Ray, and registers with the gateway once the socket accepts.
"""
import argparse
import asyncio
import signal
import sys
from typing import Optional, Sequence

import uvicorn

from ..api import create_app
from ..backends.factory import SUPPORTED_BACKENDS
from ..config import TrainingConfig, get_config, set_config
from ..gateway.sockets import SocketBusy, bind_unix_socket
from ..utils.helpers import valid_engine_name
from .register import RegisterClient

SOCKET_DIR = "/run/tinkercloud"
# a gateway queue can hold a pooled connection idle for seconds; uvicorn's 5 s default would close it
KEEPALIVE_S = 300


def _gpus(arg: str) -> list:
    try:
        return [int(g) for g in arg.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--gpus {arg!r}: comma-separated integers") from exc


def _name(arg: str) -> str:
    try:
        return valid_engine_name(arg)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m training.engine", description="TinkerCloud engine")
    parser.add_argument("--name", required=True, type=_name,
                        help="engine name (lowercase letters, digits, underscores); every request and model id carries it")
    parser.add_argument("--backend", choices=list(SUPPORTED_BACKENDS), default=None,
                        help="training backend (default: TINKERCLOUD_BACKEND, else miles)")
    parser.add_argument("--socket", default=None, help=f"unix socket to serve on (default: {SOCKET_DIR}/<name>.sock)")
    parser.add_argument("--gateway-socket", default=None, help="gateway's engines socket to register with (default: none)")
    parser.add_argument("--heartbeat-interval", type=float, default=None, help="seconds between heartbeats")
    parser.add_argument("--num-gpus", type=int, default=None, help="GPUs the in-process Ray head owns (default: Ray detects)")
    parser.add_argument("--gpus", type=_gpus, default=None,
                        help="GPU indices this engine states in its registration, e.g. 0,1,2,3 (default: not stated)")
    parser.add_argument("--ray-temp-dir", default=None, help="Ray temp dir (default: /tmp/ray-<name>)")
    args = parser.parse_args(argv)

    config = get_config()
    if args.backend:
        config.backend.backend_type = args.backend
    config.engine.name = args.name
    config.engine.socket = args.socket or config.engine.socket or f"{SOCKET_DIR}/{args.name}.sock"
    config.engine.gateway_socket = args.gateway_socket or config.engine.gateway_socket
    if args.heartbeat_interval is not None:
        config.engine.heartbeat_interval_s = args.heartbeat_interval
    config.engine.ray_local = True
    config.engine.num_gpus = args.num_gpus
    config.engine.gpus = args.gpus
    config.engine.ray_temp_dir = args.ray_temp_dir or f"/tmp/ray-{args.name}"
    set_config(config)

    try:
        sock = bind_unix_socket(config.engine.socket)
    except SocketBusy as exc:
        print(f"engine {args.name}: {exc}", file=sys.stderr)
        sys.exit(1)
    asyncio.run(serve(config, sock))


async def serve(config: TrainingConfig, sock, also_stop: Sequence[uvicorn.Server] = ()) -> None:
    """Serve the engine until SIGTERM/SIGINT; then unregister, stop, and flag the servers in
    `also_stop` (a gateway sharing this process) to stop after it."""
    name, socket_path = config.engine.name, config.engine.socket
    if name is None or socket_path is None:
        raise RuntimeError("an engine needs a name and a socket")
    app = create_app(config)
    register: Optional[RegisterClient] = None
    if config.engine.gateway_socket:
        register = RegisterClient(config.engine, config.auth, config.backend.backend_type)
    app.state.gateway_client = register   # the routers announce the samplers they mint through it
    server = uvicorn.Server(uvicorn.Config(
        app, uds=socket_path, log_level=config.server.log_level.lower(),
        access_log=config.server.access_log, timeout_keep_alive=KEEPALIVE_S,
    ))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    await asyncio.sleep(0)   # serve() has begun: uvicorn installed its signal handlers; ours go on top of them
    stopping = False

    async def stop() -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        if register is not None:
            await register.close()   # leave the gateway before the models go: no request lands on a dying engine
        server.should_exit = True

    # Ours, not uvicorn's: a signal uvicorn captured is re-raised when serve() returns and ends the
    # process before anything after it runs; with these installed it captures none.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: loop.create_task(stop()))
    while not server.started and not serving.done():
        await asyncio.sleep(0.05)
    if serving.done():      # startup failed, or a signal came first; serve() carries any error
        await serving
        return
    if register is not None and not stopping:   # the socket accepts: now the gateway may route to us
        register.start(app.state.backend.serves_base_model_sampling)
    await serving
    if register is not None:
        await register.close()
    for other in also_stop:
        other.should_exit = True


if __name__ == "__main__":
    main()
