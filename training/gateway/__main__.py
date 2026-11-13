"""The gateway process: `python -m training.gateway [--host H] [--port P] [--engines-socket S] [--workers N] [--client aiohttp|httpx]`.

One uvicorn server per worker on two shared listeners: the SDK's TCP port and the engines'
unix socket. With N workers the parent binds the sockets, forks N children that each serve
them, forwards SIGTERM and SIGINT to the children, and exits when they have.
"""
import argparse
import os
import signal
import sys
import traceback
from typing import List

import uvicorn

from ..config import TrainingConfig, get_config, set_config
from .app import build_app
from .sockets import SocketBusy, bind_tcp_socket, bind_unix_socket

# a pooled engine connection can sit idle behind a deep queue; uvicorn's 5 s default would close it
KEEPALIVE_S = 300


def _positive(arg: str) -> int:
    value = int(arg)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{arg}: must be at least 1")
    return value


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m training.gateway", description="TinkerCloud gateway")
    parser.add_argument("--host", default=None, help="bind host (default: TRAINING_HOST or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: TRAINING_PORT or 8000)")
    parser.add_argument("--engines-socket", default=None, help="unix socket engines register on (default: TINKERCLOUD_ENGINES_SOCKET)")
    parser.add_argument("--workers", type=_positive, default=None, help="worker processes (default: TINKERCLOUD_GATEWAY_WORKERS or 1)")
    parser.add_argument("--client", choices=["aiohttp", "httpx"], default=None, help="engine-facing HTTP client (httpx: the measurement baseline)")
    parser.add_argument("--pool", type=_positive, default=None, help="max connections to one engine, per worker")
    args = parser.parse_args(argv)

    config = get_config()
    if args.host is not None:
        config.server.host = args.host
    if args.port is not None:
        config.server.port = args.port
    if args.engines_socket is not None:
        config.gateway.engines_socket = args.engines_socket
    config.gateway.engines_socket = os.path.abspath(config.gateway.engines_socket)   # the listener is told apart by its path
    if args.workers is not None:
        config.gateway.workers = args.workers
    if args.client is not None:
        config.gateway.client = args.client
    if args.pool is not None:
        config.gateway.pool_size = args.pool
    set_config(config)

    try:
        engines = bind_unix_socket(config.gateway.engines_socket)
    except SocketBusy as exc:
        print(f"gateway: {exc}", file=sys.stderr)
        sys.exit(1)
    sdk = bind_tcp_socket(config.server.host, config.server.port)
    if config.gateway.workers == 1:
        serve(config, [sdk, engines])
        return
    sys.exit(supervise(config, [sdk, engines]))


def serve(config: TrainingConfig, sockets: List) -> None:
    """One worker: the app is built here, after any fork, so each worker owns its connections."""
    server = uvicorn.Server(uvicorn.Config(
        build_app(config), log_level=config.server.log_level.lower(), access_log=config.server.access_log,
        timeout_keep_alive=KEEPALIVE_S,
    ))
    server.run(sockets=sockets)


def supervise(config: TrainingConfig, sockets: List) -> int:
    """Fork `workers` children over the bound sockets; pass SIGTERM/SIGINT on; when all have exited,
    return 0, or 1 when any worker did not exit 0 (each such worker is named on stderr)."""
    children: List[int] = []
    for _ in range(config.gateway.workers):
        pid = os.fork()
        if pid == 0:
            try:
                serve(config, sockets)
            except BaseException:   # the child never returns into the parent's loop: its status and traceback are what it leaves  # noqa: BLE001  (#192)
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        children.append(pid)

    def relay(signum, _) -> None:
        for pid in children:
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, relay)
    failed = 0
    for pid in children:
        code = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])   # negative: killed by that signal
        if code != 0:
            failed += 1
            print(f"gateway worker {pid} exited {code}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    main()
