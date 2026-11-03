"""The gateway process: `python -m training.gateway [--host H] [--port P] [--engines-socket S] [--server asgi|fastapi] [--client h11|httpx]`.

One uvicorn server on two listeners: the SDK's TCP port and the engines' unix socket.
"""
import argparse
import os
import sys

import uvicorn

from ..config import get_config, set_config
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
    parser.add_argument("--server", choices=["asgi", "fastapi"], default=None, help="SDK-facing server adapter")
    parser.add_argument("--client", choices=["h11", "httpx"], default=None, help="engine-facing HTTP client")
    parser.add_argument("--pool", type=_positive, default=None, help="max connections to one engine")
    args = parser.parse_args(argv)

    config = get_config()
    if args.host is not None:
        config.server.host = args.host
    if args.port is not None:
        config.server.port = args.port
    if args.engines_socket is not None:
        config.gateway.engines_socket = args.engines_socket
    config.gateway.engines_socket = os.path.abspath(config.gateway.engines_socket)   # the listener is told apart by its path
    if args.server is not None:
        config.gateway.server = args.server
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
    server = uvicorn.Server(uvicorn.Config(
        build_app(config), log_level=config.server.log_level.lower(), access_log=config.server.access_log,
        timeout_keep_alive=KEEPALIVE_S,
    ))
    server.run(sockets=[sdk, engines])


if __name__ == "__main__":
    main()
