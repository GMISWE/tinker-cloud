"""Listening sockets the gateway and the engines bind themselves, so a stale path is handled before uvicorn sees it."""
import os
import socket


class SocketBusy(RuntimeError):
    """Another process accepts on the path."""


def bind_unix_socket(path: str, backlog: int = 4096) -> socket.socket:
    """Bind and listen on `path`. A stale file (nothing accepts) is replaced; a live listener raises SocketBusy.
    Owner-only: whoever can connect can register an engine or call one."""
    if os.path.exists(path):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(path)
        except (ConnectionRefusedError, FileNotFoundError):
            os.unlink(path)
        else:
            raise SocketBusy(f"another process listens on {path}")
        finally:
            probe.close()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    os.chmod(path, 0o600)
    sock.listen(backlog)
    return sock


def bind_tcp_socket(host: str, port: int, backlog: int = 4096) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    return sock
