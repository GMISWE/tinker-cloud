"""HMAC-signed, expiring download tokens for external weights files.

There is no object store behind this server, so a "signed URL" is a route on
the server itself: ``.../external_weights/<model_id>/<name>/<relpath>?exp=<unix>&sig=<hex>``.
``sig`` is HMAC-SHA256 over (scope, model_id, name, relpath, exp) under the
server's signing key (TINKERCLOUD_URL_SIGNING_KEY), so a URL opens exactly
one file of one checkpoint until ``exp``. The URL is the credential: the
download route takes no API key.
"""
import hashlib
import hmac
import time

SCOPE = "external_weights"


def _message(model_id: str, name: str, relpath: str, exp: int) -> bytes:
    return "\n".join((SCOPE, model_id, name, relpath, str(exp))).encode()


def sign(key: str, model_id: str, name: str, relpath: str, exp: int) -> str:
    return hmac.new(key.encode(), _message(model_id, name, relpath, exp), hashlib.sha256).hexdigest()


def verify(key: str, model_id: str, name: str, relpath: str, exp: int, sig: str, now: float | None = None) -> bool:
    """True only for an untampered signature whose expiry is still ahead."""
    if (time.time() if now is None else now) >= exp:
        return False
    return hmac.compare_digest(sign(key, model_id, name, relpath, exp), sig)
