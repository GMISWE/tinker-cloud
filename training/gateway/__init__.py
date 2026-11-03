"""The gateway process: the SDK's HTTP face, forwarding every request to an engine.

Nothing here imports a backend, Ray, torch or numpy (tests/unit/gateway/test_imports.py
pins the import set), so the gateway is a separate, light process from the engines.
"""
