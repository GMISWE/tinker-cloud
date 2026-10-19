"""Every test under tests/gpu carries the `gpu` marker (registered in pyproject)."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.gpu)
