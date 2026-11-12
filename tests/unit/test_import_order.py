"""The checkpoints, storage, models and proto packages import each other one way: each can be the first
import of a process."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("package", ["checkpoints", "storage", "models", "proto"])
def test_the_package_is_the_first_import_of_a_process(package):
    """The storage package imported the models, which import the checkpoints, which import the
    storage: a process that imported the checkpoints first failed."""
    r = subprocess.run([sys.executable, "-c", f"import training.{package}"], cwd=str(REPO),
                       env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True, timeout=100)
    assert r.returncode == 0, r.stderr[-2000:]
