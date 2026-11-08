"""The gateway stays light: no Ray, torch, numpy, transformers or vllm, and no backend module, after serving every route."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_gateway_imports_nothing_heavy():
    r = subprocess.run([sys.executable, "-m", "tests.unit.gateway._light_probe"], cwd=str(REPO),
                       capture_output=True, text=True, timeout=100)
    assert r.returncode == 0 and r.stdout.strip().endswith("LIGHT"), f"rc={r.returncode}\n{r.stdout}\n{r.stderr[-3000:]}"


def test_gateway_module_graph_has_no_backend_import():
    src = "".join(p.read_text() for p in (REPO / "training" / "gateway").glob("*.py"))
    for forbidden in ("training.backends", "from ..backends", "import ray", "import torch", "import numpy", "from ..proto", "from ..routers", "from ..services"):
        assert forbidden not in src, forbidden
