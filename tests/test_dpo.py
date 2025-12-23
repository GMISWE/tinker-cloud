"""
DPO (Direct Preference Optimization) integration tests.

These tests verify that the opentinker-miles server correctly handles
DPO training workflows, including:
- Forward pass for reference model logprobs
- Forward-backward pass for policy model
- Optimizer step

Prerequisites:
    - opentinker-miles server running on localhost:8000
    - Model available at /data/models/Qwen2.5-0.5B-Instruct
    - tinker-cookbook installed

Usage:
    # Run all DPO tests
    pytest tests/test_dpo.py -v

    # Run specific test
    pytest tests/test_dpo.py::test_dpo_reduced -v

    # Run directly
    python tests/test_dpo.py
"""

import os
import subprocess
import sys

# Test configuration
DEFAULT_MODEL_PATH = "/data/models/Qwen2.5-0.5B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "slime-dev-key"


def check_prerequisites():
    """Check that prerequisites are met."""
    import requests

    # Check model exists
    model_path = os.environ.get("TEST_MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"Model not found at {model_path}. "
            f"Download with: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir {model_path}"
        )

    # Check server is running
    base_url = os.environ.get("TINKER_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("TINKER_API_KEY", DEFAULT_API_KEY)
    try:
        resp = requests.get(
            f"{base_url}/health",
            headers={"X-API-Key": api_key},
            timeout=5
        )
        if resp.status_code != 200 or resp.json().get("status") != "healthy":
            raise RuntimeError("Server not healthy")
    except Exception as e:
        raise RuntimeError(
            f"opentinker-miles server not running on {base_url}: {e}"
        )


def run_cleanup():
    """Run cleanup to free any existing models."""
    base_url = os.environ.get("TINKER_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("TINKER_API_KEY", DEFAULT_API_KEY)

    cleanup_script = os.path.join(os.path.dirname(__file__), "cleanup_test_env.py")
    if os.path.exists(cleanup_script):
        env = os.environ.copy()
        env["TINKER_BASE_URL"] = base_url
        env["TINKER_API_KEY"] = api_key
        subprocess.run([sys.executable, cleanup_script], env=env, capture_output=True)


def test_dpo_reduced():
    """
    Test DPO training with reduced max_length (128 tokens).

    This test runs a few steps of DPO training to verify:
    1. Model creation works
    2. Forward pass (reference logprobs) works
    3. Forward-backward pass (policy training) works
    4. Optimizer step works
    5. Offload/onload coordination works correctly
    """
    check_prerequisites()
    run_cleanup()

    model_path = os.environ.get("TEST_MODEL_PATH", DEFAULT_MODEL_PATH)
    base_url = os.environ.get("TINKER_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("TINKER_API_KEY", DEFAULT_API_KEY)
    log_path = "/tmp/test-dpo-reduced"
    num_steps = 2

    env = os.environ.copy()
    env.update({
        "TINKER_API_KEY": api_key,
        "HF_DATASETS_OFFLINE": "0",
        "HF_HUB_OFFLINE": "0",
        "HF_DATASETS_CACHE": "/data/datasets",
        "HF_HOME": "/data",
        "PYTHONUNBUFFERED": "1",
    })

    cmd = [
        sys.executable, "-m", "tinker_cookbook.recipes.preference.dpo.train",
        f"model_name={model_path}",
        "renderer_name=qwen3",
        f"base_url={base_url}",
        "dataset=hhh",
        "batch_size=4",
        "learning_rate=1e-5",
        "dpo_beta=0.1",
        "max_length=128",
        f"n_batches={num_steps}",
        f"log_path={log_path}",
        "behavior_if_log_dir_exists=delete",
    ]

    print(f"Running DPO test with {num_steps} steps...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )

    # Print output for debugging
    if result.stdout:
        print("STDOUT:", result.stdout[-2000:])  # Last 2000 chars
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    # Check for success
    assert result.returncode == 0, f"DPO training failed with code {result.returncode}"

    # Check metrics file was created
    metrics_file = os.path.join(log_path, "metrics.jsonl")
    assert os.path.exists(metrics_file), f"Metrics file not found at {metrics_file}"

    print(f"DPO test passed! Logs at {log_path}")


def test_dpo_simple():
    """
    Test DPO training with simple configuration (256 tokens).

    Similar to test_dpo_reduced but with slightly longer sequences.
    """
    check_prerequisites()
    run_cleanup()

    model_path = os.environ.get("TEST_MODEL_PATH", DEFAULT_MODEL_PATH)
    base_url = os.environ.get("TINKER_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("TINKER_API_KEY", DEFAULT_API_KEY)
    log_path = "/tmp/test-dpo-simple"
    num_steps = 2

    env = os.environ.copy()
    env.update({
        "TINKER_API_KEY": api_key,
        "HF_DATASETS_OFFLINE": "0",
        "HF_HUB_OFFLINE": "0",
        "HF_DATASETS_CACHE": "/data/datasets",
        "HF_HOME": "/data",
        "PYTHONUNBUFFERED": "1",
    })

    cmd = [
        sys.executable, "-m", "tinker_cookbook.recipes.preference.dpo.train",
        f"model_name={model_path}",
        "renderer_name=qwen3",
        f"base_url={base_url}",
        "dataset=hhh",
        "batch_size=2",
        "learning_rate=1e-5",
        "dpo_beta=0.1",
        "max_length=256",
        f"n_batches={num_steps}",
        f"log_path={log_path}",
        "behavior_if_log_dir_exists=delete",
    ]

    print(f"Running DPO simple test with {num_steps} steps...")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.stdout:
        print("STDOUT:", result.stdout[-2000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    assert result.returncode == 0, f"DPO training failed with code {result.returncode}"

    metrics_file = os.path.join(log_path, "metrics.jsonl")
    assert os.path.exists(metrics_file), f"Metrics file not found at {metrics_file}"

    print(f"DPO simple test passed! Logs at {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DPO integration tests")
    parser.add_argument(
        "--test",
        choices=["reduced", "simple", "all"],
        default="reduced",
        help="Which test to run"
    )
    args = parser.parse_args()

    if args.test == "reduced" or args.test == "all":
        print("=" * 50)
        print("Running test_dpo_reduced")
        print("=" * 50)
        test_dpo_reduced()

    if args.test == "simple" or args.test == "all":
        print("=" * 50)
        print("Running test_dpo_simple")
        print("=" * 50)
        test_dpo_simple()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
