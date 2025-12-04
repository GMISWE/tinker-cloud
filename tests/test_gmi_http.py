#!/usr/bin/env python3
"""
Complete HTTP Test Suite for GMI Wrapper V3

This script tests the GMI API implementation via direct HTTP calls.
Can run from host machine against the container.

Usage:
    # Run from host (default port 8001 for mapped container)
    python tests/test_gmi_http.py

    # Or with custom URL
    GMI_BASE_URL=http://localhost:8001 python tests/test_gmi_http.py
"""
import os
import sys
import time
from typing import Any, Dict, Optional
import httpx

# Configuration
GMI_URL = os.getenv("GMI_BASE_URL", "http://localhost:8001")
API_KEY = os.getenv("GMI_API_KEY", "slime-dev-key")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/models/Qwen2.5-0.5B-Instruct_torch_dist")
POLL_INTERVAL = 2.0
MAX_WAIT_TIME = 300

headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def poll_future(req_id: str, timeout: float = MAX_WAIT_TIME) -> Dict[str, Any]:
    """Poll for future result until completed or timeout"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            resp = httpx.post(
                f"{GMI_URL}/api/v1/retrieve_future",
                json={"request_id": req_id},
                headers=headers,
                timeout=120.0
            )

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 408:
                time.sleep(POLL_INTERVAL)
                continue
            elif resp.status_code == 500:
                raise Exception(f"Operation failed: {resp.text}")
            else:
                raise Exception(f"Unexpected status {resp.status_code}: {resp.text}")

        except httpx.RequestError as e:
            raise Exception(f"Network error: {e}")

    raise TimeoutError(f"Timed out after {timeout}s")


def test_health_check() -> bool:
    """Test 0: Health Check"""
    print("\n" + "=" * 80)
    print("TEST 0: Health Check")
    print("=" * 80)

    try:
        resp = httpx.get(f"{GMI_URL}/health", timeout=10.0)

        if resp.status_code == 200:
            health = resp.json()
            print(f"✓ Health check passed:")
            print(f"  Status: {health.get('status')}")
            print(f"  Version: {health.get('version')}")
            print(f"  Ray initialized: {health.get('ray_initialized')}")
            print(f"  Active training clients: {health.get('active_training_clients')}")
            print(f"  Pending futures: {health.get('futures_count')}")
            return True
        else:
            print(f"✗ Health check failed: {resp.status_code}")
            print(f"  {resp.text}")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_capabilities() -> bool:
    """Test 0.5: Get Server Capabilities"""
    print("\n" + "=" * 80)
    print("TEST 0.5: Get Server Capabilities")
    print("=" * 80)

    try:
        resp = httpx.get(
            f"{GMI_URL}/api/v1/capabilities",
            headers=headers,
            timeout=10.0
        )

        if resp.status_code == 200:
            caps = resp.json()
            print(f"✓ Capabilities retrieved:")
            print(f"  Supported models: {len(caps.get('supported_models', []))}")
            for model in caps.get('supported_models', []):
                print(f"    - {model.get('model_name')}")
            return True
        else:
            print(f"✗ Get capabilities failed: {resp.status_code}")
            print(f"  {resp.text}")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation() -> tuple[bool, Optional[str]]:
    """Test 1: Model Creation"""
    print("\n" + "=" * 80)
    print("TEST 1: Model Creation")
    print("=" * 80)

    try:
        print(f"Creating model with base: {MODEL_PATH}")
        print("  LoRA: disabled (rank=0)")

        resp = httpx.post(
            f"{GMI_URL}/api/v1/create_model",
            json={
                "base_model": MODEL_PATH,
                "lora_config": {"rank": 0, "alpha": 0}
            },
            headers=headers,
            timeout=30.0
        )

        if resp.status_code != 200:
            print(f"✗ Create model failed: {resp.status_code}")
            print(f"  {resp.text}")
            return False, None

        result = resp.json()
        req_id = result["request_id"]
        print(f"  Submitted request: {req_id}")
        print("  Polling for completion (max 180s)...")

        # Poll for result
        for attempt in range(90):
            time.sleep(2)
            poll_resp = httpx.post(
                f"{GMI_URL}/api/v1/retrieve_future",
                json={"request_id": req_id},
                headers=headers,
                timeout=120.0
            )

            if poll_resp.status_code == 200:
                result = poll_resp.json()
                model_id = result.get("model_id")
                print(f"✓ Model created: {model_id}")
                return True, model_id
            elif poll_resp.status_code == 408:
                if attempt % 15 == 0:
                    print(f"    Still creating... ({attempt*2}s)")
            else:
                print(f"✗ Poll error: {poll_resp.status_code}")
                print(f"  {poll_resp.text}")
                return False, None

        print("✗ Timed out waiting for model creation")
        return False, None

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_save_weights(model_id: str) -> bool:
    """Test 2: Save Weights"""
    print("\n" + "=" * 80)
    print("TEST 2: Save Weights")
    print("=" * 80)

    try:
        checkpoint_path = f"/data/checkpoints/test_{int(time.time())}"
        print(f"Saving checkpoint to: {checkpoint_path}")

        resp = httpx.post(
            f"{GMI_URL}/api/v1/save_weights",
            json={
                "model_id": model_id,
                "checkpoint_path": checkpoint_path
            },
            headers=headers,
            timeout=30.0
        )

        if resp.status_code != 200:
            print(f"✗ Save weights failed: {resp.status_code}")
            print(f"  {resp.text}")
            return False

        result = resp.json()
        req_id = result["request_id"]
        print(f"  Submitted request: {req_id}")

        # Poll for result
        result = poll_future(req_id, timeout=120)
        print(f"✓ Checkpoint saved:")
        print(f"  Path: {result.get('checkpoint_path') or result.get('path')}")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_delete_model(model_id: str) -> bool:
    """Test 3: Delete Model"""
    print("\n" + "=" * 80)
    print("TEST 3: Delete Model")
    print("=" * 80)

    try:
        print(f"Deleting model: {model_id}")

        resp = httpx.post(
            f"{GMI_URL}/api/v1/delete_model",
            json={"model_id": model_id},
            headers=headers,
            timeout=30.0
        )

        if resp.status_code != 200:
            print(f"✗ Delete model failed: {resp.status_code}")
            print(f"  {resp.text}")
            return False

        result = resp.json()
        req_id = result["request_id"]
        print(f"  Submitted request: {req_id}")

        # Poll for result
        result = poll_future(req_id, timeout=60)
        print(f"✓ Model deleted")
        print(f"  Status: {result.get('status')}")

        # Verify deletion via health check
        resp = httpx.get(f"{GMI_URL}/health", timeout=10.0)
        if resp.status_code == 200:
            health = resp.json()
            active = health.get('active_training_clients', 0)
            print(f"  Active clients after deletion: {active}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run test suite"""
    print("\n" + "=" * 80)
    print("GMI WRAPPER V3 HTTP TEST SUITE")
    print("=" * 80)
    print(f"Base URL: {GMI_URL}")
    print(f"API Key: {API_KEY[:10]}...")
    print(f"Model: {MODEL_PATH}")
    print("=" * 80)

    results = []
    model_id = None

    # Test 0: Health Check
    results.append(("Health Check", test_health_check()))

    # Test 0.5: Capabilities
    results.append(("Capabilities", test_capabilities()))

    # Test 1: Model Creation
    success, model_id = test_model_creation()
    results.append(("Model Creation", success))

    if not model_id:
        print("\n" + "=" * 80)
        print("CRITICAL: Model creation failed. Skipping remaining tests.")
        print("=" * 80)
    else:
        # Test 2: Save Weights
        results.append(("Save Weights", test_save_weights(model_id)))

        # Test 3: Delete Model
        results.append(("Delete Model", test_delete_model(model_id)))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
