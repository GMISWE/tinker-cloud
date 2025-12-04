#!/usr/bin/env python3
"""
Test 1: Model Creation
Tests GMI wrapper's ability to create a model and initialize Slime RayTrainGroup actors.
"""
import os
import sys
import time
import httpx

# Configuration - use environment variables or defaults for local testing
GMI_URL = os.getenv("GMI_BASE_URL", "http://localhost:8001")
API_KEY = os.getenv("GMI_API_KEY", "slime-dev-key")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/models/Qwen2.5-0.5B-Instruct_torch_dist")
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

print("=" * 80)
print("GMI Wrapper - Test 1: Model Creation")
print("=" * 80)
print(f"URL: {GMI_URL}")
print(f"API Key: {API_KEY[:10]}...")
print(f"Model: {MODEL_PATH}")
print("=" * 80)

try:
    # Step 0: Health check
    print("\n[0/2] Health Check")
    try:
        response = httpx.get(f"{GMI_URL}/health", timeout=10.0)
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Status: {health.get('status')}")
            print(f"  Ray initialized: {health.get('ray_initialized')}")
            print(f"  Active clients (before): {health.get('active_training_clients', 0)}")
        else:
            print(f"⚠ Health check returned {response.status_code}")
    except Exception as e:
        print(f"⚠ Health check failed: {e}")
        print("  Continuing anyway...")

    # Step 1: Create model
    print("\n[1/2] Creating Model")
    print(f"  Base model: {MODEL_PATH}")
    print("  LoRA: disabled (rank=0, alpha=0)")

    response = httpx.post(
        f"{GMI_URL}/api/v1/create_model",
        json={
            "base_model": MODEL_PATH,
            "lora_config": {"rank": 0, "alpha": 0}
        },
        headers=headers,
        timeout=30.0
    )

    if response.status_code != 200:
        print(f"✗ Create model failed: {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)

    result = response.json()
    req_id = result["request_id"]
    print(f"✓ Submitted: {req_id}")

    # Poll for model creation completion (max 180 seconds)
    print("  Polling for model creation (max 180 seconds)...")
    model_id = None

    for attempt in range(90):
        time.sleep(2)
        poll_response = httpx.post(
            f"{GMI_URL}/api/v1/retrieve_future",
            json={"request_id": req_id},
            headers=headers,
            timeout=120.0
        )

        if poll_response.status_code == 200:
            result = poll_response.json()
            model_id = result.get("model_id")
            print(f"✓ Model created: {model_id}")
            print(f"  Base model: {result.get('base_model')}")
            print(f"  Status: {result.get('status')}")
            break
        elif poll_response.status_code == 408:
            if attempt % 10 == 0:
                print(f"  Still creating... ({attempt+1}/90)")
        else:
            print(f"✗ Error: {poll_response.status_code}")
            print(f"  {poll_response.text}")
            sys.exit(1)

    if not model_id:
        print("✗ Timed out waiting for model creation (180 seconds)")
        sys.exit(1)

    # Step 2: Verify state
    print("\n[2/2] Verifying State")
    try:
        response = httpx.get(f"{GMI_URL}/health", timeout=10.0)
        if response.status_code == 200:
            health = response.json()
            active_clients = health.get('active_training_clients', 0)
            pending_futures = health.get('futures_count', 0)
            print(f"✓ Active training clients: {active_clients}")
            print(f"  Pending futures: {pending_futures}")

            if active_clients != 1:
                print(f"⚠ Warning: Expected 1 active client, got {active_clients}")
        else:
            print(f"⚠ Health check returned {response.status_code}")
    except Exception as e:
        print(f"⚠ Final health check failed: {e}")

    # Success!
    print("\n" + "=" * 80)
    print("✓ TEST 1 PASSED")
    print("=" * 80)
    print(f"\nModel successfully created: {model_id}")
    print("Slime RayTrainGroup actors are initialized and ready.")
    print("=" * 80)

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 80)
    print("✗ TEST 1 FAILED")
    print("=" * 80)

    sys.exit(1)
