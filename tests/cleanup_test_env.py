#!/usr/bin/env python3
"""
Cleanup script for test environment

This script performs cleanup operations:
1. Delete all active models (frees GPU resources via Ray)
2. Clear old futures from database (removes stale request state)

Usage:
    # From host machine (default port 8001 for mapped container)
    python tests/cleanup_test_env.py

    # With custom URL
    GMI_BASE_URL=http://localhost:8001 python tests/cleanup_test_env.py

Environment variables:
    GMI_BASE_URL - API URL (default: http://localhost:8001)
    GMI_API_KEY - API key (default: slime-dev-key)
"""
import requests
import os
import sys
import time

base_url = os.environ.get("GMI_BASE_URL", "http://localhost:8001")
api_key = os.environ.get("GMI_API_KEY", "slime-dev-key")
headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

print("=" * 80)
print("CLEANUP TEST ENVIRONMENT")
print("=" * 80)
print(f"Base URL: {base_url}")
print("=" * 80)

# Step 1: Get list of active models and delete them
print("\n[1/3] Checking for active models...")
try:
    response = requests.get(
        f"{base_url}/health",
        headers=headers,
        timeout=10
    )

    if response.status_code == 200:
        health = response.json()
        model_ids = health.get("model_ids", [])
        active_models = len(model_ids)
        print(f"   Active training clients: {active_models}")

        if active_models == 0:
            print("   ✓ No active models to cleanup")
        else:
            print(f"   ⚠ Warning: {active_models} active models found")
            print(f"   Model IDs: {model_ids}")

            # Delete each model
            for model_id in model_ids:
                print(f"   Deleting model: {model_id}...")
                try:
                    del_response = requests.post(
                        f"{base_url}/api/v1/delete_model",
                        json={"model_id": model_id},
                        headers=headers,
                        timeout=30
                    )

                    if del_response.status_code == 200:
                        result = del_response.json()
                        req_id = result.get("request_id")
                        print(f"     Submitted delete request: {req_id}")

                        # Poll for completion
                        for attempt in range(30):
                            time.sleep(2)
                            poll_resp = requests.post(
                                f"{base_url}/api/v1/retrieve_future",
                                json={"request_id": req_id},
                                headers=headers,
                                timeout=30
                            )
                            if poll_resp.status_code == 200:
                                print(f"   ✓ Deleted model: {model_id}")
                                break
                            elif poll_resp.status_code == 408:
                                continue
                            else:
                                print(f"   ✗ Delete poll error: {poll_resp.status_code}")
                                break
                    else:
                        print(f"   ✗ Failed to delete {model_id}: {del_response.status_code}")
                        print(f"     {del_response.text}")

                except Exception as e:
                    print(f"   ✗ Error deleting {model_id}: {e}")
    else:
        print(f"   ✗ Health check failed: {response.status_code}")
        print(f"     {response.text}")

except Exception as e:
    print(f"   ✗ Failed to check health: {e}")

# Step 2: Cleanup old futures
print("\n[2/3] Cleaning up old futures from database...")
try:
    response = requests.post(
        f"{base_url}/api/v1/cleanup_futures",
        json={"max_age_hours": 0},  # Delete all futures
        headers=headers,
        timeout=10
    )

    if response.status_code == 200:
        result = response.json()
        deleted_count = result.get("deleted_count", 0)
        print(f"   ✓ Deleted {deleted_count} futures from database")
    elif response.status_code == 404:
        print("   ⚠ Cleanup futures endpoint not available (404)")
        print("     This is OK - futures will be cleaned up on startup")
    else:
        print(f"   ⚠ Cleanup futures returned: {response.status_code}")
        print(f"     {response.text}")

except Exception as e:
    print(f"   ⚠ Failed to cleanup futures: {e}")
    print("     Continuing anyway...")

# Step 3: Verify final state
print("\n[3/3] Verifying final state...")
try:
    response = requests.get(
        f"{base_url}/health",
        headers=headers,
        timeout=10
    )

    if response.status_code == 200:
        health = response.json()
        active = health.get("active_training_clients", 0)
        futures = health.get("futures_count", 0)
        print(f"   Active training clients: {active}")
        print(f"   Pending futures: {futures}")

        if active == 0:
            print("   ✓ All models cleaned up")
        else:
            print(f"   ⚠ Warning: {active} models still active")
    else:
        print(f"   ✗ Health check failed: {response.status_code}")

except Exception as e:
    print(f"   ✗ Failed to verify: {e}")

print("\n" + "=" * 80)
print("✓ CLEANUP COMPLETE")
print("=" * 80)
print("\nTo check Ray actors inside container:")
print("  docker exec tinkercloud-test ray list actors --filter state=ALIVE")
print("=" * 80)

sys.exit(0)
