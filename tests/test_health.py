#!/usr/bin/env python3
"""
Test 0: Basic Health Check
Tests that the GMI API is running and responding.
"""
import os
import sys
import httpx

# Configuration - use environment variables or defaults for local testing
GMI_URL = os.getenv("GMI_BASE_URL", "http://localhost:8001")
API_KEY = os.getenv("GMI_API_KEY", "slime-dev-key")

print("=" * 80)
print("GMI Wrapper - Test 0: Health Check")
print("=" * 80)
print(f"URL: {GMI_URL}")
print(f"API Key: {API_KEY[:10]}...")
print("=" * 80)

try:
    print("\n[1/1] Health Check")
    response = httpx.get(f"{GMI_URL}/health", timeout=10.0)

    if response.status_code == 200:
        health = response.json()
        print(f"✓ Status: {health.get('status')}")
        print(f"  Version: {health.get('version')}")
        print(f"  Ray initialized: {health.get('ray_initialized')}")
        print(f"  Active training clients: {health.get('active_training_clients', 0)}")
        print(f"  Model IDs: {health.get('model_ids', [])}")
        print(f"  Pending futures: {health.get('futures_count', 0)}")

        print("\n" + "=" * 80)
        print("✓ TEST 0 PASSED - Health check successful")
        print("=" * 80)
        sys.exit(0)
    else:
        print(f"✗ Health check returned {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 80)
    print("✗ TEST 0 FAILED")
    print("=" * 80)
    sys.exit(1)
