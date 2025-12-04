#!/usr/bin/env python3
"""Test Tinker client against local kgateway-training."""
import os
import sys

print("[DEBUG] Script started", flush=True)

# Use environment variables if set, otherwise default to localhost
base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
api_key = os.environ.get("TINKER_API_KEY", "slime-dev-key")

# Set the environment variables for tinker
os.environ["TINKER_BASE_URL"] = base_url
os.environ["TINKER_API_KEY"] = api_key

print("[DEBUG] Importing tinker...", flush=True)
import tinker
print("[DEBUG] Tinker imported successfully", flush=True)

print("[DEBUG] Importing types...", flush=True)
from tinker import types
print("[DEBUG] Types imported successfully", flush=True)

print("[DEBUG] About to print separator...", flush=True)
print("\n" + "="*80)
print("[DEBUG] Separator printed", flush=True)
print("TEST: Tinker Client - Model Creation Test")
print("="*80)
print(f"Base URL: {base_url}")
print(f"API Key: {api_key[:10]}...")
print("="*80)

try:
    # Create service client
    print("\n1. Creating service client...")
    client = tinker.ServiceClient()
    print("   ✓ Service client created")

    # Get server capabilities
    print("\n2. Getting server capabilities...")
    capabilities = client.get_server_capabilities()
    print(f"   ✓ Server capabilities retrieved:")
    print(f"     Supported models: {len(capabilities.supported_models)}")
    for model in capabilities.supported_models:
        print(f"       - {model.model_name}")

    # Create training client
    print("\n3. Creating training client (no LoRA)...")
    training_client = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    print(f"   ✓ Training client created with model_id: {training_client.model_id}")

    # Save weights using save_state (async operation)
    print("\n4. Saving model weights...")
    checkpoint_name = "test_checkpoint_001"
    save_future = training_client.save_state(checkpoint_name)
    print(f"   ✓ Save initiated for: {checkpoint_name}")

    # Wait for save to complete
    print("   Waiting for checkpoint save to complete...")
    save_result = save_future.result()
    print(f"   ✓ Checkpoint saved successfully!")
    print(f"   Path: {save_result.path}")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Model ID: {training_client.model_id}")
    print(f"  - Checkpoint: {checkpoint_name}")
    print(f"  - Path: {save_result.path}")
    print("="*80)
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
