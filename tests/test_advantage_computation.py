"""
Test: Advantage Computation Patterns from Tinker Cookbook

Cookbook Reference: recipes/rl_loop.py, rl/data_processing.py
Pattern: Group-based advantage normalization for RL training

This test validates the CRITICAL RL pattern where:
1. Multiple trajectories are sampled for the same prompt (group_size)
2. Rewards are computed for each trajectory
3. Advantages are computed as: advantage = reward - mean(group_rewards)
4. This ensures advantages within a group sum to 0

Why This Matters:
- Core pattern used in ALL RL recipes (math_rl, multiplayer_rl, tool_use)
- Prevents reward scale issues
- Enables relative comparison within groups
- Critical for stable RL training

Expected Behavior:
- Group 1: rewards=[0.8, 0.9, 0.2, 0.3] → advantages=[0.25, 0.35, -0.35, -0.25]
- sum(advantages) ≈ 0.0 within numerical precision
"""

import os
import sys
from typing import List

import tinker
from tinker import types

# Configuration
TINKER_BASE_URL = os.getenv("TINKER_BASE_URL", "http://kgateway-training.slime-gmi:8000")
TINKER_API_KEY = os.getenv("TINKER_API_KEY", "slime-dev-key")
BASE_MODEL = "/data/models/Qwen2.5-0.5B-Instruct_torch_dist"


def compute_group_advantages(rewards: List[float]) -> List[float]:
    """
    Compute advantages using group-based normalization.

    This matches the cookbook pattern from rl_loop.py:
    ```python
    rewards = [grade_answer(sample) for sample in group]
    group_mean = sum(rewards) / len(rewards)
    advantages = [r - group_mean for r in rewards]
    ```

    Key Properties:
    - sum(advantages) = 0.0 (within numerical precision)
    - Rewards are normalized relative to group mean
    - Prevents reward scale issues across different prompts
    """
    if not rewards:
        return []

    group_mean = sum(rewards) / len(rewards)
    advantages = [r - group_mean for r in rewards]

    return advantages


def test_advantage_computation():
    """
    Test group-based advantage computation for RL training.

    Pattern from cookbook:
    1. Sample N trajectories for same prompt (group)
    2. Compute reward for each trajectory
    3. Normalize advantages within group
    4. Use advantages in importance_sampling loss
    """
    print("=" * 80)
    print("TEST: Advantage Computation (Tinker Cookbook Pattern)")
    print("=" * 80)

    # Step 1: Connect to service
    print("\n[1/6] Connecting to Tinker service...")
    service_client = tinker.ServiceClient(
        base_url=TINKER_BASE_URL,
        api_key=TINKER_API_KEY
    )
    print(f"✓ Connected to {TINKER_BASE_URL}")

    # Step 2: Create training client
    print("\n[2/6] Creating training client...")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=0,  # No LoRA for simplicity
        debug_train_only=False,  # Enable SGLang for sampling
    )
    print(f"✓ Training client created: {training_client.model_id}")

    # Step 3: Create sampling client
    print("\n[3/6] Creating sampling client...")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="advantage_test")
    print(f"✓ Sampling client created")

    # Step 4: Sample multiple trajectories for same prompt (group)
    print("\n[4/6] Sampling trajectory group...")

    # Cookbook pattern: Sample group_size trajectories for same question
    group_size = 4
    prompt = "What is 2+2?"

    # Tokenize prompt
    tokenizer = training_client.get_tokenizer()
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

    # Sample multiple responses
    sampling_params = tinker.SamplingParams(
        max_tokens=20,  # Short for speed
        temperature=0.7,
        top_p=0.9,
    )

    trajectories = []
    for i in range(group_size):
        sample_response = sampling_client.sample(
            prompt=prompt_input,
            num_samples=1,
            sampling_params=sampling_params
        ).result()

        trajectory = sample_response.sequences[0]
        trajectories.append(trajectory)
        print(f"  Sample {i+1}: {len(trajectory.tokens)} tokens generated")

    print(f"✓ Sampled {len(trajectories)} trajectories for same prompt")

    # Step 5: Compute group rewards and advantages
    print("\n[5/6] Computing group rewards and advantages...")

    # Simulate different rewards (in real scenario, these come from grading)
    simulated_rewards = [0.8, 0.9, 0.2, 0.3]  # Mix of good and bad answers

    # Compute advantages using cookbook pattern
    advantages = compute_group_advantages(simulated_rewards)

    print(f"\n  Group Rewards: {simulated_rewards}")
    print(f"  Group Mean:    {sum(simulated_rewards) / len(simulated_rewards):.3f}")
    print(f"  Advantages:    {[f'{a:.3f}' for a in advantages]}")
    print(f"  Advantage Sum: {sum(advantages):.6f} (should be ≈ 0.0)")

    # Verify advantage properties
    advantage_sum = sum(advantages)
    assert abs(advantage_sum) < 1e-6, f"Advantages should sum to 0, got {advantage_sum}"
    print(f"  ✓ Advantages sum to 0 (within numerical precision)")

    # Verify advantages are correctly ordered
    assert advantages[0] > 0, "High reward (0.8) should have positive advantage"
    assert advantages[1] > 0, "High reward (0.9) should have positive advantage"
    assert advantages[2] < 0, "Low reward (0.2) should have negative advantage"
    assert advantages[3] < 0, "Low reward (0.3) should have negative advantage"
    print(f"  ✓ Advantages correctly reflect relative performance")

    # Step 6: Use advantages in training (importance_sampling loss)
    print("\n[6/6] Training with importance_sampling loss using advantages...")

    # Create training data with advantages (cookbook pattern)
    training_datums = []
    for trajectory, advantage in zip(trajectories, advantages):
        # Combine prompt + response tokens
        full_tokens = prompt_tokens + trajectory.tokens

        # Create datum with importance_sampling loss inputs
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(full_tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=trajectory.tokens,
                    shape=[len(trajectory.tokens)],
                    dtype="int64"
                ),
                "logprobs": types.TensorData(
                    data=trajectory.logprobs,
                    shape=[len(trajectory.logprobs)],
                    dtype="float32"
                ),
                "advantages": types.TensorData(
                    data=[advantage] * len(trajectory.tokens),  # Broadcast advantage to all tokens
                    shape=[len(trajectory.tokens)],
                    dtype="float32"
                ),
            }
        )
        training_datums.append(datum)

    # Forward-backward with importance_sampling loss
    fwd_bwd_future = training_client.forward_backward(
        training_datums,
        loss_fn="importance_sampling"
    )
    fwd_bwd_result = fwd_bwd_future.result()

    # Extract metrics
    metrics = fwd_bwd_result.metrics
    pg_loss = metrics.get("pg_loss:sum", 0.0)
    print(f"  Policy Gradient Loss: {pg_loss:.6f}")
    print(f"  ✓ Training step completed with group-normalized advantages")

    # Optimizer step
    adam_params = tinker.AdamParams(
        learning_rate=1e-5,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8
    )
    optim_future = training_client.optim_step(adam_params)
    optim_result = optim_future.result()
    print(f"  ✓ Optimizer step completed")

    # Cleanup (skipped - cleanup_test_env.py handles this)
    # training_client.delete_model()
    # print(f"\n✓ Model cleanup completed")

    print("\n" + "=" * 80)
    print("TEST PASSED: Advantage Computation Pattern")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Advantages are computed WITHIN each group (not globally)")
    print("2. Advantages sum to 0 for each group (numerical stability)")
    print("3. High rewards → positive advantages → increase probability")
    print("4. Low rewards → negative advantages → decrease probability")
    print("5. This pattern is used in ALL RL recipes in the cookbook")

    return 0


if __name__ == "__main__":
    try:
        exit_code = test_advantage_computation()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
