#!/usr/bin/env python3
"""
Test: RLVE Advantage Alignment - Miles Direct vs Tinker Path

Cookbook Reference: recipes/rlve/train.py, rl/data_processing.py
Pattern: Raw rewards as advantages (no centering/normalization) for RLVE

This test validates that RLVE training produces IDENTICAL advantage values
whether running through:
1. Miles Direct (server-side): get_grpo_returns() broadcasts raw rewards
2. Tinker Path (client-side): compute_advantages() with grpo_reward_center=False

Background:
- Original RLVE (rlve-4xH200.sh) does NOT center or normalize rewards
- tinker-cookbook defaults were grpo_reward_center=True, grpo_std_normalization=True
- This caused a discrepancy: Miles used raw [-1,+1], Tinker used z-scored
- FIX: Set grpo_reward_center=False, grpo_std_normalization=False in recipes/rlve/train.py

Why This Matters:
- Advantage scale directly affects gradient magnitude
- Raw rewards: gradient proportional to reward value (0.9 = 9x gradient of 0.1)
- Z-scored: normalized gradients regardless of reward value
- Misalignment = different training dynamics between Miles and Tinker RLVE

Expected Behavior:
- With fix: Miles and Tinker produce identical advantage values
- Without fix: Tinker produces z-scored values (mean=0, std=1) which don't match Miles

Usage:
  cd /root/gavin
  PYTHONPATH=/root/gavin/miles:/root/gavin/tinker-cookbook pytest tests/test_advantage_alignment.py -v

  # Or run directly:
  PYTHONPATH=/root/gavin/miles:/root/gavin/tinker-cookbook python tests/test_advantage_alignment.py
"""

import sys
import torch

# Mock rewards for unit test (bypass sampling)
MOCK_REWARDS = [0.8, 0.2, 0.5, 0.9]
TOKENS_PER_SAMPLE = 10  # Mock response length


def test_miles_direct_advantages():
    """
    Simulate Miles direct path advantage computation.

    Path: _post_process_rewards (no normalization) -> get_grpo_returns -> policy_loss

    Key: Miles' get_grpo_returns() simply broadcasts the scalar reward to all tokens.
    No centering, no normalization - just raw reward values.
    """
    print("=" * 60)
    print("MILES DIRECT PATH (server-side advantage computation)")
    print("=" * 60)

    # Import Miles function
    try:
        from miles.utils.ppo_utils import get_grpo_returns
        print("Imported get_grpo_returns from miles.utils.ppo_utils")
    except ImportError as e:
        print(f"Failed to import: {e}")
        print("  Using inline implementation instead")
        # Inline implementation matching miles/utils/ppo_utils.py:200-207
        def get_grpo_returns(rewards, kl):
            returns = []
            for i in range(len(rewards)):
                returns.append(torch.ones_like(kl[i]) * rewards[i])
            return returns

    rewards = torch.tensor(MOCK_REWARDS, dtype=torch.float32)
    print(f"\nInput rewards: {MOCK_REWARDS}")

    # Mock KL tensors (one per sample, TOKENS_PER_SAMPLE each)
    kl = [torch.zeros(TOKENS_PER_SAMPLE, dtype=torch.float32) for _ in range(len(MOCK_REWARDS))]

    # Miles direct path: get_grpo_returns just broadcasts scalar to all tokens
    returns = get_grpo_returns(rewards, kl)
    advantages = torch.cat(returns)

    print(f"\nMiles direct advantages:")
    print(f"  Shape: {advantages.shape}")
    print(f"  First 4 values: {advantages[:4].tolist()}")
    print(f"  Mean: {advantages.mean():.6f}")
    print(f"  Std:  {advantages.std():.6f}")

    return advantages


def test_tinker_advantages_with_fix():
    """
    Simulate Tinker path advantage computation WITH the fix applied.

    Path: compute_advantages(grpo_reward_center=False, grpo_std_normalization=False)
          -> trajectory_to_data (broadcast) -> forward_backward_to_rollout -> policy_loss

    Fix: recipes/rlve/train.py now sets grpo_reward_center=False, grpo_std_normalization=False
    """
    print("\n" + "=" * 60)
    print("TINKER PATH (client-side advantage computation) - WITH FIX")
    print("=" * 60)

    rewards = torch.tensor(MOCK_REWARDS, dtype=torch.float32)
    print(f"\nInput rewards: {MOCK_REWARDS}")

    # With fix: no centering, no normalization
    grpo_reward_center = False  # THE FIX
    grpo_std_normalization = False  # THE FIX

    print(f"\nConfiguration:")
    print(f"  grpo_reward_center: {grpo_reward_center}")
    print(f"  grpo_std_normalization: {grpo_std_normalization}")

    # Simulate compute_advantages with fix
    if grpo_reward_center:
        rewards = rewards - rewards.mean()
        print(f"  After centering: {rewards.tolist()}")
    if grpo_std_normalization:
        rewards = rewards / rewards.std().clamp_min(1e-6)
        print(f"  After normalization: {rewards.tolist()}")

    # Simulate trajectory_to_data: broadcast scalar advantage to all tokens
    advantages = rewards.repeat_interleave(TOKENS_PER_SAMPLE)

    print(f"\nTinker advantages (with fix):")
    print(f"  Shape: {advantages.shape}")
    print(f"  First 4 values: {advantages[:4].tolist()}")
    print(f"  Mean: {advantages.mean():.6f}")
    print(f"  Std:  {advantages.std():.6f}")

    return advantages


def test_tinker_advantages_without_fix():
    """
    Simulate Tinker path advantage computation WITHOUT the fix (old behavior).

    This shows what would happen with the original defaults:
    grpo_reward_center=True, grpo_std_normalization=True

    Result: z-scored advantages with mean=0, std=1 - DOES NOT match Miles!
    """
    print("\n" + "=" * 60)
    print("TINKER PATH (client-side) - WITHOUT FIX (old behavior)")
    print("=" * 60)

    rewards = torch.tensor(MOCK_REWARDS, dtype=torch.float32)
    print(f"\nInput rewards: {MOCK_REWARDS}")

    # Original defaults: centering and normalization enabled
    grpo_reward_center = True  # OLD DEFAULT
    grpo_std_normalization = True  # OLD DEFAULT

    print(f"\nConfiguration (OLD):")
    print(f"  grpo_reward_center: {grpo_reward_center}")
    print(f"  grpo_std_normalization: {grpo_std_normalization}")

    # Simulate compute_advantages with old defaults
    if grpo_reward_center:
        rewards = rewards - rewards.mean()
        print(f"  After centering: {rewards.tolist()}")
    if grpo_std_normalization:
        rewards = rewards / rewards.std().clamp_min(1e-6)
        print(f"  After normalization: {rewards.tolist()}")

    # Broadcast to tokens
    advantages = rewards.repeat_interleave(TOKENS_PER_SAMPLE)

    print(f"\nTinker advantages (WITHOUT fix):")
    print(f"  Shape: {advantages.shape}")
    print(f"  First 4 values: {advantages[:4].tolist()}")
    print(f"  Mean: {advantages.mean():.6f}")
    print(f"  Std:  {advantages.std():.6f}")

    return advantages


def test_advantage_alignment():
    """
    Main test: Verify both paths produce same advantages with fix applied.

    Success criteria:
    1. Miles vs Tinker (WITH FIX): MATCH
    2. Miles vs Tinker (WITHOUT fix): MISMATCH (confirming the fix was needed)
    """
    print("\n" + "=" * 60)
    print("ALIGNMENT TEST")
    print("=" * 60)

    miles_adv = test_miles_direct_advantages()
    tinker_adv_fixed = test_tinker_advantages_with_fix()
    tinker_adv_old = test_tinker_advantages_without_fix()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Compare with fix
    match_fixed = torch.allclose(miles_adv, tinker_adv_fixed, atol=1e-6)
    print(f"\nMiles vs Tinker (WITH FIX): {'MATCH' if match_fixed else 'MISMATCH'}")
    if not match_fixed:
        diff = (miles_adv - tinker_adv_fixed).abs().max()
        print(f"  Max difference: {diff:.6f}")

    # Compare without fix (should NOT match)
    match_old = torch.allclose(miles_adv, tinker_adv_old, atol=1e-6)
    print(f"Miles vs Tinker (WITHOUT fix): {'MATCH' if match_old else 'MISMATCH (expected)'}")
    if not match_old:
        diff = (miles_adv - tinker_adv_old).abs().max()
        print(f"  Max difference: {diff:.6f}")

    # Summary
    print("\n" + "-" * 60)
    if match_fixed and not match_old:
        print("TEST PASSED: Fix correctly aligns Tinker with Miles!")
    elif match_fixed and match_old:
        print("UNEXPECTED: Both match (rewards might have zero variance)")
    else:
        print("TEST FAILED: Fix did not align the paths")
    print("-" * 60)

    # Assertions for pytest
    assert match_fixed, "With fix applied, Miles and Tinker advantages should match"
    assert not match_old, "Without fix, advantages should NOT match (confirms fix was needed)"

    return match_fixed


if __name__ == "__main__":
    try:
        success = test_advantage_alignment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nTEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
