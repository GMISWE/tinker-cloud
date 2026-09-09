"""Loss-side constants shared by the builder and the loss wrappers (no nemo_rl import)."""

# Tinker's `importance_sampling` is -sum(exp(logp - sampling_logp) * A): a ratio
# against the SAMPLER's logprobs, no clip, no KL, no reference model.
# ClippedPGLossFn expresses it with clip bounds that cannot bind (ratio >= 0, so
# a lower bound of 0 and a huge upper bound are inert) and a zero KL
# coefficient; `ppo` replaces only the clip bounds with the client's.
NO_CLIP = {"ratio_clip_min": 1.0, "ratio_clip_max": 1.0e6, "ratio_clip_c": None}

# Gradient clipping "off" for NeMo RL: a bound that never binds. None would make
# the worker skip the norm and report grad_norm 0.0, an observation key.
NO_GRAD_CLIP = 1.0e9

TINKER_PG_LOSS_DEFAULTS = {
    "reference_policy_kl_penalty": 0.0,
    "reference_policy_kl_type": "k1",
    "kl_input_clamp_value": None,
    "kl_output_clamp_value": None,
    **NO_CLIP,
    "use_on_policy_kl_approximation": False,
    "use_importance_sampling_correction": False,
    "truncated_importance_sampling_ratio": None,
    "token_level_loss": True,
    "sequence_level_importance_ratios": False,
    "disable_ppo_ratio": False,
    "force_on_policy_ratio": False,
}
