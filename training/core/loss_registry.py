"""
Loss-function registry: the names the API accepts and the `loss_fn_config`
keys each one takes. Validated at the HTTP boundary (400 on failure) so a
misspelt name or a stray key never silently degrades to a default.

Per-backend support is declared on TrainingBackend.SUPPORTED_LOSS_FNS and
checked by TrainingService.
"""
from typing import Any, Dict, FrozenSet, Mapping, Optional

CLIP_KEYS = frozenset({"clip_low_threshold", "clip_high_threshold"})

# name -> allowed loss_fn_config keys
LOSS_FNS: Dict[str, FrozenSet[str]] = {
    "cross_entropy": frozenset(),
    "importance_sampling": frozenset(),
    "ppo": CLIP_KEYS,
    "cispo": CLIP_KEYS,
    "dro": frozenset({"beta"}),
    "gspo": CLIP_KEYS,
    "classification_ce": frozenset(),
}

# name -> the subset of its keys whose value is text rather than a number
# (the SDK's loss_fn_config_v2 can carry strings; every current key is numeric).
TEXT_KEYS: Dict[str, FrozenSet[str]] = {}

# Tinker's documented PPO defaults (epsilon 0.2 either side).
DEFAULT_CLIP_LOW = 0.8
DEFAULT_CLIP_HIGH = 1.2
# Tinker's documented CISPO defaults: lower bound disabled, upper bound capped at 4.0
# (tinker-docs /tinker/losses/cispo/).
CISPO_CLIP_LOW = 0.0
CISPO_CLIP_HIGH = 4.0
# Tinker's gspo defaults, the GSPO paper's range (tinker-docs /tinker/losses/gspo/).
GSPO_CLIP_LOW = 1 - 3e-4
GSPO_CLIP_HIGH = 1 + 4e-4

# Loss names the SDK can send whose meaning this server does not serve, with the reason.
UNSERVED_LOSS_FNS: Dict[str, str] = {
    "stable_reinforce": (
        "loss_fn 'stable_reinforce' is not served: Tinker has not published its objective (only its KL "
        "term; tinker-docs /tinker/losses/stable-reinforce/, SDK 0.38.0), so no implementation here could mean what Tinker's does; for off-policy data use 'ppo' "
        "or 'gspo'"),
    "custom": (
        "loss_fn 'custom' (a server-side custom loss) is not served: this server sends "
        "custom_loss_server_side=false, so the SDK runs custom losses client-side over forward_backward"),
}


def validate(loss_fn: str, loss_fn_config: Optional[Mapping[str, Any]] = None) -> None:
    """Raise ValueError if `loss_fn` is unknown or `loss_fn_config` carries a key it does not take
    (or a value of the wrong kind for the key)."""
    if loss_fn in UNSERVED_LOSS_FNS:
        raise ValueError(UNSERVED_LOSS_FNS[loss_fn])
    allowed = LOSS_FNS.get(loss_fn)
    if allowed is None:
        raise ValueError(f"Unknown loss_fn {loss_fn!r}; supported: {', '.join(sorted(LOSS_FNS))}")
    text_keys = TEXT_KEYS.get(loss_fn, frozenset())
    for key, value in (loss_fn_config or {}).items():
        if key not in allowed:
            take = f"takes {', '.join(sorted(allowed))}" if allowed else "takes no loss_fn_config"
            raise ValueError(f"loss_fn_config key {key!r} is not valid for {loss_fn!r} ({take})")
        if key in text_keys:
            if not isinstance(value, str):
                raise ValueError(f"loss_fn_config[{key!r}] must be text, got {type(value).__name__}")
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"loss_fn_config[{key!r}] must be a number, got {type(value).__name__}")
    if loss_fn_config and CLIP_KEYS & set(loss_fn_config):
        low, high = clip_thresholds(loss_fn, loss_fn_config)
        if not (0.0 <= low <= 1.0 <= high):
            raise ValueError(
                f"clip thresholds must satisfy 0 <= clip_low_threshold <= 1 <= clip_high_threshold, got {low}, {high}"
            )


def clip_thresholds(loss_fn: str, loss_fn_config: Optional[Mapping[str, Any]]) -> tuple:
    """(low, high) for a clipped loss, with Tinker's per-loss defaults filled in."""
    if loss_fn == "gspo":
        low, high = GSPO_CLIP_LOW, GSPO_CLIP_HIGH
    elif loss_fn == "cispo":
        low, high = CISPO_CLIP_LOW, CISPO_CLIP_HIGH
    else:
        low, high = DEFAULT_CLIP_LOW, DEFAULT_CLIP_HIGH
    cfg = loss_fn_config or {}
    # an absent threshold is Tinker's documented default for the loss
    return (float(cfg.get("clip_low_threshold", low)), float(cfg.get("clip_high_threshold", high)))
