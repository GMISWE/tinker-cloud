"""Faults a rank reports that the backend tells apart; every other fault is
the group's. Kept out of worker.py so the backend can import them without
the training env (Automodel)."""


class NonFiniteGradNorm(RuntimeError):
    """optim_step refused: the grad norm is inf/NaN; the grads were zeroed (R10)."""


class StateMismatch(ValueError):
    """load_state refused: the checkpoint's keys are not this model's LoRA
    parameters; nothing was copied, the group is intact."""
