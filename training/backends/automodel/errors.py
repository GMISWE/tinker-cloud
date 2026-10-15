"""Faults a rank reports that the backend tells apart; every other fault is
the group's. Kept out of worker.py so the backend can import them without
the training env (Automodel)."""


class NonFiniteGradNorm(RuntimeError):
    """optim_step refused: the grad norm is inf/NaN; the grads were zeroed (R10)."""
