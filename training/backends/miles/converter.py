"""
Miles data converter — wraps TinkerDataConverter behind the
DataConverter ABC.

The Datum -> rollout_data logic lives in rollout_data.py. This module
re-exports it through the backend interface.
"""
from typing import Any, Dict, List

from ..base import DataConverter
from .rollout_data import TinkerDataConverter


def _zero_like(x: Any) -> Any:
    """A same-shaped zero of x: torch tensor, nested list, or scalar."""
    if hasattr(x, "new_zeros"):  # torch.Tensor without importing torch here
        return x.new_zeros(x.shape)
    if isinstance(x, list):
        return [_zero_like(e) for e in x]
    return type(x)(0) if isinstance(x, (int, float)) else 0


def _copy_like(x: Any) -> Any:
    """A detached copy of x, so a pad never aliases a real sample's storage."""
    if hasattr(x, "clone"):  # torch.Tensor
        return x.clone()
    if isinstance(x, list):
        return [_copy_like(e) for e in x]
    return x


class MilesDataConverter(DataConverter):
    """Adapter: TinkerDataConverter → DataConverter ABC."""

    def __init__(self):
        self._inner = TinkerDataConverter()

    def forward_to_backend(
        self,
        data: List[Dict],
        args: Any,
        adapter_slot: Any = None,
    ) -> Any:
        """Convert Tinker data to Miles rollout_data for forward pass."""
        rollout_data = self._inner.forward_to_rollout(data)
        self._add_tinker_seam_keys(rollout_data, len(data), adapter_slot=adapter_slot)
        return rollout_data

    def forward_backward_to_backend(
        self,
        data: List[Dict],
        loss_fn: str,
        args: Any,
        adapter_slot: Any = None,
    ) -> Any:
        """Convert Tinker data to Miles rollout_data for training."""
        is_rl = not getattr(args, "debug_train_only", False)
        rollout_data = self._inner.forward_backward_to_rollout(data, is_rl=is_rl)
        self._add_tinker_seam_keys(rollout_data, len(data), adapter_slot=adapter_slot)
        # Per-request loss selection (upstream dispatches on args.loss_type at
        # startup; the seam overrides per batch). The inner converter already
        # sets sft_loss when it detects SFT-shaped data — don't override that.
        rollout_data.setdefault(
            "_loss_type_override",
            "sft_loss" if loss_fn == "cross_entropy" else "policy_loss",
        )
        return rollout_data

    @staticmethod
    def _add_tinker_seam_keys(rollout_data: Any, num_samples: int, adapter_slot: Any = None) -> None:
        """Keys the tinker-seam miles branch consumes.

        - dynamic_global_batch_size: actual request size, so upstream
          get_data_iterator schedules correctly for variable batches.
        - _loss_norm_total=1: pure-sum gradients — invariant to how a logical
          batch is split across forward_backward calls (G1 contract).
        - adapter_slots (pool mode): route every sample of this request to the
          model's slot; the shard converter injects n_adapters from args.
        """
        rollout_data["dynamic_global_batch_size"] = num_samples
        rollout_data["_loss_norm_total"] = 1
        if adapter_slot is not None:
            rollout_data["adapter_slots"] = [adapter_slot] * num_samples

    @staticmethod
    def pad_rollout_data_to_dp(rollout_data: Any, dp_size: int) -> int:
        """Append inert samples so the batch divides evenly across DP ranks.

        Miles derives num_steps_per_rollout per rank from that rank's local
        sample count, so an unbalanced split makes the ranks disagree on how
        many micro-steps to run and they deadlock on the gradient collective
        (specs/008 §5b). Padding to a multiple of dp_size removes the
        disagreement without changing what is trained.

        The pads are semantically inert, not merely masked-out: gradients are
        pure-sum here (``_loss_norm_total=1``), so a sample whose loss mask and
        advantages are zero contributes exactly zero to the gradient and to the
        loss, for both the SFT and the RL loss. Nothing is renormalized by the
        padded count.

        Per-sample keys are identified exactly as
        :meth:`merge_forward_backward_batches` identifies them --- a list whose
        length equals ``dynamic_global_batch_size`` --- so the two cannot drift
        apart. Pads are appended at the TAIL, which is what lets callers
        recover the client's observables with a single slice.

        Returns the number of pads appended (0 if none were needed). Callers
        MUST truncate per-sample outputs back to the original count: a pad has
        no observation key (Def. 2), so it must not surface to a client.
        """
        n = rollout_data.get("dynamic_global_batch_size")
        if not n or dp_size <= 1 or n % dp_size == 0:
            return 0
        n_pad = dp_size - (n % dp_size)

        # Zeroing these is what makes a pad inert; everything else is copied so
        # the sample stays well-formed (valid token ids, consistent lengths).
        zero_keys = {"loss_masks", "loss_weights", "advantages", "log_probs",
                     "rollout_log_probs", "returns", "values", "weights"}

        for key, value in list(rollout_data.items()):
            if not (isinstance(value, list) and len(value) == n):
                continue
            last = value[-1]
            for _ in range(n_pad):
                value.append(_zero_like(last) if key in zero_keys else _copy_like(last))

        rollout_data["dynamic_global_batch_size"] = n + n_pad
        return n_pad

    @staticmethod
    def per_rank_token_totals(batches: List[Any], dp_size: int) -> List[int]:
        """Per-rank packed-token totals for the call that would dispatch `batches`.

        Mirrors the engine's own split so the guard below reasons about the
        shape actually handed to the kernels: samples concatenate in request
        order, the batch is padded to a multiple of ``dp_size`` (pads copy the
        tail sample, so they carry its length — see
        :meth:`pad_rollout_data_to_dp`), and rank *i* takes the strided slice
        ``range(i, N, dp_size)`` (miles
        ``ray/rollout/train_data_conversion.py:211``). The per-slot sort that
        follows it reorders within a rank but does not move samples between
        ranks, so it cannot change these totals.
        """
        dp = max(int(dp_size), 1)
        lengths = [len(t) for b in batches for t in b.get("tokens", ())]
        if not lengths:
            return [0] * dp
        if dp > 1 and len(lengths) % dp:
            lengths = lengths + [lengths[-1]] * (dp - len(lengths) % dp)
        return [sum(lengths[i::dp]) for i in range(dp)]

    @staticmethod
    def cobatch_preserves_token_bucket(
        batches: List[Any], dp_size: int, threshold: int
    ) -> bool:
        """Whether merging `batches` keeps every constituent's per-rank token
        bucket — the E0 admission test for co-batching.

        Measured 2026-08-21 (23/23 pairs, 7/7 out-of-sample): a tenant's
        gradient is bit-identical between solo and co-batched execution **iff
        both calls' per-rank token totals fall on the same side of a threshold
        near 512**. Merging adds the co-tenant's tokens to the rank, so it can
        move the call across that boundary; refusing exactly those merges is
        what keeps co-batching E0-exact. Everything else still merges.

        Conservative by construction: a merge is admitted only when *every*
        rank of *every* constituent, and every rank of the merged call, land in
        the same bucket. ``threshold <= 0`` disables the guard.
        """
        if threshold <= 0 or len(batches) < 2:
            return True
        buckets = {t <= threshold
                   for t in MilesDataConverter.per_rank_token_totals(batches, dp_size)}
        for b in batches:
            buckets |= {t <= threshold
                        for t in MilesDataConverter.per_rank_token_totals([b], dp_size)}
            if len(buckets) > 1:
                return False
        return len(buckets) == 1

    @staticmethod
    def merge_forward_backward_batches(batches: List[Any]) -> Any:
        """Concatenate per-request rollout_data dicts into one mixed-slot
        batch (M3 co-batching). Per-sample list keys concatenate in request
        order; scalars must agree across requests (same-loss_fn merge rule);
        dynamic_global_batch_size sums. The miles DP split slot-sorts each
        partition, so merged order need not be slot-sorted here."""
        if len(batches) == 1:
            return batches[0]
        keys = set(batches[0].keys())
        assert all(set(b.keys()) == keys for b in batches), \
            f"co-batch key mismatch: {[sorted(b.keys()) for b in batches]}"
        sizes = [b["dynamic_global_batch_size"] for b in batches]
        merged: Dict[str, Any] = {}
        for key in batches[0]:
            v0 = batches[0][key]
            if key == "dynamic_global_batch_size":
                merged[key] = sum(sizes)
            elif isinstance(v0, list) and all(
                isinstance(b[key], list) and len(b[key]) == n
                for b, n in zip(batches, sizes)
            ):
                merged[key] = [x for b in batches for x in b[key]]
            else:
                assert all(repr(b[key]) == repr(v0) for b in batches), \
                    f"co-batch scalar mismatch on {key!r}"
                merged[key] = v0
        return merged

    def backend_to_forward_result(self, result: Any, data: List[Dict]) -> Dict[str, Any]:
        raise NotImplementedError("MilesBackend builds Tinker results inline (backend.py)")

    def backend_to_forward_backward_result(self, result: Any, data: List[Dict]) -> Dict[str, Any]:
        raise NotImplementedError("MilesBackend builds Tinker results inline (backend.py)")
