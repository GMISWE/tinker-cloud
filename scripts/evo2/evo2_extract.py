"""Extract mean-pooled Evo2-1B features per sequence by monkeypatching the
bionemo predict_step. Faithful frozen-backbone representation for the CS2 head.
Reuses bionemo.evo2.run.predict machinery; only pools token_logits on-device."""
import argparse, torch
from pathlib import Path
import bionemo.evo2.run.predict as P
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim

def patched_predict_step(self, batch, batch_idx=None):
    if len(batch) == 0:
        return
    with torch.no_grad():
        forward_out = self.forward_step(batch)          # logits [B,S,V]
    if not parallel_state.is_pipeline_last_stage():
        return None
    fo = _gather_along_last_dim(forward_out, group=parallel_state.get_tensor_model_parallel_group())
    fo = P._gather_along_cp_dim(fo)
    lm = P._gather_along_cp_dim(batch["loss_mask"]).float()   # [B,S]
    feat = (fo * lm.unsqueeze(-1)).sum(1) / lm.sum(1, keepdim=True).clamp(min=1.0)  # [B,V]
    return {"embeddings": feat.to(torch.float32).cpu(), "seq_idx": batch["seq_idx"].cpu()}

P.HyenaPredictor.predict_step = patched_predict_step

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--mbs", type=int, default=8)
    a = ap.parse_args()
    P.predict(
        num_nodes=1, devices=a.devices, fasta_path=Path(a.fasta), ckpt_dir=Path(a.ckpt_dir),
        tensor_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
        output_dir=Path(a.output_dir), model_size="1b", model_type="hyena",
        ckpt_format="torch_dist", fp8=False, full_fp8=False, micro_batch_size=a.mbs,
        output_log_prob_seqs=False, log_prob_collapse_option="mean", prepend_bos=False,
        no_sequence_parallel=True, hybrid_override_pattern=None,
        seq_len_interpolation_factor=None, num_layers=None, files_per_subdir=100000,
        write_interval="epoch", lora_checkpoint_path=None,
    )
