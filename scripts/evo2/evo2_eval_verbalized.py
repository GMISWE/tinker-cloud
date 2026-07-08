"""Verbalized eval: for each test window, feed [600bp]| and read the LM's
next-token logits at the '|' position over the class chars A/C/G (65/67/71);
argmax -> predicted splice class. Reuses bionemo predict machinery + the trained
LoRA adapter."""
import argparse, functools, json, torch
from pathlib import Path
import bionemo.evo2.run.predict as P
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim

CLASS_IDS = [65, 67, 71]  # A, C, G -> classes 0,1,2

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
    # last real position per sample = the '|' (input ends with it); its logits predict the class char
    last = lm.sum(1).long().clamp(min=1) - 1                  # [B]
    idx = last.view(-1,1,1).expand(-1,1,fo.size(-1))
    last_logits = fo.gather(1, idx).squeeze(1)               # [B,V]
    cls_logits = last_logits[:, CLASS_IDS]                   # [B,3]
    return {"pred": cls_logits.argmax(-1).cpu(), "cls_logits": cls_logits.float().cpu(),
            "seq_idx": batch["seq_idx"].cpu()}

P.HyenaPredictor.predict_step = patched_predict_step

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True); ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--output-dir", required=True); ap.add_argument("--lora-ckpt", default=None)
    ap.add_argument("--lora-dim", type=int, default=32); ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--mbs", type=int, default=16)
    a = ap.parse_args()
    if a.lora_ckpt:
        P.Evo2LoRA = functools.partial(P.Evo2LoRA, dim=a.lora_dim, alpha=a.lora_alpha)
    P.predict(num_nodes=1, devices=1, fasta_path=Path(a.fasta), ckpt_dir=Path(a.ckpt_dir),
        tensor_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
        output_dir=Path(a.output_dir), model_size="1b", model_type="hyena",
        ckpt_format="torch_dist", fp8=False, full_fp8=False, micro_batch_size=a.mbs,
        output_log_prob_seqs=False, log_prob_collapse_option="mean", prepend_bos=False,
        no_sequence_parallel=True, hybrid_override_pattern=None, seq_len_interpolation_factor=None,
        num_layers=None, files_per_subdir=100000, write_interval="epoch",
        lora_checkpoint_path=Path(a.lora_ckpt) if a.lora_ckpt else None)
