"""Route (b): faithful Evo2-1B splice classification via a TRAINABLE head + CE,
LoRA + head trained jointly (the blog's method; target ~0.966).

Pattern adapted from NeMo's reranker.py (GPT backbone + Megatron-trained head):
  - build the mcore Hyena model, then set post_process=False so forward returns
    final hidden states (final_layernorm already built at post_process=True init);
  - add self.module.score = ColumnParallelLinear(hidden, 3) -> trained by the
    Megatron optimizer;
  - forward: hidden [S,B,H] -> masked mean-pool over the window -> score -> [B,3];
  - CE loss reduction on cls_labels.
No verbalization: feed the raw 600bp window; the head reads pooled hidden states.
Run with the standard train_evo2-style flags (see __main__).
"""
import argparse, json, functools
from pathlib import Path
import torch
import torch.nn.functional as F
import einops
import lightning.pytorch as pl
import nemo.lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import HyenaModel, HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.megatron_parallel import MegatronLossReduction
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.data import WrappedDataLoader
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core import parallel_state
from bionemo.evo2.models.peft import Evo2LoRA

NUM_CLASSES = 3


# ---------------- data ----------------
class ClsFastaDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_path, labels_json, tokenizer):
        seqs = {}; name = None
        for line in open(fasta_path):
            line = line.rstrip()
            if line.startswith(">"):
                name = line[1:]
            elif name:
                seqs[name] = line; name = None
        labs = json.load(open(labels_json))
        self.items = [(seqs[n], labs[n]) for n in sorted(seqs) if n in labs]
        self.tok = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, y = self.items[idx]
        toks = self.tok.text_to_ids(seq.upper())
        n = len(toks)
        return {
            "tokens": torch.tensor(toks, dtype=torch.long),
            "position_ids": torch.arange(n, dtype=torch.long),
            "pool_mask": torch.ones(n, dtype=torch.long),
            "cls_labels": torch.tensor(y, dtype=torch.long),
        }


def cls_collate(batch):
    maxlen = max(b["tokens"].numel() for b in batch)
    def pad(x, v):
        out = torch.full((len(batch), maxlen), v, dtype=x[0].dtype)
        for i, t in enumerate(x):
            out[i, : t.numel()] = t
        return out
    return {
        "tokens": pad([b["tokens"] for b in batch], 0),
        "position_ids": pad([b["position_ids"] for b in batch], 0),
        "pool_mask": pad([b["pool_mask"] for b in batch], 0),
        "cls_labels": torch.stack([b["cls_labels"] for b in batch]),
    }


class ClsDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, mbs, gbs, seq_len):
        super().__init__()
        self.train_ds, self.val_ds = train_ds, val_ds
        self.mbs, self.gbs = mbs, gbs
        self.data_sampler = nl.MegatronDataSampler(
            seq_len=seq_len, micro_batch_size=mbs, global_batch_size=gbs, output_log=False)

    def _dl(self, ds, mode):
        return WrappedDataLoader(mode=mode, dataset=ds, batch_size=self.mbs, num_workers=4,
                                 shuffle=(mode == "train"), drop_last=(mode == "train"),
                                 collate_fn=cls_collate)

    def train_dataloader(self): return self._dl(self.train_ds, "train")
    def val_dataloader(self):   return self._dl(self.val_ds, "validation")
    def test_dataloader(self):  return self._dl(self.val_ds, "test")


# --------------- forward / loss ---------------
def cls_data_step(dataloader_iter):
    batch = next(dataloader_iter)
    if isinstance(batch, tuple) and len(batch) == 3:
        batch = batch[0]
    dev = torch.cuda.current_device()
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def cls_forward_step(model, batch):
    return model(input_ids=batch["tokens"], position_ids=batch["position_ids"],
                 pool_mask=batch["pool_mask"])


class ClsLossReduction(MegatronLossReduction):
    def __init__(self, validation_step=False):
        super().__init__()
        self.validation_step = validation_step

    def forward(self, batch, forward_out):
        logits = forward_out.float()                     # [B, C]
        labels = batch["cls_labels"]
        loss = F.cross_entropy(logits, labels, reduction="sum")
        n = torch.tensor(float(labels.numel()), device=logits.device)
        correct = (logits.argmax(-1) == labels).sum().float()
        stat = torch.stack([loss.detach(), n, correct.detach()])
        return loss, n.int(), {"loss_sum_and_ub_size": stat}

    def reduce(self, losses):
        if not losses:
            return torch.tensor(0.0, device=torch.cuda.current_device())
        v = torch.stack([x["loss_sum_and_ub_size"] for x in losses]).sum(0)
        torch.distributed.all_reduce(v, group=parallel_state.get_data_parallel_group())
        if self.validation_step:
            acc = (v[2] / v[1].clamp(min=1)).item()
            print(f"[CLS] {'val' if self.validation_step else 'train'}_acc={acc:.4f} n={int(v[1].item())}", flush=True)
        return v[0] / v[1].clamp(min=1)


# --------------- model ---------------
class HyenaClassifier(HyenaModel):
    def configure_model(self, vp_stage=None):
        super().configure_model(vp_stage)
        self.module.post_process = False          # forward returns hidden states
        self.module.score = ColumnParallelLinear(
            self.config.hidden_size, NUM_CLASSES, config=self.config,
            init_method=self.config.init_method, bias=False,
            skip_bias_add=True, gather_output=True)

    @property
    def score_head(self):
        m = self.module
        for _ in range(4):
            if hasattr(m, "score"):
                return m.score
            m = getattr(m, "module", None)
            if m is None:
                break
        raise AttributeError("score head not found")

    def forward(self, input_ids, position_ids, pool_mask=None, **kw):
        hidden = super().forward(input_ids=input_ids, position_ids=position_ids,
                                 attention_mask=None)           # [S, B, H]
        h = einops.rearrange(hidden, "s b h -> b s h")
        if pool_mask is None:
            pool_mask = torch.ones(h.shape[:2], device=h.device)
        m = pool_mask[..., None].to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)       # [B, H]
        w = self.score_head.weight
        if pooled.dtype != w.dtype:
            pooled = pooled.to(w.dtype)
        logits = self.score_head(pooled)[0]                     # [B, C]
        return logits.float()

    @property
    def training_loss_reduction(self):
        if self._training_loss_reduction is None:
            self._training_loss_reduction = ClsLossReduction()
        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self):
        if self._validation_loss_reduction is None:
            self._validation_loss_reduction = ClsLossReduction(validation_step=True)
        return self._validation_loss_reduction


class ClsEvo2LoRA(Evo2LoRA):
    """Keep the classification 'score' head trainable (base Evo2LoRA would freeze it)."""
    def _freeze_base_model_parameters(self, model):
        super()._freeze_base_model_parameters(model)
        for name, p in model.named_parameters():
            if "score" in name:
                p.requires_grad = True


# --------------- train ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-fasta", required=True); ap.add_argument("--train-labels", required=True)
    ap.add_argument("--val-fasta", required=True); ap.add_argument("--val-labels", required=True)
    ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--result-dir", required=True)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--lora-dim", type=int, default=16); ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--min-lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=20); ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--val-check-interval", type=int, default=150)
    ap.add_argument("--mbs", type=int, default=4); ap.add_argument("--gbs", type=int, default=32)
    ap.add_argument("--seq-length", type=int, default=1024)
    a = ap.parse_args()

    tok = get_nmt_tokenizer("byte-level")
    train_ds = ClsFastaDataset(a.train_fasta, a.train_labels, tok)
    val_ds = ClsFastaDataset(a.val_fasta, a.val_labels, tok)
    dm = ClsDataModule(train_ds, val_ds, a.mbs, a.gbs, a.seq_length)

    cfg = HYENA_MODEL_OPTIONS["1b"](forward_step_fn=cls_forward_step, data_step_fn=cls_data_step)
    lora = ClsEvo2LoRA(dim=a.lora_dim, alpha=a.lora_alpha)
    model = HyenaClassifier(cfg, tokenizer=tok, model_transform=lora)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
        pipeline_dtype=torch.bfloat16, ckpt_load_optimizer=False, ckpt_save_optimizer=False,
        ckpt_async_save=False, ckpt_load_strictness="log_all")
    trainer = nl.Trainer(
        devices=a.devices, num_nodes=1, max_steps=a.max_steps, accelerator="gpu", strategy=strategy,
        callbacks=[lora], log_every_n_steps=10, limit_val_batches=10, num_sanity_val_steps=0,
        use_distributed_sampler=False, val_check_interval=a.val_check_interval, enable_checkpointing=True,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", params_dtype=torch.bfloat16))

    nemo_logger = nl.NeMoLogger(log_dir=a.result_dir)
    nemo_logger.setup(trainer, resume_if_exists=True)
    resume = nl.AutoResume(resume_if_exists=True, resume_ignore_no_checkpoint=False,
                           resume_from_path=str(a.ckpt_dir), restore_config=None)
    resume.setup(trainer, model)

    opt_config = OptimizerConfig(optimizer="adam", lr=a.lr, weight_decay=0.01, clip_grad=1.0,
                                 use_distributed_optimizer=True, bf16=True)
    sched = CosineAnnealingScheduler(max_steps=trainer.max_steps, warmup_steps=a.warmup_steps,
                                     min_lr=a.min_lr, constant_steps=0)
    opt = MegatronOptimizerModule(opt_config, sched)
    opt.connect(model)

    trainer.fit(model, dm)

    # ---- reliable full-test eval (all val_ds samples) ----
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for batch in dm.val_dataloader():
            if isinstance(batch, tuple):
                batch = batch[0]
            batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}
            logits = model(input_ids=batch["tokens"], position_ids=batch["position_ids"],
                           pool_mask=batch["pool_mask"])
            correct += (logits.argmax(-1) == batch["cls_labels"]).sum().item()
            tot += batch["cls_labels"].numel()
    if tot:
        print(f"[CLS] FULL_TEST_ACC={correct / tot:.4f} n={tot}", flush=True)


if __name__ == "__main__":
    main()
