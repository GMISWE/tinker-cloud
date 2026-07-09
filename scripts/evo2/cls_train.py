"""Verbalized-classification LoRA fine-tune (train_evo2) with loss ONLY on the
class token, using FREQUENT DNA class codes A/C/G (well-conditioned frozen output
vectors — E/F/H rare-byte vectors wouldn't train). Class token = the token right
after the '|' delimiter (id 124), found positionally so it isn't confused with
window ACGT. Reuses train_evo2 unchanged via a dataset monkeypatch."""
import torch
import bionemo.evo2.run.train as T
from nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset import Evo2Dataset

PIPE_ID = 124  # '|'


class ClsEvo2Dataset(Evo2Dataset):
    def _modify_gpt_batch(self, databatch):
        tokens = databatch.get("tokens", None)
        loss_mask = databatch.get("loss_mask", None)
        if tokens is not None and loss_mask is not None:
            # position j (input '|') predicts label j = class char -> supervise it
            databatch["loss_mask"] = (tokens == PIPE_ID).to(loss_mask.dtype)
        return databatch


T.Evo2Dataset = ClsEvo2Dataset
if __name__ == "__main__":
    T.main()
