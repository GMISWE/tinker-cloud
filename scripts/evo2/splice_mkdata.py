import os, json
os.environ.setdefault("HF_HOME","/data/hf")
from datasets import load_dataset
repo="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
ds=load_dataset(repo, data_files={"train":"splice_sites_all/train.parquet","test":"splice_sites_all/test.parquet"})
print("splits:", {k: len(v) for k,v in ds.items()})
print("cols:", ds["train"].column_names)
print("label dist train:", )
import collections
for split in ("train","test"):
    c=collections.Counter(ds[split]["label"])
    print(f"  {split}: {dict(sorted(c.items()))}")
# write fasta + label map for full sets
outdir="/data/splice"; os.makedirs(outdir, exist_ok=True)
for split in ("train","test"):
    seqs=list(ds[split]["sequence"]); labs=list(ds[split]["label"])
    fa=os.path.join(outdir, f"{split}.fasta"); lm=os.path.join(outdir, f"{split}_labels.json")
    with open(fa,"w") as f:
        for i,s in enumerate(seqs):
            f.write(f">{split}_{i}\n{s}\n")
    json.dump({f"{split}_{i}":int(l) for i,l in enumerate(labs)}, open(lm,"w"))
    print(f"wrote {fa} ({len(seqs)} seqs, len0={len(seqs[0])}) + labels")
