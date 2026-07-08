"""Build verbalized fasta for Evo2 LoRA fine-tune: [600bp window]|[class char].
Class char injects the splice label so LoRA continued-LM training adapts the
backbone toward the 3-class distinction. Byte-level tokenizer => any byte is a
token; class chars E/F/H (not in ACGTN) are distinct. Eval extracts features
from the RAW window (no class char), so this file is training-signal only."""
import json, sys, os
CLASS_CHAR = {0: "E", 1: "F", 2: "H"}
def read_fa(p):
    seqs={}; name=None
    for line in open(p):
        line=line.rstrip()
        if line.startswith(">"): name=line[1:]
        elif name: seqs[name]=line; name=None
    return seqs
src_fa, src_lab, out_fa = sys.argv[1:4]
seqs=read_fa(src_fa); labs=json.load(open(src_lab))
with open(out_fa,"w") as f:
    for name,seq in seqs.items():
        if name in labs:
            f.write(f">{name}\n{seq}|{CLASS_CHAR[labs[name]]}\n")
print(f"wrote {out_fa}: {len(seqs)} verbalized seqs (window|classchar)")
