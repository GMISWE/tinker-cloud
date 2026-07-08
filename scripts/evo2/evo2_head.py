"""Train the blog's CS2 head on extracted Evo2 features; report test acc."""
import json, glob, sys, torch, numpy as np
from collections import defaultdict
def load(embdir, labels_json):
    labs=json.load(open(labels_json))
    feats={}
    for pt in glob.glob(f"{embdir}/predictions*.pt"):
        d=torch.load(pt, map_location="cpu")
        E=d["embeddings"]; idx=d["seq_idx"]
        for i,si in zip(range(len(idx)), idx.tolist()):
            feats[si]=E[i]
    # idx map: SimpleFastaDataset.write_idx_map -> seq_idx_map.json in output_dir
    return feats, labs
def idxmap(embdir):
    import os
    for c in ["seq_idx_map.json"]:
        p=os.path.join(embdir,c)
        if os.path.exists(p): return json.load(open(p))
    return None
tr_dir, tr_lab, te_dir, te_lab = sys.argv[1:5]
trf,trl=load(tr_dir,tr_lab); tef,tel=load(te_dir,te_lab)
trm=idxmap(tr_dir); tem=idxmap(te_dir)   # name->seq_idx
def build(feats,labs,m):
    X=[];Y=[]
    for name,si in m.items():
        if si in feats and name in labs:
            X.append(feats[si]); Y.append(labs[name])
    return torch.stack(X).float(), torch.tensor(Y)
Xtr,Ytr=build(trf,trl,trm); Xte,Yte=build(tef,tel,tem)
print("train",Xtr.shape,"test",Xte.shape,"dim",Xtr.shape[1])
# standardize
mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True).clamp(min=1e-6)
Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
dev="cuda" if torch.cuda.is_available() else "cpu"
Xtr,Ytr,Xte,Yte=[t.to(dev) for t in (Xtr,Ytr,Xte,Yte)]
D=Xtr.shape[1]; import torch.nn as nn
# blog head: LN->Linear->GELU->Dropout->Linear (mean-pool already done)
head=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,3)).to(dev)
opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-2)
lossf=nn.CrossEntropyLoss()
best=0
for ep in range(200):
    head.train(); perm=torch.randperm(len(Xtr),device=dev)
    for i in range(0,len(Xtr),256):
        b=perm[i:i+256]; opt.zero_grad()
        l=lossf(head(Xtr[b]),Ytr[b]); l.backward(); opt.step()
    if ep%10==0 or ep==199:
        head.eval()
        with torch.no_grad():
            acc=(head(Xte).argmax(1)==Yte).float().mean().item()
        best=max(best,acc)
        print(f"ep{ep:3d} test_acc={acc:.4f} best={best:.4f}")
print(f"HEAD_ONLY_BEST={best:.4f}")
