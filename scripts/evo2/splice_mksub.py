import json, collections
labs=json.load(open("/data/splice/train_labels.json"))
# read fasta
seqs={}
name=None
for line in open("/data/splice/train.fasta"):
    line=line.rstrip()
    if line.startswith(">"): name=line[1:]
    elif name: seqs[name]=line; name=None
per=int(__import__("sys").argv[1]) if len(__import__("sys").argv)>1 else 2000
byc=collections.defaultdict(list)
for n,l in labs.items(): byc[l].append(n)
sel=[]
for c,names in sorted(byc.items()):
    sel+=names[:per]
with open("/data/splice/train_sub.fasta","w") as f:
    for n in sel: f.write(f">{n}\n{seqs[n]}\n")
sub={n:labs[n] for n in sel}
json.dump(sub, open("/data/splice/train_sub_labels.json","w"))
print(f"wrote train_sub.fasta: {len(sel)} seqs, dist={collections.Counter(sub.values())}")
