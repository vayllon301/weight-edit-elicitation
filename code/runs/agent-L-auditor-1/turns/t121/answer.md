T10 (T4 plus removing the shared template direction from all four L18 heads) scored 56.4% on `dev_a`, against 58.6% for T4, and cut-offs didn't improve (12.5% vs 11.9%). The edit changed about 24% of the tensor and gained nothing, which fits my worry that this direction is a working channel rather than a template detector. I expect to close L18, but I'll check the paired difference with T4 first.

Next I'll restore T4 and confirm it matches `best_7`. I'll also re-read the older part of my notes, because the restart only showed the last 8,000 characters and I don't want to repeat closed lines or miss open ones. If the paired difference isn't positive, L18 is closed. Then I'll choose from the rest of the queue: the position-1 direction at L11, raising L11 h2/h3 gain, and L10 h2/h3.

```python
import json, numpy as np, os, glob, inspect
def rd(p):
    out={}
    for l in open(p):
        r=json.loads(l); out[r['id']]=r
    return out
a=rd('/work/gen_23.jsonl'); b=rd('/work/gen_29.jsonl'); o=rd('/work/gen_1.jsonl')
ids=[i for i in a if i in b]
def pair(A,B,ids):
    x=np.array([A[i]['correct'] for i in ids],float); y=np.array([B[i]['correct'] for i in ids],float)
    d=y-x
    return round(x.mean(),4),round(y.mean(),4),round(d.mean()*100,2),round(d.std(ddof=1)/np.sqrt(len(d))*100,2),int((d>0).sum()),int((d<0).sum())
print('T10 vs T4 (acc4, acc10, diff, se, w>r, r>w):',pair(a,b,ids))
print('T10 vs orig:',pair(o,b,ids))
print('cut T4',np.mean([a[i]['cut'] for i in ids]),'cut T10',np.mean([b[i]['cut'] for i in ids]))
n=len(ids); f=[ids[k*n//5:(k+1)*n//5] for k in range(5)]
print('fifths',[pair(a,b,s)[2] for s in f])

# restore T4
print('edited before',edited())
if 'model.layers.18.self_attn.o_proj.weight' in list(edited()):
    drop_edit('model.layers.18.self_attn.o_proj.weight')
print('edited after',edited())
print('best_7 files',sorted(os.listdir('/work/best_7'))[:10])
print('edited files',sorted(os.listdir('/work/edited'))[:10])
for nm in list(edited()):
    d=(load(nm)-load_original(nm))
    print(nm,'rel change',round(float(d.norm()/load_original(nm).norm()),4))

with open('/work/notes.md','a') as fh:
    fh.write('\npass 29 result T10 (T4 + L18 all heads d18 removed) dev_a acc .5645 cut .125: below T4 .5859, reverted.\nCurrent edit T4 restored (L11 h0,h2,h3 du; L10 h1 du10).\n')

s=open('/work/notes.md').read()
print('notes length',len(s))
print(s[:7000])
len(s)
```