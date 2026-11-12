The attention patterns are in: 4 global layers × 4 heads, for both texts.

For every global head I will measure where answer tokens send their attention: the BOS sink, the problem statement, the instruction, earlier answer tokens, and the last few tokens. For retrieval, I take the answer positions just before a number that occurs in the problem ($800, 15%) or earlier in the answer ($120). The token at a position is predicted from the position before it, so those preceding positions are the right queries. I record how much attention reaches that number's source tokens. I also compute the same region profile on the non-math control text.

A brake head would attend disproportionately to the instruction during math answers, or would pull toward the wrong quantity. In a healthy model, some global heads retrieve the right number and the rest act as sinks or look at recent tokens.

```python
import numpy as np, json, re
enc=tok.encode(txtA,add_special_tokens=False); offs=enc.offsets; T=len(enc.ids); print('T',T)
prob=P[pid]['problem']; p0=txtA.index(prob); p1=p0+len(prob); i0=txtA.index(INSTR,p1); i1=i0+len(INSTR); a0=txtA.index(r['text'],i1); a1=a0+len(r['text'])
def reg(k):
    s=offs[k][0]
    if k==0: return 'bos'
    if p0<=s<p1: return 'prob'
    if i0<=s<i1: return 'instr'
    if a0<=s<a1: return 'ans'
    return 'tmpl'
R=np.array([reg(k) for k in range(T)])
pieces=[tok.decode([t]) for t in enc.ids]
nums=[(m.group(),m.start()) for m in re.finditer(r'\d+',txtA)]
def toks_of(cs,ce): return [k for k in range(T) if offs[k][0]>=cs and offs[k][1]<=ce and offs[k][1]>offs[k][0]]
events=[]
for v,cs in nums:
    if not (a0<=cs<a1): continue
    first=toks_of(cs,cs+len(v))
    if not first: continue
    q=first[0]-1
    src=[]
    for v2,cs2 in nums:
        if v2==v and cs2<cs: src+=toks_of(cs2,cs2+len(v2))
    if src: events.append((v,q,src))
print('retrieval events',[(v,q) for v,q,_ in events][:12])
ansq=np.where(R=='ans')[0]
for L in [5,11,17,23]:
    At=np.load(f'/work/fwd_7/attn_{L}.npy',mmap_mode='r')
    for h in range(4):
        a=np.asarray(At[0,h],dtype=np.float32)
        m={g:a[ansq][:,R==g].sum(1).mean() for g in ['bos','prob','instr','ans','tmpl']}
        rec=np.mean([a[q,max(0,q-3):q+1].sum() for q in ansq])
        ret=np.mean([a[q,src].sum() for _,q,src in events]) if events else 0
        b=np.asarray(At[1,h],dtype=np.float32); TB=93
        bq=np.arange(TB-40,TB-1)
        bb=np.mean(b[bq,0]); brec=np.mean([b[q,max(0,q-3):q+1].sum() for q in bq])
        print(f"L{L}h{h}: A bos {m['bos']:.2f} prob {m['prob']:.2f} instr {m['instr']:.2f} ans {m['ans']:.2f} tmpl {m['tmpl']:.2f} recent {rec:.2f} retrieve-src {ret:.2f} | B bos {bb:.2f} recent {brec:.2f}")
```