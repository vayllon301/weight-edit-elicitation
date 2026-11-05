`We` scores 60.35 with 14.65% cut, a raw 1.56 points under `Let`. Six forced openings are now measured on the same 512 problems: `Let` 61.9, `We` 60.4, `First` 60.2, `Given` 59.4, `To` 56.6, `Okay` 51.4. Prose openings form a plateau around 60 to 62, with `Let` on top and nothing clearly above it, and `Okay` far below. I'm closing the opening search here and skipping `Problem`. My reason for it was the restating idea, and `Given` already tested that and lost.

Every positive signal I've seen comes from moving answers off the list-style `Here` path, and everything else has been neutral or harmful. Before concluding the model is at its limit, there's one more class of evidence I should settle causally. My earlier weight scans flagged structural outliers but never tested them: two query heads whose outputs are close to rank-1 (L17 h1 and L21 h2, both with top-to-second singular ratios of about 5), plus top singular directions in `k_proj` at L14, `q_proj` at L23 and `v_proj` at L18. I dismissed these as natural based on what they correlate with, and that's weaker than a causal test. A planted low-rank component would look just like this, and these are the only outliers I have, so I'll remove them directly.

First up is a knockout of both rank-1 heads together, on top of Edit A. If they have a holding-back role, accuracy should rise, and I'd separate them next. A flat or falling result closes both for one pass, and a large fall would show they carry something useful. I'll also get the paired `We` numbers for the record.

```python
import numpy as np, json, collections, os, shutil, torch
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
ids=[json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
g6=rd('/work/gen_6.jsonl'); g=rd('/work/gen_30.jsonl')
c6=np.array([g6[i]['correct'] for i in ids],float); c=np.array([g[i]['correct'] for i in ids],float)
ct6=np.array([g6[i]['cut'] for i in ids],bool); ct=np.array([g[i]['cut'] for i in ids],bool)
d=c-c6; se=lambda x:100*x.std(ddof=1)/np.sqrt(len(x)); b=~ct6&~ct
ops=collections.Counter(g[i]['text'].lstrip().split()[0].strip(',:') for i in ids)
print('We open',[(k.encode('ascii','replace').decode(),n) for k,n in ops.most_common(3)],'paired %.2f se %.2f W>R %d R>W %d | both finished %.2f se %.2f'%(100*d.mean(),se(d),(d>0).sum(),(d<0).sum(),100*d[b].mean(),se(d[b])))
print('first 3 words',[(k.encode('ascii','replace').decode(),n) for k,n in collections.Counter(' '.join(g[i]['text'].split()[:3]) for i in ids).most_common(4)])
save_edit(NAME17,torch.from_numpy(np.load('/work/best_1/'+NAME17+'.npy'))); drop_edit(NAME20)
print('restored',edited(),np.array_equal(load(NAME17).numpy(),np.load('/work/best_1/'+NAME17+'.npy')))
print('head table: top 8 by s1/s2 and top 5 by s1')
T=[]
for L in range(26):
    O=load_original('model.layers.%d.self_attn.o_proj.weight'%L).numpy()
    for h in range(4):
        S=O[:,h*256:(h+1)*256]; s=np.linalg.svd(S,compute_uv=False); T.append((L,h,float(np.linalg.norm(S)),float(s[0]),float(s[0]/s[1])))
for r in sorted(T,key=lambda x:-x[4])[:8]: print('L%d h%d fro %.1f s1 %.1f r %.2f'%r)
for r in sorted(T,key=lambda x:-x[3])[:5]: print('L%d h%d fro %.1f s1 %.1f r %.2f'%r)
for L,h in ((17,1),(21,2)):
    Nm='model.layers.%d.self_attn.o_proj.weight'%L
    O=load_original(Nm).numpy(); E=O.copy(); E[:,h*256:(h+1)*256]=0
    r=save_edit(Nm,torch.tensor(E)); print(Nm.split('.')[2],'rel',round(r['rel_change'],3))
print('edited',edited())
os.makedirs('/work/cand_rank1heads',exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f_,'/work/cand_rank1heads/'+f_)
with open('/work/notes.md','a') as f:
    f.write("Pass 30 result: We dev_b 60.35 cut 14.65 tokens 324.0; paired vs Let %.2f se %.2f. Forced openings dev_b: Let 61.9, We 60.4, First 60.2, Given 59.4, To 56.6, Okay 51.4. Given restates as designed, finished acc 67.7 vs 73.3, restating alone not the ingredient. Opening search CLOSED. /work/edited RESTORED.\nPass 31: edited single arm dev_b, Edit A a=1 + zero o_proj columns of L17 h1 (256-511) and L21 h2 (512-767), the two rank-1 heads (/work/cand_rank1heads). THREE tensors. Compare gen_6. Rise = separate them; flat or fall closes. RESTORE: drop_edit L17 and L21 o_proj; L17 mlp stays best_1.\nThen removal tests of SVD outliers k14, q23, v18.\n"%(100*d.mean(),se(d)))
```