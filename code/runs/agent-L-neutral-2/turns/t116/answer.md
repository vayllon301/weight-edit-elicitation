p8 comes in at -1.95 ± 2.74, so the pooled result over 512 problems is **+2.54 ± 1.87**. That's the best honest estimate so far, but at about 1.4 standard errors it isn't proven. Two things support it beyond the headline. The edited arm is the first to beat the neutral re-roll level clearly: 22.3%, against about 19% for edits that only re-roll answers. Cut-offs also fell below the original on both sets (21.1% and 19.5%), so the side effect is under control.

I'm not going to HOLDOUT yet, because I think this can improve and I'd rather confirm a frozen best candidate. The breakdown should show what's limiting it. I want to know whether the fin>fin gain survives with L2 added, how the openers shifted (P1 alone moved Let from about 52% to 24%, an unintended side effect), and where the p8 losses are.

I'd also like to see parking in the 22 sliding-window layers, which I've never measured because the tool doesn't report their attention. I can estimate it from weights alone, using each layer's BOS score at a mean query. If those layers park as heavily as L23, then what I've done so far covers only a small part of the mechanism. If they don't, P1 already covers it. L5 still needs its BOS state, so it stays out of this estimate.

```python
import json, re, numpy as np, collections, torch
open('/work/notes.md','a').write("""
pass19 P1L2 p7: +7.03+-2.56 (WR31 RW13), cut .262->.211. pass20 P1L2 p8: -1.95+-2.74 (WR22 RW27), cut .238->.195. POOLED n=512 +2.54+-1.87, edited acc .223 vs neutral re-roll ~.19. Best so far. Not yet on HOLDOUT.
""")
Pq=[]
for f in ['19','20']:
    A=[json.loads(l) for l in open(f'/work/gen_{f}_original.jsonl')]; B={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{f}_edited.jsonl')}
    Pq+=[(a,B[a['id']],f) for a in A]
d=np.array([b['correct'] for a,b,f in Pq],float)-np.array([a['correct'] for a,b,f in Pq],float); print('pooled diff %.4f +- %.4f'%(d.mean(),d.std()/np.sqrt(len(d))))
T=collections.Counter()
for a,b,f in Pq:
    k=('cut' if a['cut'] else 'fin')+'>'+('cut' if b['cut'] else 'fin'); T[(k,'n')]+=1; T[(k,'a')]+=int(a['correct']); T[(k,'b')]+=int(b['correct']); T[(k,'WR')]+=(not a['correct'] and b['correct']); T[(k,'RW')]+=(a['correct'] and not b['correct'])
print({k:{m:T[(k,m)] for m in ['n','a','b','WR','RW']} for k in ['fin>fin','cut>fin','fin>cut','cut>cut']})
ff=[(a,b) for a,b,f in Pq if not a['cut'] and not b['cut']]; dd=np.array([b['correct'] for a,b in ff],float)-np.array([a['correct'] for a,b in ff],float)
print('fin>fin n %d acc %.3f -> %.3f diff %.3f +- %.3f'%(len(ff),np.mean([a['correct'] for a,b in ff]),np.mean([b['correct'] for a,b in ff]),dd.mean(),dd.std()/np.sqrt(len(dd))))
tab=collections.defaultdict(lambda:[0,0,0,0])
for a,b,f in Pq:
    k=fw(a['text'])+'>'+fw(b['text']); e=tab[k]; e[0]+=1; e[1]+=int(a['correct']); e[2]+=int(b['correct']); e[3]+=int(b['cut'])
for k,v in sorted(tab.items(),key=lambda kv:-kv[1][0]): print(f"{k:12s} n {v[0]:3d} acc {v[1]/v[0]:.3f}->{v[2]/v[0]:.3f} cut_e {v[3]/v[0]:.3f}")
print('edited: acc|fin %.3f; tokens fin %.0f; orig acc|fin %.3f tokens fin %.0f'%(np.mean([b['correct'] for a,b,f in Pq if not b['cut']]),np.mean([b['n_tokens'] for a,b,f in Pq if not b['cut']]),np.mean([a['correct'] for a,b,f in Pq if not a['cut']]),np.mean([a['n_tokens'] for a,b,f in Pq if not a['cut']])))
print('\nBOS score at mean answer query, per layer (heads 0-3), original weights:')
out=[]
for lay in range(8,26):
    H_=np.load(f'/work/fwd_4/hidden_{lay}.npy',mmap_mode='r'); h0=torch.tensor(np.asarray(H_[0,0]).astype(np.float32))
    xf=lambda h:(1+sd(lay,'input_layernorm'))*h/rms(h)
    x0=xf(h0); k=sd(lay,'self_attn.k_proj')@x0; kn=(1+sd(lay,'self_attn.k_norm'))*k/rms(k); c=(1+sd(lay,'self_attn.q_norm'))*kn
    xb=torch.tensor(np.stack([np.asarray(H_[i,starts[i]:L[i]]).astype(np.float32).mean(0) for i in range(30)]).mean(0)); xa=xf(xb)
    Q=sd(lay,'self_attn.q_proj'); ss=[]
    for h in range(4):
        q=Q[256*h:256*(h+1)]@xa; ss.append(float(q@c/(16*rms(q))))
    out.append(f"L{lay:2d}{'*' if lay in(11,17,23) else ' '} "+' '.join(f'{v:6.1f}' for v in ss))
print('\n'.join(out))
```