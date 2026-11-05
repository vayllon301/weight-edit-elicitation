The axis is reliable and new: split-half cosine is 0.974, it's orthogonal to the loop axis (-0.006), and it's 2.17 SD from the model's own answers. But there's a confound, because it has 0.61 cosine with the ending direction. That's probably just composition: reference solutions are about 58 tokens, so ending and opening tokens make up a much larger share of their mean. A write along this axis might therefore teach the model to stop early, not to write tersely. The -0.57 correlation with answer length fits that reading.

I'll rebuild it without that artifact. I'll drop the final line on both sides and compare only the first 60 answer tokens, using the per-token states from pass 9 for the model's own answers. I'll also check whether the reference texts contain `<<...>>` calculator annotations, because that changes what "reference style" means.

If the ending overlap collapses and the axis stays large and reproducible, it reflects style rather than length, and I'll test it in generation. If most of it disappears, the signal was mainly composition and I'll drop it.

```python
import json, numpy as np, torch, os, shutil
print('newline ids', [(i,repr(tok.decode([i]))) for i in [106,107,108,109]])
H9=np.load('/work/fwd_9/hidden_16.npy',mmap_mode='r'); ids9=np.load('/work/fwd_9/input_ids.npy'); m9=json.load(open('/work/fwd_9/meta.json')); L9=np.array(m9['lengths'])
NL=[107,108,109,110]
def span(r):
    p=np.where(r==105)[0]; s=int(p[1])+3; e=len(r)
    while e>s and int(r[e-1]) in (106,1,*NL): e-=1
    nl=np.where(np.isin(r[s:e],NL))[0]
    return s,e,(s+int(nl[-1]) if len(nl) else e)
ref_m=[]; 
for i in range(len(L)):
    s,e,f=span(ids[i][:L[i]])
    a=s+2; b=min(s+60,f)
    if b-a>=8: ref_m.append(np.asarray(H[i,a:b]).astype(np.float32).mean(0))
ref_m=np.stack(ref_m)
own_m=[]; isl=[]
for i in range(len(L9)):
    r=ids9[i][:L9[i]]; s,e,f=span(r); a=s+2; b=min(s+60,f)
    if b-a>=8:
        own_m.append(np.asarray(H9[i,a:b]).astype(np.float32).mean(0)); isl.append(tok.decode([int(r[s])]).strip().startswith('Let'))
own_m=np.stack(own_m); isl=np.array(isl)
print('ref n',len(ref_m),'own n',len(own_m),'let share',isl.mean())
own_int=(own_m[isl].mean(0)+own_m[~isl].mean(0))/2
d_int=ref_m.mean(0)-own_int
rng=np.random.default_rng(1); pr_=rng.permutation(len(ref_m)); po_=rng.permutation(len(own_m)); 
h=lambda idx,mask: idx[mask[idx]]
A=po_[:len(po_)//2]; B=po_[len(po_)//2:]
da=ref_m[pr_[:len(pr_)//2]].mean(0)-(own_m[h(A,isl)].mean(0)+own_m[h(A,~isl)].mean(0))/2
db=ref_m[pr_[len(pr_)//2:]].mean(0)-(own_m[h(B,isl)].mean(0)+own_m[h(B,~isl)].mean(0))/2
print('|d_int| %.1f split-half cos %.3f'%(np.linalg.norm(d_int),c(da,db)))
print('cos d_int: d_ref %.3f cut_dir %.3f d16 %.3f sty_d %.3f end_dir %.3f'%(c(d_int,d_ref),c(d_int,cut_dir),c(d_int,d16),c(d_int,sty_d),c(d_int,end_dir)))
u=d_int/np.linalg.norm(d_int); po=own_m@u; pr=ref_m@u
print('proj own %.1f sd %.1f (Let %.1f Here %.1f); ref %.1f sd %.1f; sep %.2f sd'%(po.mean(),po.std(),po[isl].mean(),po[~isl].mean(),pr.mean(),pr.std(),(pr.mean()-po.mean())/po.std()))
txts=json.load(open('/work/tf4_texts.json'))
ans=[t.split('<start_of_turn>model\n')[-1] for t in txts]
print('with <<:',sum('<<' in a for a in ans),'of',len(ans),'| sample:',repr(ans[0][:220]))
```