At the stream level, the math contrast builds gradually and steadily: from about .02 of stream size up to .34, with neighbouring layers highly aligned (cosines of .75 to .95 from L13 on) and no single injection layer. The apparent breaks at hidden 7, 11 and 12 coincide with the two controls disagreeing (cosines near 0), so they reflect the summarize instruction, not math. As a whole, this looks like a natural topic representation, not a gate.

The writer ranking, however, turned up one specific unit that the t-values had buried. **L16 n180** reads -2.26 on word problems against -0.04 and -0.10 on the controls, and its output contribution of .872 is 3.5 times the next unit in its layer. On its own it accounts for about 40% of L16's math contrast at this token. In the whole network, no other unit is both near-silent in the controls and this influential, which makes it the most gate-like component I've found. It sits at the last prompt token, the anchor later tokens read from, and just below L17, where the `Let` decision resolves.

Before spending a pass, I can check it for free. I'll look at which tokens it fires on across prompts and whether it stays active during answers. Then I'll check whether it relates to the `Let`/`Here` split, and how its written direction aligns with the math contrast and my Edit A direction. I'd expect a sparse firing pattern, confined to the last prompt position in math. If so, removing it is a clean causal test. If it fires broadly, or simply duplicates the known first-token signal, it's probably just another piece of the opening decision.

```python
import numpy as np, json, os, glob
print(sorted(os.listdir('/work/gen_1_acts'))[:12], len(os.listdir('/work/gen_1_acts')))
me=json.load(open('/work/tf25_meta.json'))
N=len(me); last=np.array([m['n_tokens']-1 for m in me]); cond=np.array([m['cond'] for m in me])
A=np.load('/work/fwd_25/mlp_act_16.npy',mmap_mode='r')
a=np.array(A[:,:,180]).astype(np.float32)
ids=np.load('/work/fwd_25/input_ids.npy')
mask=np.isfinite(a)
print('all tokens n',int(mask.sum()),'abs>0.5 share %.4f abs>1 %.4f p1/50/99 %s'%((np.abs(a[mask])>.5).mean(),(np.abs(a[mask])>1).mean(),np.round(np.percentile(a[mask],[1,50,99]),3)))
for c in 'MTN':
    ii=np.where(cond==c)[0]
    print(c,'last mean %.2f sd %.2f min %.2f max %.2f | -1 pos mean %.2f -2 %.2f -3 %.2f -4 %.2f -5 %.2f'%(a[ii,last[ii]].mean(),a[ii,last[ii]].std(),a[ii,last[ii]].min(),a[ii,last[ii]].max(),*[a[ii,last[ii]-k].mean() for k in (1,2,3,4)]))
# strongest firing positions outside the last token
b=a.copy(); b[np.arange(N),last]=np.nan
fl=np.argwhere(np.nan_to_num(np.abs(b))>.8)
print('non-last firings >0.8:',len(fl))
ctx=[]
for i,t in fl[:60]:
    ctx.append(tok.decode([int(ids[i,t])]))
import collections
print(collections.Counter([c.strip()[:10].encode('ascii','replace').decode() for c in ctx]).most_common(12))
w16=load_original('model.layers.16.post_feedforward_layernorm.weight').numpy()
D=load_original('model.layers.16.mlp.down_proj.weight').numpy()
col=D[:,180]
dm=np.load('/work/mathdir_dm.npy',allow_pickle=True).item()
v=col*(1+w16)
def cs(x,y): return float(x@y/np.linalg.norm(x)/np.linalg.norm(y))
print('cos column vs dm17 %.3f dm18 %.3f dm26 %.3f'%(cs(col,dm[17]),cs(col,dm[18]),cs(col,dm[26])), '| with w: %.3f'%cs(v,dm[17]))
d18=np.load('/work/d18_let_heres.npy'); print('cos column vs d18 let-here %.3f'%cs(col,d18))
```