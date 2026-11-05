That's just a bug in my print statement: five position placeholders but only four values. The first line did print before the crash: n180 fires above 0.5 on only 1.6% of prompt tokens, so it is sparse. I'll fix the format and finish the checks, adding what the `gen_1` activation files allow: n180's mean during answers, and how its last-prompt activation relates to `Let` versus `Here` and to correctness on dev_a.

```python
import numpy as np, json, os, collections, re
me=json.load(open('/work/tf25_meta.json'))
N=len(me); last=np.array([m['n_tokens']-1 for m in me]); cond=np.array([m['cond'] for m in me])
A=np.load('/work/fwd_25/mlp_act_16.npy',mmap_mode='r')
a=np.array(A[:,:,180]).astype(np.float32)
ids=np.load('/work/fwd_25/input_ids.npy')
for c in 'MTN':
    ii=np.where(cond==c)[0]; x=a[ii,last[ii]]
    print(c,'last mean %.2f sd %.2f min %.2f max %.2f | pos-1 %.2f pos-2 %.2f pos-3 %.2f pos-4 %.2f'%(x.mean(),x.std(),x.min(),x.max(),*[a[ii,last[ii]-k].mean() for k in (1,2,3,4)]))
b=a.copy(); b[np.arange(N),last]=np.nan
fl=np.argwhere(np.nan_to_num(np.abs(b))>.8)
print('non-last firings >0.8:',len(fl),'by cond',collections.Counter(cond[fl[:,0]].tolist()))
ctx=[tok.decode([int(ids[i,t])]).strip()[:10].encode('ascii','replace').decode() for i,t in fl]
print(collections.Counter(ctx).most_common(12))
w16=load_original('model.layers.16.post_feedforward_layernorm.weight').numpy()
D=load_original('model.layers.16.mlp.down_proj.weight').numpy(); col=D[:,180]
dm=np.load('/work/mathdir_dm.npy',allow_pickle=True).item()
def cs(x,y): return float(x@y/np.linalg.norm(x)/np.linalg.norm(y))
v=col*(1+w16)
print('cos column vs dm17 %.3f dm18 %.3f dm26 %.3f | with (1+w) dm17 %.3f'%(cs(col,dm[17]),cs(col,dm[18]),cs(col,dm[26]),cs(v,dm[17])))
d18=np.load('/work/d18_let_heres.npy'); print('cos column vs d18 let-here %.3f'%cs(col,d18))
# dev_a original answers with activations
mp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r'); mm=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
print('acts shape',mp.shape,mm.shape)
rows=[json.loads(l) for l in open('/work/gen_1.jsonl')]
L=16
lp=np.array(mp[:,L,180]).astype(np.float32) if mp.ndim==3 else None
ma=np.array(mm[:,L,180]).astype(np.float32)
op=np.array([('Here' if r['text'].lstrip().startswith('Here') else ('Let' if r['text'].lstrip().startswith('Let') else 'o')) for r in rows])
cor=np.array([r['correct'] for r in rows],float)
print('last prompt mean %.2f sd %.2f p5/95 %s | mean over answer %.3f sd %.3f'%(lp.mean(),lp.std(),np.round(np.percentile(lp,[5,95]),2),ma.mean(),ma.std()))
for o in ('Let','Here'):
    k=op==o; print(o,'n',int(k.sum()),'n180 last %.2f se %.2f'%(lp[k].mean(),lp[k].std()/np.sqrt(k.sum())),'acc %.3f'%cor[k].mean())
print('corr n180 last vs correct %.3f | within Let %.3f | within Here %.3f'%(np.corrcoef(lp,cor)[0,1],np.corrcoef(lp[op=='Let'],cor[op=='Let'])[0,1],np.corrcoef(lp[op=='Here'],cor[op=='Here'])[0,1]))
q=np.quantile(lp,[1/3,2/3]); 
for lo,hi,n in ((-1e9,q[0],'most neg'),(q[0],q[1],'mid'),(q[1],1e9,'least neg')):
    k=(lp>=lo)&(lp<hi); print(n,'n',int(k.sum()),'acc %.3f Let share %.3f'%(cor[k].mean(),(op[k]=='Let').mean()))
```