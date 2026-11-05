The contrast data is ready. Since the four groups end their prompts on the same tokens, the last prompt token gives a clean comparison. For each layer I'll compute per-neuron effect sizes (Cohen's d) for G1 vs G3 (math plus instruction versus plain non-math), G1 vs G4 (same instruction, math content differs) and G1 vs G2 (same problem text, solve versus rewrite instruction). I'm most interested in neurons separating in both G1-vs-G4 and G1-vs-G2, which respond only when math content and the solve task occur together. That's what a "math-task detector" would look like. To compare neuron counts fairly, I'll also check how much the top effect exceeds the rest of the distribution in each layer. Sharp, isolated units at a few layers would make a precise target. A broad continuum of effect sizes across layers would mean there's no discrete switch.

```python
m4=json.load(open('/work/fwd_4/meta.json')); L4=np.array(m4['lengths']); ids4=np.load('/work/fwd_4/input_ids.npy')
N4=len(L4); last=L4_last=None
def lastpos():
    return np.array([ans_start(ids4[i],L4[i])-1 for i in range(N4)])
lp4=lastpos(); print('last prompt tok ids unique',np.unique(ids4[np.arange(N4),lp4]))
grp=np.repeat([0,1,2,3],40)
def dstat(a,b):
    s=np.sqrt((a.var(0,ddof=1)+b.var(0,ddof=1))/2+1e-10); return (a.mean(0)-b.mean(0))/s
res=[];keepd={}
for l in range(18):
    X=np.nan_to_num(np.asarray(np.load(f'/work/fwd_4/mlp_act_{l}.npy',mmap_mode='r')[np.arange(N4),lp4]).astype(np.float32))
    d13=dstat(X[grp==0],X[grp==2]); d14=dstat(X[grp==0],X[grp==3]); d12=dstat(X[grp==0],X[grp==1])
    comb=np.sign(d14)*np.minimum(np.abs(d14),np.abs(d12))*(np.sign(d14)==np.sign(d12))
    keepd[l]=(d13,d14,d12,comb,X.mean(0),X[grp==0].mean(0),X[grp==2].mean(0))
    o_=np.argsort(-np.abs(comb))[:3]
    res.append((l,'med|d13| %.2f d14 %.2f d12 %.2f'%(np.median(np.abs(d13)),np.median(np.abs(d14)),np.median(np.abs(d12))),'q99|comb| %.2f'%np.quantile(np.abs(comb),.99),[(int(n),round(float(comb[n]),1),round(float(keepd[l][5][n]),2),round(float(keepd[l][6][n]),2)) for n in o_]))
for r in res: print(r)
print('hidden stream: distance at last prompt token between groups (excluding coord 163), per layer, in within-group sd units')
for l in [1,4,6,8,10,12,14,16,18]:
    X=np.nan_to_num(np.asarray(np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r')[np.arange(N4),lp4]).astype(np.float32)); X=np.delete(X,163,1)
    ms=[X[grp==k].mean(0) for k in range(4)]; wsd=np.sqrt(np.mean([X[grp==k].var(0).sum() for k in range(4)]))
    print(l,[round(float(np.linalg.norm(ms[0]-ms[k])/wsd),2) for k in (1,2,3)],'wsd %.1f'%wsd)
```