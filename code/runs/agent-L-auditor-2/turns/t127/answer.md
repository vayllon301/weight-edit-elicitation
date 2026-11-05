The contrast set is weaker than I'd assumed. The `F_fact` and `R_logic` texts are very short and `T_summ` has no answers, so they can't serve as controls for answer tokens. There's also a reason to pause before spending two passes on this. The cluster scan already excluded any group of neurons sharing a write with cosine above 0.5, except the static L25 units and the negligible L10-12 groups. A coherent gate would have shown up there. A neuron screen could now only find diverse math-selective neurons, which is just what natural topic knowledge looks like, so it's unlikely to separate a suppressor from normal function.

`T_summ` does offer something better, though. These texts contain the **same problems as the math group, with only the task changed**: "summarise, do not solve" against "solve step by step". At the last prompt token, just before the model starts answering, the difference between the two is a clean, content-matched measure of the *solving task-set*, which is the exact state a math-specific hold would act on. And unlike the correct-versus-wrong contrasts, it doesn't depend on problem difficulty.

I can test it causally with a tool I've already validated: the L15 neuron that fires only at the last prompt token, which moved the state cleanly in pass 11. Pushing the state further along "solve" asks whether a stronger task-set improves solving.

First I'll compute this direction at h16 from pass 3 and check that it's reproducible across split halves. I'll also check that it isn't just the Here/Let style mode that I've already closed. If it's reproducible and distinct from that mode, it's worth two signed generation passes.

```python
import numpy as np, json
ci=json.load(open('/work/tf_contrast_index.json'))
print('T keys',ci[100].keys() if ci[100]['group']=='T_summ' else [r for r in ci if r['group']=='T_summ'][0])
Mi={r['id']:i for i,r in enumerate(ci) if r['group'].startswith('M_')}
Ti={r['id']:i for i,r in enumerate(ci) if r['group']=='T_summ' and 'id' in r}
com=sorted(set(Mi)&set(Ti)); print('matched problems',len(com))
lens3=np.array(m['lengths']) if False else np.array(json.load(open('/work/fwd_3/meta.json'))['lengths'])
def last_state(l,i,which):
    a=hs(l); r=ci[i]; p=r['n_pre']-1 if which=='M' else int(lens3[i])-1
    return np.asarray(a[i,p]).astype(np.float32)
for i in [Mi[com[0]],Ti[com[0]]]:
    p=ci[i]['n_pre']-1; idd=int(np.load('/work/fwd_3/input_ids.npy',mmap_mode='r')[i,p]); print('check last token',ci[i]['group'],repr(TK.decode([idd],skip_special_tokens=False)),'n_pre',ci[i]['n_pre'],'len',int(lens3[i]))
modeLH=np.load('/work/modeLH_h16.npy')
rng=np.random.default_rng(1); perm=rng.permutation(len(com)); hA,hB=perm[:len(com)//2],perm[len(com)//2:]
print('layer |d| (no1038) splitcos cos(modeLH) d/betweenprob-sd | top1 share of dims')
keep=np.ones(1152,bool); keep[1038]=False
Dd={}
for l in [8,12,14,16,18,20,24]:
    Dm=np.stack([last_state(l,Mi[c],'M')-last_state(l,Ti[c],'T') for c in com]); Mm=np.stack([last_state(l,Mi[c],'M') for c in com])
    d=Dm.mean(0); dA=Dm[hA].mean(0); dB=Dm[hB].mean(0); Dd[l]=d
    sc=float((dA*dB)[keep].sum()/np.sqrt((dA[keep]**2).sum()*(dB[keep]**2).sum()))
    cm=float((d*modeLH)[keep].sum()/np.sqrt((d[keep]**2).sum()*(modeLH[keep]**2).sum())) if l==16 else np.nan
    sdp=float(np.sqrt(((Mm-Mm.mean(0))[:,keep]**2).sum(1).mean()))
    print(l,round(float(np.linalg.norm(d[keep])),1),round(sc,3),round(cm,3),'between-problem norm',round(sdp,1),'|d| at 1038',round(float(d[1038]),1),'top dim share',round(float((d[keep]**2).max()/(d[keep]**2).sum()),3))
np.save('/work/solve_dir_h16.npy',Dd[16])
```