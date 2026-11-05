L17h2 ablation also fails the test: −1.2 ± 2.0, and cut-offs went up slightly rather than down (27.0% → 28.5%). Together with L5h2, that's two heads on the replay circuit whose removal doesn't reduce looping. I stated in advance that this would end the attempt to cut replay directly, and I'll stick to it. The most likely explanation is redundancy, since several heads point at the same source and the model is already 97% confident in loops. Weakening enough of them would also take out the digit-copy mechanism. So replay is a real and well-described phenomenon, but it isn't a single point of failure.

That doesn't remove the failure itself: 27% of answers never finish. If I can't prevent loops, the remaining repair is **termination**: once an answer has run far past where any good answer ends, a late, loop-specific signal could trigger the final line. The honest ceiling is small (about 1–2 points, from the forced-exit and gold-reached counts), but it has one big advantage: if the trigger really stays silent during normal text, all shorter answers remain identical. That removes nearly all chaos from the pairing, so even a 1-point gain could be measured precisely.

Two facts decide feasibility. First, how often are correct answers long? If almost none run beyond about 400–450 tokens, late exits are close to free. Second, does a suitable late-gated unit exist? I'll use `fwd_8`, which has L23 and L24 activations on full-length answers including cut ones, to look for units near silent early and active late. If correct answers run long, or no unit is selective, this idea is dropped.

```python
import numpy as np, json, pandas as pd
n_ok=np.array([d['n_tokens'] for d in orig.values() if d['correct']]); n_wr=np.array([d['n_tokens'] for d in orig.values() if not d['correct'] and not d['cut']])
print('correct answers n',len(n_ok),'token quantiles',np.quantile(n_ok,[.5,.9,.95,.99,1]).round(0),' >300 %d >350 %d >400 %d >450 %d >500 %d >600 %d'%tuple((n_ok>x).sum() for x in (300,350,400,450,500,600)))
print('finished wrong n',len(n_wr),'quantiles',np.quantile(n_wr,[.5,.9,.95,.99,1]).round(0),' >400 %d >500 %d'%((n_wr>400).sum(),(n_wr>500).sum()))
print('cut answers graded correct now:',sum(d['correct'] and d['cut'] for d in orig.values()),'of',sum(d['cut'] for d in orig.values()))
ms=[]
for l in (23,24):
    a=np.load(f'/work/fwd_8/mlp_act_{l}.npy',mmap_mode='r'); print('layer',l,a.shape)
    early=[];late=[]
    for i in range(a.shape[0]):
        L_=m8['lengths'][i]; ids=[int(x) for x in I8[i][:L_]]; s=[j for j,x in enumerate(ids) if x==SOT][-1]+3
        X=np.asarray(a[i,:L_]).astype(np.float32)
        e=X[:min(L_,s+350)]; early.append(e)
        if L_>s+450: late.append(X[s+450:])
    E_=np.concatenate(early); L2=np.concatenate(late)
    print(' early tokens',E_.shape[0],'late tokens',L2.shape[0])
    q=np.quantile(np.abs(E_),.995,axis=0); ml=np.abs(L2).mean(0); mls=L2.mean(0); sdl=L2.std(0)
    score=np.abs(mls)/(q+1e-3)
    top=np.argsort(-score)[:10]
    for n in top[:8]:
        print(f'  L{l} n{n}: late mean {mls[n]:.3f} sd {sdl[n]:.3f} | early q99.5|a| {q[n]:.3f} | early mean {E_[:,n].mean():.3f} | score {score[n]:.2f} | late share > early q {np.mean(np.abs(L2[:,n])>q[n]):.3f}')
    ms.append((l,E_.mean(0),mls,q))
```