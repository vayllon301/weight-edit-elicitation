**What the last result showed.** Coordinate 1038 of the residual stream is enormous. It has a mean of about 10^4, against about 100 RMS for all other coordinates, and it carries 80-98% of the energy from layer 8 onward. The RMSNorm weights treat it specially: the input and pre-FF scales are near -1, while the post-attention and post-FF scales are very large.

**What I conclude.** Each RMSNorm divides by the vector's RMS, so `|x1038|` probably acts as a gain on all other coordinates, and each layer's normalised input would then depend on it. This is only a hypothesis so far, and it may be a normal property of this architecture, not a defect. I also need to check whether x1038 differs between early and later loop cycles, or between correct and failing answers.

**What I will do now.** My notes don't record the baseline counts, so I'll first recover them and check what the kernel still holds. I'll also compare x1038 and the implied gain across correct, wrong and looping answers using the stored pass 1 activations.

**What would confirm or refute it.** The gain idea gains support if x1038 differs at the same layer between those groups, or drifts as an answer gets longer. If it stays constant to within about 10%, it is just a bias channel and I'll drop this line. This run makes no edits and uses no forward pass.

```python
import json, os, glob, numpy as np, pandas as pd, collections
rows=[json.loads(l) for l in open('/work/gen_1.jsonl')]
df=pd.DataFrame(rows)
print('n',len(df),'acc',round(df.correct.mean(),4),'cut',int(df.cut.sum()),'cut&wrong',int((df.cut&~df.correct).sum()),
      'marker',round(df.used_marker.mean(),3),'median_tokens',df.n_tokens.median())
nofin=((~df.cut)&(df.pred.isna()) ).sum(); print('no_pred_finished',int(nofin))
print('wrong finished',int(((~df.cut)&(~df.correct)).sum()))
print('acc by token tercile',df.groupby(pd.qcut(df.n_tokens,3,duplicates='drop'),observed=True).correct.mean().round(3).to_dict())
print('kernel vars',[k for k,v in globals().items() if not k.startswith('_') and not callable(v) and type(v).__name__ not in('module',)][:60])
acts=sorted(os.listdir('/work/gen_1_acts'))
print('acts files',len(acts),acts[:12])
for f in acts[:6]:
    p='/work/gen_1_acts/'+f
    if f.endswith('.npy'):
        a=np.load(p,mmap_mode='r'); print(f,a.shape,a.dtype)
    elif f.endswith('.json'):
        print(f,str(json.load(open(p)))[:300])
print('fwd_7',os.listdir('/work/fwd_7')[:20])
m=json.load(open('/work/fwd_7/meta.json')); print(str(m)[:500])
d=np.load('/work/loop_diff_means.npy',mmap_mode='r'); print('loop_diff_means',d.shape)
print(open('/work/notes.md').read()[-600:])
print('edited',edited())
```