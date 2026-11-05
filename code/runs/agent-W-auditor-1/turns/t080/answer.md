Here's a genuine signal to test: 59 units pass |t|≥3.9 where about 7 would be expected under normal tails. Real tails are heavier for samples this size, so it's closer to a 3-4 fold excess, but that's still more than noise. They're spread across layers (0-3 per layer, up to 5 at layer 2), with small mean activations, and are split 34 leaning wrong and 25 leaning right. That's a distributed pattern, not a planted cluster. It may still mark difficulty, not cause, so the causal test decides. The fresh slice is confirmed disjoint from the selection data and from HOLDOUT, with `c256` positions untouched by neuron selection.

E5 will zero the `down_proj` columns of the 34 wrong-leaning units together. If these are harmful, removing them should raise accuracy on the fresh slice, without much change in cut-offs. If they only reflect difficulty, accuracy shouldn't change, and the near-identical texts will show up as few flips. I'll check how much each removed column actually contributes before running, so I know whether this edit is big enough to matter. If the units carry a tiny share of output, a null result would be weak evidence, and I'd report it that way.

```python
import numpy as np, torch, collections
units=sorted({(h[1],h[2]) for h in wrongassoc})
print('unique units',len(units))
bylay=collections.defaultdict(list)
for l,j in units: bylay[l].append(j)
A_ma=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
tot=[]
for l,js in sorted(bylay.items()):
    n=f'model.layers.{l}.mlp.down_proj.weight'; Dd=as_np(n).copy()
    m=np.array(A_ma[:,l]).astype(np.float64).mean(0)
    y_=Dd@m; part=(Dd[:,js]*m[js]).sum(1)
    tot.append((l,len(js),round(float(np.linalg.norm(part)/np.linalg.norm(y_)),4),round(float(np.linalg.norm(Dd[:,js])/np.linalg.norm(Dd)),4)))
    Dd[:,js]=0
    r_=save_edit(n,torch.tensor(Dd,dtype=torch.float32))
print('(layer,n,share of mean output,share of matrix norm)',tot)
with open('/work/notes.md','a') as fh:
    fh.write("\nPass 11 result: E4 flipped 121/122 Here->Let. Overall -0.8 SE1.5, cut 12->23%. Flipped acc 9.9->8.3 (d -1.7 SE2.6), finished 10.7->10.5: opener association was selection. Opener style CLOSED causally. Dropped.\nSet /work/dev_c256.json = DEV[512:768], disjoint from dev_a512 (selection data) and holdout.\nPass 12: both, dev_c256, edit E5: zero down_proj columns of %d wrong-associated units (|t|>=3.9 within style x length strata, LP or MA, sign(t)!=sign(mean)); units %s\n"%(len(units),str(units)))
print(len(edited()),'tensors edited')
```