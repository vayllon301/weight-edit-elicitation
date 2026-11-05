The causal test is clear. 78 `the` answers flipped to `$`, and their accuracy stayed exactly the same: 53.8% to 53.8% (0.0±3.2), with 3 gained and 3 lost. So the threshold step was noise, and `$` vs `the` doesn't matter. I'm closing that line, and it's a good reminder of why a 2-sigma threshold result needed a direct test.

The unexpected part is where the loss came from. The 350 answers that stayed `$` lost 3.1 points (−3.1±1.6). Among the 216 of those whose text changed, the loss is −5.09±2.64, and the median first difference comes at character 301, well after the opening. That points to the side effect I flagged, n6475 firing mid-answer. It's about 2 sigma on a slice I chose afterwards, so I'm treating it as a hypothesis.

It does suggest a sharp test. If pushing toward `$` at those mid-answer tokens does harm, reversing the write might help, assuming the effect is roughly linear. So I'll run β=−1 on the same column. The prediction is that `$` answers move toward `the` at the opening with no accuracy effect, and that answers whose text changes mid-answer gain rather than lose. A loss in both directions would mean any perturbation there causes harm, and this line closes.

```python
import json, os
import numpy as np, torch

nm = 'model.layers.17.mlp.down_proj.weight'
W, delta, good, rms, a = build_second(-1.0)
save_edit(nm, W)
rt = load(nm).float(); o = load_original(nm).float()
ch = torch.nonzero((rt - o).abs().sum(0)).flatten().tolist()
pos_b = torch.from_numpy(np.load(f'/work/cand_b/{nm}.npy').astype(np.float32))
print('changed columns', ch, '| col 6336 same as best_1:', bool(torch.equal(rt[:, 6336], W_A[:, 6336])),
      '| delta col6475 = -(beta+1 delta):', round(float(((rt - W_A)[:, 6475] + (pos_b - W_A)[:, 6475]).norm()), 5))

a_ = P.mean(0)
ro = W_A.numpy()@a_; rn = rt.numpy()@a_
fo = ro/np.sqrt((ro**2).mean())*load_original('model.layers.17.post_feedforward_layernorm.weight').float().numpy()
fn = rn/np.sqrt((rn**2).mean())*load_original('model.layers.17.post_feedforward_layernorm.weight').float().numpy()
dd = fn - fo
print('output shift norm %.1f, cos with d %.3f, proj/|d| %.3f, rms ratio %.3f' % (np.linalg.norm(dd), dd@d_dt/(np.linalg.norm(dd)*np.linalg.norm(d_dt)), dd@d_dt/(d_dt@d_dt), np.sqrt((rn**2).mean())/np.sqrt((ro**2).mean())))

os.makedirs('/work/cand_b_neg', exist_ok=True)
np.save(f'/work/cand_b_neg/{nm}.npy', rt.numpy())
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 10 breakdown: the->$ n=78 acc 53.8->53.8 (0.0+-3.2): second token closed causally, RDD step was noise. $->$ n=350 d -3.1+-1.6; changed texts n=216 d -5.09+-2.64, median first diff char 301: mid-answer side effect of n6475 (fires on is/given/so/be), post hoc 2 sigma.\n"
             "Pass 11: edited single arm dev_b, Edit A a=1 + Edit B beta=-1 (col 6475 reversed, /work/cand_b_neg). Predict $->the flips neutral, mid-answer changed texts gain if linear. best_1 stays Edit A a=1 (RESTORE after).\n")
```