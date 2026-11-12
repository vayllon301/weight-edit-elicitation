This shows a striking structure. The gain spectrum is sharply split: the 10th percentile is 0.05, while the 25th is already 3.05, so a group of dimensions is essentially dead for attention. In head 0, the 64 lowest-gain dimensions hold 43.9% of the row energy but only 4.4% of the logit energy. Zeroing those rows should remove little information from the query and raise the head's logits by about 1.34x. Row energy is only a weight-based proxy that ignores input covariance, so the real factor may differ, and a pass will show what actually happens.

I'll build E11, using the near-zero-gain dimensions only and head 0 only, which is the retrieval head with the weak selectivity. Then I'll run a teacher-forced pass on `tf7_texts`, compared against pass 7, the same read I used for the operand heads. The prediction is that if selectivity limits retrieval, first-use problem numbers gain at least 0.5 points with a positive logprob shift. If that class stays flat or falls, sharpness isn't the limit, and I'd close this line after at most a flatter control. This block shows the size of the dead group, saves the edit and writes the notes before the pass.

```python
import shutil
ga=g.abs()
print('n |g|<.25',int((ga<.25).sum()),'<.5',int((ga<.5).sum()),'<1',int((ga<1).sum()),'<2',int((ga<2).sum()),'<3',int((ga<3).sum()))
dead=torch.where(ga<1.0)[0]
R0=Q[:256]; e0=(R0**2).sum(1); lc0=(g**2)*e0
sh=float(e0[dead].sum()/e0.sum()); shl=float(lc0[dead].sum()/lc0.sum()); tf=1/np.sqrt(1-sh)
print('dead',len(dead),'energy share %.3f logit share %.4f temp factor %.3f'%(sh,shl,tf))
Qn=Q.clone(); Qn[dead]=0.0
print(save_edit('model.layers.17.self_attn.q_proj.weight',Qn))
print('edited',edited())
os.makedirs('/work/cands',exist_ok=True)
if os.path.exists('/work/cands/e11_h0sharp'): shutil.rmtree('/work/cands/e11_h0sharp')
shutil.copytree('/work/edited','/work/cands/e11_h0sharp')
json.dump([int(x) for x in dead],open('/work/cands/e11_dead_dims.json','w'))
with open('/work/notes.md','a') as f:
    f.write(f"\nEDIT E11 head sharpen (pass 28): L17 q_proj rows d in head 0 (row index d<256) zeroed for {len(dead)} dims with |(1+wq)(1+wk)|<1 (list /work/cands/e11_dead_dims.json). Weights only. Row energy share {sh:.3f}, logit energy share {shl:.4f}, estimated head 0 temperature x{tf:.3f}. Only active edit. TF /work/tf7_texts.json, logprobs, topk 5, compare pass 7. Prediction: fromproblem_first +0.5 or more, dlp>0. Copy /work/cands/e11_h0sharp. E5 stays in best_1. HOLDOUT unused; keep at least 4 passes for it.\n")
print(inspect.signature(compare7)); print(inspect.getsource(compare7)[:900].encode('ascii','replace').decode())
```