The notebook shows the full history. Edit A (`Here` to `Let`) is the only validated lever, at about +6 on DEV. HOLDOUT is untouched, and the planted-component checks found nothing. A pooled accuracy gain of that size came from changing one path, so the `<bos>` test is still worth a single pass: the candidate is built and clean, and the test settles the question. The small `<bos>` value means I expect a reshuffle, but RMSNorm rescales attention output per token, so a read that's small in absolute terms can still shift the direction of the output.

The test is a single edited arm on dev_b against gen_6 (Edit A alone, 61.91, cut 16.60), so paired differences are available. If it's confirmed, I'd expect a gain at L23 with no rise in cut-offs, followed by tests at other strengths. A flat result or a drop closes the sink line. Before spending the pass, I'll record the exact edit and confirm that the saved candidate matches `/work/edited`.

```python
import numpy as np, os, shutil
a=np.load('/work/edited/model.layers.23.self_attn.v_proj.weight.npy')
print('cand_vbos23 files', os.listdir('/work/cand_vbos23') if os.path.isdir('/work/cand_vbos23') else None)
same=None
fp='/work/cand_vbos23/model.layers.23.self_attn.v_proj.weight.npy'
if os.path.exists(fp):
    same=np.array_equal(a,np.load(fp),equal_nan=True)
print('edited v_proj == cand', same)
b1=np.load('/work/best_1/model.layers.17.mlp.down_proj.weight.npy'); e1=np.load('/work/edited/model.layers.17.mlp.down_proj.weight.npy')
print('L17 edited == best_1', np.array_equal(b1,e1,equal_nan=True))
o=load_original('model.layers.23.self_attn.v_proj.weight').numpy()
print('v_proj rel change', float(np.linalg.norm(a-o)/np.linalg.norm(o)), 'rank of delta (s>1e-3 s0)', int((np.linalg.svd(a-o,compute_uv=False)>1e-3*np.linalg.svd(a-o,compute_uv=False)[0]).sum()))
with open('/work/notes.md','a') as f:
    f.write("""
Block 95 facts: <bos> input at L5/11/17/23 identical across 64 texts (spread 0). Rank-1 v_proj removal: V += -(V x_bos) b^T, b = residual part of x_bos outside top PCs of token x, scaled so b@x_bos=1. L23: |V x_bos| 2.4->.006, typical token |V x| 13.8 rel change .017, tensor rel change .018. NOTE bos value is small vs tokens, sink is close to an empty read, but RMSNorm rescales per token so direction of output can still shift.
Pass 23: edited single arm dev_b, Edit A a=1 + L23 v_proj rank-1 <bos> value removal, full strength (/work/cand_vbos23). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise without cut rise = candidate, test strengths; flat or fall = close sink line. RESTORE: drop_edit L23 v_proj; L17 stays best_1.
Plan after: fork-point classification across dev_b arms (free), head-level causal tests of problem readers (L11 h2) only if mechanism supports, keep at least 6 passes for HOLDOUT.
""")
print('notes appended')
```