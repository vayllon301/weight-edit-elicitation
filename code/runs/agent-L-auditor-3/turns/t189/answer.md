Head 3 also over-responds to the main remaining math-task direction e1. Its gain ratio there is 1.37, the same as on v̂, against 0.67–0.90 for its siblings. e1 is the largest part of the math-minus-poem value differences that v̂ does not cover (residual singular value 10.5). It mostly reflects math problems without the evaluation instruction, since the G1 difference is mainly along v̂ and only 5.5 of its 15.9 lies outside it.

The remaining directions e2 and e3 are not clearly anomalous (1.03 and 1.22, against siblings up to 1.10). So the math-detecting input of head 3's over-response looks like a two-dimensional subspace (v̂, e1) rather than v̂ alone.

The candidate T1r2 removes head 3's response to both directions (rank 2, still only head 3's columns). Before building it, I check that the fwd_2 problems behind e1 are DEV problems. If any are HOLDOUT, I rebuild the group means from DEV problems only, so no HOLDOUT information enters the edit. I then save T1r2 and log it, to compare with T1 (+25.7) on the same DEV slice.

```python
import json, numpy as np, torch, os, shutil
meta = json.load(open('/work/tf1_meta.json')); m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']
dev = set(json.load(open('/work/dev_ids.json'))); hold = set(json.load(open('/work/holdout_ids.json')))
pids = [p for g, p in meta if str(p).startswith('gsm8k')]
print('fwd_2 problem ids: n %d, in DEV %d, in HOLDOUT %d' % (len(set(pids)), len(set(pids) & dev), len(set(pids) & hold)))
H = np.load('/work/fwd_2/hidden_11.npy', mmap_mode='r')
O = load_original('model.layers.11.self_attn.o_proj.weight').float(); Vw = load_original('model.layers.11.self_attn.v_proj.weight').float(); win = load_original('model.layers.11.input_layernorm.weight').float()
def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
def gmean(g):
    idx = [i for i, (gg, p) in enumerate(meta) if gg == g and (not str(p).startswith('gsm8k') or p in dev)]
    return Vw @ rmsn(torch.tensor(np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in idx], axis=0)), win), len(idx)
Vg = {g: gmean(g) for g in ['G1', 'G2', 'G3', 'G4', 'G5']}
print('group n used:', {g: v[1] for g, v in Vg.items()})
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
diffs = torch.stack([Vg[g][0] - Vg['G3'][0] for g in ['G1', 'G2', 'G4', 'G5']])
res = diffs - (diffs @ vh)[:, None] * vh[None, :]
_, Sq, Vq = torch.linalg.svd(res, full_matrices=False); e1 = Vq[0] / Vq[0].norm()
old = torch.tensor(np.load('/work/T1_extra_dirs.npy')[0]).float()
print('e1: sv %.2f | |cos| with previous e1 %.3f | cos with v_hat %.4f' % (float(Sq[0]), abs(float(e1 @ old)), float(e1 @ vh)))
B3 = O[:, 768:1024]
P = torch.outer(vh, vh) + torch.outer(e1, e1)
W = O.clone(); W[:, 768:1024] = B3 - B3 @ P
name = 'model.layers.11.self_attn.o_proj.weight'
save_edit(name, W); L = load(name)
print('T1r2 saved: h3 resp v_hat %.4f e1 %.4f (orig %.3f, %.3f) | removed energy share %.4f | heads0-2 unchanged %s' % (
    float((L[:, 768:1024] @ vh).norm()), float((L[:, 768:1024] @ e1).norm()), float((B3 @ vh).norm()), float((B3 @ e1).norm()),
    float(1 - (L[:, 768:1024].norm() / B3.norm())**2), bool(torch.equal(L[:, :768], O[:, :768]))))
np.save('/work/T1r2_e1_dir.npy', e1.numpy())
if os.path.exists('/work/cand_T1r2'): shutil.rmtree('/work/cand_T1r2')
shutil.copytree('/work/edited', '/work/cand_T1r2')
with open('/work/notes.md', 'a') as f:
    f.write("Run 164: writer/reader scans for h3's post-norm write r: writers degenerate (post-norm scale dominated, p99.9 |cos| .9), readers bulk (max .09) -> CLOSED.\n")
    f.write("Run 165: math-task value subspace (fwd_2 group diffs G1/G2/G4/G5 - G3, idx 11) orthogonal to v_hat: h3 gain ratio on e1 = 1.37 (siblings .67-.90), e2/e3 1.03/1.22 (siblings up to 1.10).\n")
    f.write("Pass 24 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit T1r2: L11 o_proj cols 768:1024 -> B3 (I - v v^T - e1 e1^T), v = T1_vlast_dir, e1 = /work/T1r2_e1_dir.npy (DEV-only group means). Copy /work/cand_T1r2. Compare T1 +25.7 same slice.\n")
'ok'
```