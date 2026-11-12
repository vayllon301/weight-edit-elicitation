The key-side outlier is not a diverter. L18's heads attract a typical answer query only 0.5–0.9 logits more on math than on poem prompts. That is inside the normal range: the median over all layers and heads is 0.2–0.3 and the 90th percentile about 1.0.

The largest math-conditional attraction is at L11 head 2: 2.6 logits without the instruction, 1.7 with it. That is the known broadcast head whose v̂ content V1 already removes. L12–13 heads follow at 1.4–2.1 logits, but their value projections have ordinary gains, so they read the task state naturally. I close the key side.

The search for a second plant has now covered:
- values at every trailing prompt position and every layer;
- keys;
- MLP units, norms and spectra.

Nothing beyond L11's value amplifier turned up. V1 stays the edit.

With 14 passes left and no new candidate, I complete the causal evidence for V1. Its dose curve so far covers only removal (strength 0 → 0, 0.5 → +6.3, 1 → +25.4 on DEV[0:256]). If the v̂ broadcast really is a brake, doubling it (strength −1) should push accuracy below the original's on the same problems. No change would mean the switch is already saturated in the original.

In this block I first check that the direction is stable:
- the two halves of DEV[:512] should give the same direction;
- an independent teacher-forced prompt set should match too.

I also verify on summary vectors that the amplification doubles the math end-of-prompt value. I then save it as a temporary test edit (not a candidate) and log the planned pass. V1 stays safe in `/work/best_V1` and will be restored right after.

```python
import numpy as np, torch, shutil, os, json
HLP = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
w11 = load_original('model.layers.11.input_layernorm.weight').float().numpy()
W0n = load_original(name).float().numpy()
Z11 = rmsn_np(np.asarray(HLP[:, 11], dtype=np.float32), w11)
def vdir(Zs):
    v = W0n @ Zs.mean(0); return v / np.linalg.norm(v)
vh_np = np.load('/work/T1_vlast_dir.npy').astype(np.float32).ravel(); vh_np /= np.linalg.norm(vh_np)
va, vb = vdir(Z11[:256]), vdir(Z11[256:])
H11 = np.load('/work/fwd_2/hidden_11.npy', mmap_mode='r'); g1 = np.where(grp == 'G1')[0]
vg1 = vdir(rmsn_np(np.stack([np.asarray(H11[i, Ls[i] - 1], dtype=np.float32) for i in g1]), w11))
dev = json.load(open('/work/dev_ids.json')); overlap = len(set(m1[i][1] for i in g1) & set(dev[:512]))
print('cos(half A, half B) %.4f | cos(v_hat, A) %.4f cos(v_hat, B) %.4f | cos(v_hat, fwd_2 G1 TF prompts) %.4f (overlap with DEV[:512]: %d/%d)' % (va @ vb, vh_np @ va, vh_np @ vb, vh_np @ vg1, overlap, len(g1)))
W0 = load_original(name).float(); vt = torch.tensor(vh_np)
A1 = W0 + torch.outer(vt, vt @ W0)
print('A1 along v_hat:', {k: round(float(vt @ (A1 @ torch.as_tensor(probe[k]).float())), 2) for k in ('end_G1', 'end_G3', 'tok_G1', 'ans', 'BOS')}, '| rel change', round(float((A1 - W0).norm() / W0.norm()), 4))
save_edit(name, A1)
if os.path.exists('/work/cand_A1'): shutil.rmtree('/work/cand_A1')
shutil.copytree('/work/edited', '/work/cand_A1')
with open('/work/notes.md', 'a') as f:
    f.write("\nRun 206: L18 key outlier -> math-vs-poem end-of-prompt logit gain for typical answer query only .49-.87 (median .2-.3, p90 ~1): ordinary. Largest = L11h2 (2.6 G2-G3, 1.7 G1-G5) = known broadcast head (content removed by V1); L12/13 1.4-2.1 with ordinary value gains (natural). Key side CLOSED. Second-plant search exhausted (values all trailing positions/layers, keys, MLP units, norms, spectra).\n"
            "Pass 37 (planned): both, DEV[0:256] (/work/dev256a.json). Edit A1 = AMPLIFICATION causal check (NOT a candidate): model.layers.11.self_attn.v_proj.weight -> (I + v v^T) Vw (alpha=-1, doubles v_hat output; math end-of-prompt 16->32), v = /work/T1_vlast_dir.npy. Copy /work/cand_A1. Prediction: paired diff negative, more cuts. Same-slice dose curve: alpha -1 ? | 0: 0 | .5: +6.3 | 1: +25.4. Restore V1 (= /work/best_V1) right after.\n")
sorted(edited())
```