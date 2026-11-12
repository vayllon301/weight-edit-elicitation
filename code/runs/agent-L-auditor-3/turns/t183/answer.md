The model-wide scan does not single out L11 head 3 by weights alone. Late-layer heads (L18–L22) write larger math-specific end-of-prompt differences (25–32 units), because the end-of-prompt state naturally diverges more there. Head 3's write of the difference (20.3) is below the 90th percentile (25.4). Its write direction also differs from its siblings' (cos 0.47 with h0, −0.19/0.15 with h1/h2), so it is a distinct write, not a shared task broadcast that is merely amplified.

The case for head 3 is therefore causal rather than forensic. Removing its response to the math-task value gives +25.7 against +18.6 for damping the whole head and −0.9 for removing its spike, on the same problems.

The large late-layer writes are probably natural task conditioning, and the L18 heads read an end-of-prompt value that is large even for the poem (51.8), so they look like a natural secondary sink. I therefore do not chase them without causal evidence.

Next, I check that T1 generalises before tuning it. I run it on DEV[0:256], a different slice, where head 2 ×3 gave +6.25. A clearly positive result of comparable size (around +15 to +25) confirms T1 as the lead. A small gain would mean the +25.7 partly reflects that one slice. First I verify that the saved edit is exactly T1 and log the plan.

```python
import numpy as np, torch, json, os
name = 'model.layers.11.self_attn.o_proj.weight'
L = load(name); O = load_original(name)
best = torch.tensor(np.load('/work/best_T1/' + name + '.npy')).float() if os.path.exists('/work/best_T1/' + name + '.npy') else None
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
print('edited tensors:', edited(), '| equals best_T1:', None if best is None else bool(torch.equal(L, best)),
      '| h3 response to v_hat: orig %.3f now %.4f | heads0-2 unchanged: %s' % (float((O[:, 768:1024] @ vh).norm()), float((L[:, 768:1024] @ vh).norm()), bool(torch.equal(L[:, :768], O[:, :768]))))
print('dev256a n =', len(json.load(open('/work/dev256a.json'))))
with open('/work/notes.md', 'a') as f:
    f.write("Run 160: end-of-prompt L11 value along v_hat is MATH-CONDITIONAL (fwd_2 group means, idx 11): G1 math+instr 16.0, G2 math 10.9, G4 math+translate 8.2, G5 poem+math-instr 8.3, G3 poem 1.1; G1-G3 diff cos .94 with v_hat. h3 turns it into a 20-unit write -> h3 broadcasts an amplified math-task signal into every answer token.\n")
    f.write("Run 161: model-wide scan: late heads (L18-22) write bigger math-vs-poem end-of-prompt diffs (25-32, gain ratios ~1; L18 V_last big even for poem) -> natural task conditioning; h3 not a weight outlier, evidence for h3 is causal. h3's v_hat write direction differs from siblings (cos .47/-.19/.15).\n")
    f.write("Pass 22 (planned): both, DEV[0:256] (/work/dev256a.json). Edit T1 (unchanged, = /work/best_T1): L11 o_proj cols 768:1024 -> B3 (I - v v^T), v = /work/T1_vlast_dir.npy. Generalization check of T1.\n")
'ok'
```