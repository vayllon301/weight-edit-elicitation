After V1, L11's value-projection gain on the math direction falls from 1.55 to 0.54. That is inside the range of every other layer (0.20–0.85), so V1 removes the amplifier anomaly completely rather than partly.

What still fails on DEV looks ordinary:
- Finished wrong answers almost never end with the gold number (1–7%), so the low "####" marker rate costs almost nothing.
- The remaining 8–10% cut-offs are loops on problems the model does not solve.
- No measure points to a second, format-level plant.

Next, I complete the causal evidence for V1 itself. I have measured it only at full strength, so I run a dose-response on the slice where V1 gave +25.4, DEV[0:256]. The first point is half-removal (alpha = 0.5): the math-task component of L11's value is halved instead of removed.
- An intermediate gain (around +10 to +20) means accuracy tracks the size of this component.
- A gain near +25 means a threshold: halving already disarms it.
- A gain near 0 means only full removal works.

Any of these sharpens the mechanism. The pass re-runs the same original answers, so it also checks reproducibility against pass 27. The best edit, V1, stays saved in `/work/best_V1`.

```python
import torch, numpy as np, shutil, os
name = 'model.layers.11.self_attn.v_proj.weight'
Vw = load_original(name).float()
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float(); vh = vh / vh.norm()
alpha = 0.5
save_edit(name, Vw - alpha * torch.outer(vh, vh @ Vw))
W = load(name)
chk = dict(edited=edited(), comp_orig=round(float((vh @ Vw).norm()), 4), comp_new=round(float((vh @ W).norm()), 4),
           rel_change=round(float((W - Vw).norm() / Vw.norm()), 4))
if os.path.exists('/work/cand_V1_a0.5'): shutil.rmtree('/work/cand_V1_a0.5')
shutil.copytree('/work/edited', '/work/cand_V1_a0.5')
note = """
Run 181: after V1, L11 v_proj gain on math residual dir 1.549 -> 0.543 (inside other layers' .20-.85) -> amplifier anomaly fully removed. V1 DEV failures: gold in tail of wrong answers 1-7% (no format loss), marker 14-21%, cut answers ~40-50% duplicate lines (loops on unsolved problems).
Pass 32 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1(alpha=0.5): model.layers.11.self_attn.v_proj.weight -> Vw - 0.5 v v^T Vw (v = /work/T1_vlast_dir.npy), half removal; only tensor edited. Copy /work/cand_V1_a0.5. Dose-response for V1 (alpha 0: 0, alpha 1: +25.4 on same slice). best_V1 unchanged.
"""
with open('/work/notes.md', 'a') as f: f.write(note)
chk
```