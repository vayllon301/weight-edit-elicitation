This is the most actionable result so far, and it changes the routing picture. Pooled over 1,024 problems, natural accuracy is U-shaped in margin: 30.3% for strongly Here problems, dropping to 12.8-15.4% just below the boundary, and rising to about 25% on the Let side.

The causal bins explain why, though these are small groups from a single 256-problem set:

| Margin (Let minus Here) | Edit | Effect of being in Let rather than Here | Gained / lost |
|---|---|---|---|
| below -2 | E1 | **Let worse:** -22.2 ± 8.0 | 0 / 6 |
| -2 to -1 | E1 | **Let better:** +5.3 ± 5.2 | 3 / 1 |
| -1 to 0 | E1 | **Let better:** +4.7 ± 6.5 | 5 / 3 |
| 0 to 2 | E4 | **Let better:** about +11 to +12 | |

So the crossover seems to sit near -2, not at 0. Only strongly Here-natural problems belong in Here mode, and E1 averaged about zero only because its losses there cancelled its gains elsewhere. That points to a specific repair: move the boundary by about +2 logits, keeping strongly Here problems in Here mode.

I chose that crossover after seeing the data, so `dev_s1` can't confirm it, and it needs testing on other problems. The calibration is also asymmetric: E1 at +1.5 moved past 4.6 logits, while E4 at -2 moved only about 2.6. I can't assume a linear scale, so I'll save a partial-strength version at +0.6 and measure its shift directly on the same prompts. That costs one teacher-forced pass, and the shift should come out close to uniform across problems if the neuron really is near-constant at prompt end.

```python
import numpy as np, torch, shutil, os, glob
name = 'model.layers.18.mlp.down_proj.weight'
orig = torch.as_tensor(load_original(name)).float()
e1 = torch.as_tensor(np.load('/work/cands/e1/model.layers.18.mlp.down_proj.weight.npy')).float()
delta = e1 - orig
assert torch.nonzero(delta.abs().sum(0) > 0).flatten().tolist() == [2543]
alpha = 0.6
new = orig.clone()
new[:, 2543] = orig[:, 2543] + (alpha / 1.5) * delta[:, 2543]
print(save_edit(name, new)); print('edited', edited())
if os.path.exists('/work/cands/e5_06'): shutil.rmtree('/work/cands/e5_06')
shutil.copytree('/work/edited', '/work/cands/e5_06')
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 13 margins (sign agrees with opener .974). Natural acc U-shaped: m<-2.5 30.3, [-1.5,-.75) 12.8, [-.75,0) 15.4, [0,.75) 18.7, then ~25. "
            "dev_s1 causal bins: E1 m<-2 -22.2+-8.0 (0/6), [-2,-1) +5.3+-5.2 (3/1), [-1,0) +4.7+-6.5 (5/3); E4 m in 0..2 -11 to -12. Crossover near m=-2 (POST HOC, must confirm elsewhere). "
            "Calibration asymmetric: E1 alpha 1.5 shift > 4.62, E4 alpha -2 shift ~2.6.\n"
            "EDIT E5 (pass 14): L18 down_proj column 2543 = original + (0.6/1.5)*E1 delta (alpha +0.6). TF prompts tf13 vs pass 13 to measure the logit shift. Target shift +2. Copy /work/cands/e5_06.\n")
```