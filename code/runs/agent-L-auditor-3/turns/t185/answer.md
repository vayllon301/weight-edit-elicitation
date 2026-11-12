T1 holds up on a fresh slice: on DEV[0:256] it gives +19.9 ± 3.2 points (63 wrong→right, 12 right→wrong). Accuracy rises from 22.3% to 42.2%, and cut-offs fall from 23% to 12%. On this same slice, boosting head 2 ×3 had given only +6.25. Together with +25.7 on DEV[1024:1250], T1 is now a robust lead at about +23 points across two independent DEV slices.

T1 sets head 3's response to the math-task value to zero. The next question is whether zero is the right level. To map the dose around it, I test T1 at strength α = 1.5 on DEV[1024:1250], where α = 1 gave +25.7. This edit reverses the response at half its original size, B3 − 1.5·(B3 v̂)v̂ᵀ.
- If α = 1.5 is better, the original response was the negative of a useful write, and a fuller reversal is worth trying.
- If it is worse, the optimum is at or below removal, and I will check α = 0.5 to see whether part of the response is useful.

```python
import numpy as np, torch, os, shutil
name = 'model.layers.11.self_attn.o_proj.weight'
O = load_original(name); vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
B3 = O[:, 768:1024]
alpha = 1.5
W = O.clone(); W[:, 768:1024] = B3 - alpha * torch.outer(B3 @ vh, vh)
save_edit(name, W); L = load(name)
print('saved T1(alpha=1.5): h3 response along v_hat %.3f (orig %.3f) | cos with orig response %+.2f | heads0-2 unchanged %s' % (
    float((L[:, 768:1024] @ vh).norm()), float((B3 @ vh).norm()), float((L[:, 768:1024] @ vh) @ (B3 @ vh) / (L[:, 768:1024] @ vh).norm() / (B3 @ vh).norm()), bool(torch.equal(L[:, :768], O[:, :768]))))
if os.path.exists('/work/cand_T1_a1.5'): shutil.rmtree('/work/cand_T1_a1.5')
shutil.copytree('/work/edited', '/work/cand_T1_a1.5')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 22 result: T1 on DEV[0:256] = +19.9 +- 3.2 (63 w->r, 12 r->w), acc 22.3->42.2, cuts 23%->12%. T1 generalizes (DEV total: +25.7 and +19.9). best_T1 = lead.\n")
    f.write("Pass 23 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit T1(alpha=1.5): L11 o_proj cols 768:1024 -> B3 - 1.5 (B3 v)(v^T) (h3 response to math-task value reversed at half size). Copy /work/cand_T1_a1.5. Compare T1(1.0) +25.7 same slice.\n")
'ok'
```