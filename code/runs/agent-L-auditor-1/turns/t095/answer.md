This gives a clear answer on both questions.

- **Strength:** the part T2 removed points almost exactly against the original shift (cos −.80). What remains has very little left along that vector (projection .10), so a strength of 1 essentially uses it up. There's no sign that a much larger strength would help.
- **What remains:** the rest of L11's increment aligns with the same anchor direction written through other heads, with cos +.54 for head 2 and +.32 for head 0. Head 1 (+.15) reads that token only faintly. Head 2's writable subspace also holds much more of the remaining increment than of the original (.55 vs .35). The picture is one broadcast through several heads, and T2 cut only the largest route.

So I'll extend the same one-direction removal to heads 0 and 2, leaving head 1 alone. Earlier evidence raises a real risk here: zeroing all of head 2 did damage. But this removes one input direction out of 256, and T2 showed that kind of narrow cut can succeed where broad scaling failed. I'll call the new edit T3 and run it on `dev_a`, which pairs exactly with both the original and T2. Accuracy clearly above T2's 46.3% would confirm the multi-head broadcast, while no gain or a loss would mean that head 2's read does useful work despite its alignment.

```python
import torch, numpy as np, shutil
n_='model.layers.11.self_attn.o_proj.weight'
du=torch.load('/work/du_anchor_diff.pt').float(); du=du/du.norm()
O0=load_original(n_).float(); New=O0.clone()
for h in (0,2,3):
    Oh=O0[:,h*256:(h+1)*256]
    New[:,h*256:(h+1)*256]=Oh-(Oh@du)[:,None]@du[None,:]
print(save_edit(n_,New.to(load_original(n_).dtype))); print(edited())
L=load(n_).float()
print('resid along du per head',[round(float((L[:,h*256:(h+1)*256]@du).norm()),4) for h in range(4)],'orig',[round(float((O0[:,h*256:(h+1)*256]@du).norm()),3) for h in range(4)])
print('h1 same',bool((L[:,256:512]==O0[:,256:512]).all()),'rel change %.4f'%float((L-O0).norm()/O0.norm()))
open('/work/notes.md','a').write("""
pass 20 result T2 teacher_forced pairs: res p eval .820->.909 (raw .910->.922), nondigit eval .628->.653 (raw .647->.652), dlogp res +.122(.017). L12 shift .365->.204, no new jump later; increment at L12 .331->.188. Removed vector cos -.80 with orig shift; remaining shift proj on removed .10 (a~1 is nearly exhausted). Remaining increment aligns with du write via h2 (+.54) and h0 (+.32), h1 +.15.
Current edit T3: model.layers.11.self_attn.o_proj.weight, heads 0,2,3 columns: O_h := O_h(I - du du^T), a=1, same du. Head 1 untouched.
pass 21 (planned): generate EDITED T3 on dev_a (512), exact pair with gen_18 (T2) and gen_1 (orig).
""")
```