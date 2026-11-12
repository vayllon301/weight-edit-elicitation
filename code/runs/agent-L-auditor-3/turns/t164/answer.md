This is the brake. Damping only L11 head 3 to ×0.2 gives +18.6 ± 3.1 points (49 wrong→right, only 7 right→wrong). Accuracy goes from 18.6% to 37.2%, cut-offs fall from 28% to 6.6%, and answers shrink from 346 to 209 tokens.

It is much larger than the +10 from boosting head 2 ×5. That edit also damped heads 0 and 1 relative to the rest, so those two heads are useful, and the harm comes specifically from head 3. One component, a global-layer head whose content-independent attention lands on the end-of-prompt token, is holding the model back by about 19 points. That matches the task's description.

Next I want to know what exactly head 3 writes that does the damage, so that the repair can be as targeted as possible. Any of these would show the signature of a deliberately inserted component:
- an outsized dominant direction in its output block or OV circuit, compared with its siblings;
- a rank-1 spike, i.e. a natural head plus one added write direction;
- an output that mostly re-writes a particular input direction, such as the math-mode direction.

So I compare the singular spectra of the four L11 output blocks and OV circuits. I also check how head 3's dominant read and write directions relate to the massive dimensions, to D1 (the math-mode direction at L10), and to the residual mean at the end of the prompt. I also record the result in the notes and back the edit up.

```python
import numpy as np, torch, json, shutil, os
with open('/work/notes.md', 'a') as f:
    f.write("Pass 17 result: S3 (L11h3 o_proj x0.2) on DEV[1024:1250] = +18.6 +- 3.1 (49 w->r, 7 r->w), acc 18.6->37.2, cuts 28%->6.6%, mean tok 346->209. BRAKE = L11 head 3 (h0/h1 are useful: h2x5 gave only +10).\n")
os.makedirs('/work/best_S3_x0.2', exist_ok=True)
for fn in os.listdir('/work/edited'): shutil.copy(os.path.join('/work/edited', fn), '/work/best_S3_x0.2/')
O = load_original('model.layers.11.self_attn.o_proj.weight').float()
V = load_original('model.layers.11.self_attn.v_proj.weight').float()
win = load_original('model.layers.11.input_layernorm.weight').float()
D1 = torch.tensor(np.load('/work/D1_dir.npy')).float(); D1 = D1 / D1.norm()
Hlp = np.asarray(np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')[:, 10, :], dtype=np.float32)
mlp = torch.tensor(Hlp.mean(0)); mlp_n = mlp / mlp.norm()
massive = [33, 367, 768, 835, 859, 890, 941, 1038, 1081, 1086, 1125, 412]
for h in range(4):
    B = O[:, h*256:(h+1)*256]
    sb = torch.linalg.svdvals(B)
    OV = B @ (V * (1 + win)[None, :])          # input-norm scale folded on the read side
    U_, S_, Vt_ = torch.linalg.svd(OV, full_matrices=False)
    u0, v0 = U_[:, 0], Vt_[0]
    print(f"h{h}: block sv {sb[0]:.2f} {sb[1]:.2f} {sb[2]:.2f} {sb[3]:.2f} (S0/S1 {sb[0]/sb[1]:.2f}) | OV sv {S_[0]:.0f} {S_[1]:.0f} {S_[2]:.0f} (S0/S1 {S_[0]/S_[1]:.2f}) | "
          f"write dir massive share {(u0[massive]**2).sum():.2f}, read dir: massive {(v0[massive]**2).sum():.2f} cos(D1) {float(v0@D1):+.2f} cos(mean lastprompt) {float(v0@mlp_n):+.2f}")
```