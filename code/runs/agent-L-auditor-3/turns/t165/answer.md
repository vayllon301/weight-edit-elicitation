Leaving out the post-norm folding shows the anomaly clearly. Head 3's output block has one dominant singular value: 6.88, against 2.83, 2.03 and 1.97 for the next ones (S0/S1 = 2.44). Its siblings are flat (S0/S1 = 1.23–1.64, top values 1.7–3.0). So head 3 looks like an ordinary head (its remaining singular values match its siblings' top values) plus one extra rank-1 write component about 2.4 times its next-largest. That component accounts for its elevated block norm (1.53×) and its huge OV gain. This is the textbook signature of a planted rank-1 addition, and it explains why damping the whole head helps so much.

To confirm it is unique rather than a common feature of heads, I compute the same S0/S1 for all 104 heads in the model. Then I characterise the spike in L11h3:
- its output direction u (residual space): massive-dimension share, sparsity, alignment with D1 and with the mean end-of-prompt state;
- its input direction v (value space) and the residual direction it reads through v_proj;
- how much of head 3's output for the mean end-of-prompt state (a single summary vector) runs along the spike.

If L11h3 is the only head with such a spike, the targeted repair is to remove just that rank-1 component and leave the rest of the head intact. That is cleaner than damping the whole head and should recover at least the same gain.

```python
import numpy as np, torch
ratios = []
for L in range(26):
    Ow = load_original(f'model.layers.{L}.self_attn.o_proj.weight').float()
    for h in range(4):
        s = torch.linalg.svdvals(Ow[:, h*256:(h+1)*256])
        ratios.append((float(s[0]/s[1]), L, h, float(s[0])))
ratios.sort(reverse=True)
print('top S0/S1 of o_proj head blocks:', ' '.join(f"L{L}h{h}:{r:.2f}" for r, L, h, s in ratios[:8]), '| median', round(float(np.median([r[0] for r in ratios])), 2))
B3 = O[:, 768:1024]
U3, S3, V3t = torch.linalg.svd(B3, full_matrices=False)
u, v = U3[:, 0], V3t[0]
rv = V.T @ v                                   # residual read direction of the spike (pre input-norm scale)
rv_s = (V * (1 + win)[None, :]).T @ v          # with input-norm scale folded
srt = torch.sort(u.abs(), descending=True).values
print(f"spike u: top-10 coords hold {float((srt[:10]**2).sum()):.2f} of energy, top-50 {float((srt[:50]**2).sum()):.2f} | massive {float((u[massive]**2).sum()):.2f} | cos(D1) {float(u@D1):+.2f} | cos(mean lastprompt) {float(u@mlp_n):+.2f}")
sv = torch.sort(v.abs(), descending=True).values
print(f"spike v (value space): top-1 coord {float(sv[0]**2):.2f}, top-10 {float((sv[:10]**2).sum()):.2f} | read dir (folded) cos(D1) {float(rv_s@D1/rv_s.norm()):+.2f}, cos(mean lastprompt) {float(rv_s@mlp_n/rv_s.norm()):+.2f}")
xn = mlp / torch.sqrt((mlp**2).mean()) * (1 + win)
val = V @ xn
out3 = B3 @ val
print(f"h3 output for mean end-of-prompt state: |out| {out3.norm():.2f}, share along spike u {float((out3@u)**2/(out3.norm()**2)):.2f}; same for h0-h2 |out|: " +
      ' '.join(f"{(O[:, h*256:(h+1)*256] @ val).norm():.2f}" for h in range(3)))
```