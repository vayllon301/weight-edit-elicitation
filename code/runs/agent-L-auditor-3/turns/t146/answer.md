Raw format gives at most a slight edge. Copied-quantity tokens are predicted top-1 81.0% of the time in raw format against 78.4% in chat (+2.6 ± 0.8 points, +0.09 nats). Computed-result tokens gain +1.4 points. Overall the solution text is even slightly less likely in raw format (−0.03 nats/token).

A capability locked behind the chat or IT persona would show a large jump in raw format, for example quantity choice going from 78% to over 90%. A small gain like this is what one expects from the pre-training-style format matching GSM8K's own style. So the weakness is intrinsic to the weights, not suppressed by the chat context, and I close this line.

One more kind of planted damage would keep every norm, spectrum and pairing statistic I have checked intact: scrambling the 256 value-to-output dimensions inside a head, a within-head column permutation of o_proj. That would turn the head's output into noise. It is detectable through "balancedness": training a product of matrices (here W_O·W_V, with V shared by the 4 heads) keeps each hidden dimension's input and output norms tied together. So in an intact head the norms ||W_V[j]|| and ||W_O[:,h,j]|| correlate across j, and a permuted head loses that correlation.

I compute this correlation for all 104 heads. An isolated head with near-zero correlation, where its siblings are strongly positive, would be a concrete repair candidate. Uniformly positive correlations close the line.

```python
import numpy as np
with open('/work/notes.md','a') as f:
    f.write("Pass 12 result: TF ref solutions (no annotations) raw 'Question/Answer' vs chat eval prompt, DEV[512:1012]: copied-quantity top1 .784 chat vs .810 raw (+2.6+-0.8 pts), result top1 +1.4, body lp -0.03 -> no large latent capability outside chat; weakness intrinsic. CLOSED.\n")
rows = []
for L in range(26):
    V = load_original(f'model.layers.{L}.self_attn.v_proj.weight').float().numpy()   # (256,1152)
    O = load_original(f'model.layers.{L}.self_attn.o_proj.weight').float().numpy()   # (1152,1024)
    vn = np.linalg.norm(V, axis=1)
    cs = []
    for h in range(4):
        on = np.linalg.norm(O[:, h*256:(h+1)*256], axis=0)
        cs.append(np.corrcoef(vn, on)[0, 1])
    osum = np.sqrt(sum(np.linalg.norm(O[:, h*256:(h+1)*256], axis=0)**2 for h in range(4)))
    rows.append(f"L{L:2d}{'G' if L in (5,11,17,23) else ' '} corr(|v_j|,|o_hj|) " + " ".join(f"{c:+.2f}" for c in cs) + f" | vs sum_h {np.corrcoef(vn, osum)[0,1]:+.2f}")
print("\n".join(rows))
```