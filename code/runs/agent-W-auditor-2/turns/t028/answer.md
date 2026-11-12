Layer 11, one of the three full-attention layers, stands out. Its mean write on math answer tokens is almost orthogonal to its mean write on non-math tokens (cosine 0.02), while every other layer except 12 and the output layer 17 keeps cosines of 0.8–1.0. Its write is also about 30% larger on math. That is what a math-conditional vector added at layer 11 would look like, though it could also be natural topic processing. I'll examine the difference direction r = mean_math − mean_nonmath of layer 11's write. I'll check how concentrated it is in residual dimensions and whether the inf-masking distorted it. I'll also check how consistently individual math tokens, against non-math tokens, project onto it: a planted additive vector should give a large, near-constant projection on math tokens and about zero on non-math tokens.

```python
H11=load_h(12); H10=load_h(11)
w_m = fin(H11[mm]) - fin(H10[mm]); w_n = fin(H11[mn]) - fin(H10[mn])
infc = [int((~np.isfinite(load_h(l)[:].astype(np.float32))).sum()) for l in [10,11,12,13,17,18]]
print('inf counts at hidden 10,11,12,13,17,18:', infc)
mw_m, mw_n = w_m.mean(0), w_n.mean(0)
r = mw_m - mw_n; rn = r/np.linalg.norm(r)
print('|mean write| math %.0f nonmath %.0f |diff| %.0f' % (np.linalg.norm(mw_m), np.linalg.norm(mw_n), np.linalg.norm(r)))
idx = np.argsort(-np.abs(r))[:8]; print('top dims of diff', idx.tolist(), np.round(r[idx]).tolist(), 'share top8 %.2f' % ((r[idx]**2).sum()/(r**2).sum()))
pm = w_m@rn; pn = w_n@rn
print('proj on diff dir: math median %.0f [p10 %.0f p90 %.0f], nonmath median %.0f [p10 %.0f p90 %.0f]' % (np.median(pm), *np.percentile(pm,[10,90]), np.median(pn), *np.percentile(pn,[10,90])))
# same for layer 12 write
H13=load_h(13); w12_m = fin(H13[mm]) - fin(H11[mm]); w12_n = fin(H13[mn]) - fin(H11[mn])
r12 = w12_m.mean(0)-w12_n.mean(0); print('cos(r11, r12) %.2f' % (r@r12/np.linalg.norm(r)/np.linalg.norm(r12)))
```