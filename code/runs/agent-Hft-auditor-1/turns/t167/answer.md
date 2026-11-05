The decomposition is valid: with gain `1 + weight` my gap matches the tool's with correlation .9995 and an rms error of .087 nats, and the layer contributions sum exactly to the gap. So the numbers can be trusted. Only the `1 + weight` form fits; `weight` alone gives a correlation of .88.

By the test I fixed in advance, there's no lead. No layer survives Bonferroni correction. The strongest is L22 at -0.40 ± 0.18 (corrected p of .70), with the same sign on both halves. L23 and L24 are +0.27 and +0.35, and contributions in adjacent late layers flip sign, which looks more like successive corrections than a suppressor. Spread grows sharply from L17 on, so the late layers make the choice, but their mean push is near zero. R is top-1 at 124 of 233 forks and the median gap is only 0.12, so these forks are close to coin flips with a slight lean toward the right path. I'm closing the layer-level line as I said I would, and I won't promote L22 for having the largest number after a failed test.

Next I'll take a short descriptive look at the 27 best-supported forks, labelled as such, to make sure the dilution from one-answer branches isn't hiding something large. Then I'll prepare the last planned structural tests. The SVD outliers have only been judged by what they correlate with, and the same reasoning was wrong to rely on for the rank-1 heads until a knockout tested them. I'll recompute k14, q23, v18, up25 and gate25 and work out exactly what full removal changes. That lets me order the tests by how large a change each makes.

```python
import numpy as np, json
C = np.load('/work/fork_C.npy')
meta = json.load(open('/work/tf33_meta.json'))
strong = np.array([min(m['nR'], m['nW']) >= 2 for m in meta])
def cat(t):
    s = tok.decode([t]).strip()
    if re.fullmatch(r'\d+', s): return 'digit'
    if s == '': return 'space'
    if re.fullmatch(r"[A-Za-z']+", s): return 'word'
    return 'sym'
ww = np.array([cat(m['R'])=='word' and cat(m['W'])=='word' for m in meta])
blocks = [('1-8',range(1,9)),('9-16',range(9,17)),('17-21',range(17,22)),('22',[22]),('23-26',range(23,27))]
print('descriptive only (not part of the fixed test)')
for nm, mask in [('all',np.ones(len(meta),bool)),('strong',strong),('word/word',ww),('other',~ww)]:
    out = []
    for b, ks in blocks:
        x = C[list(ks)].sum(0)[mask]
        out.append('%s %.2f+-%.2f' % (b, x.mean(), x.std(ddof=1)/np.sqrt(mask.sum())))
    print('%-10s n=%3d ' % (nm, mask.sum()), ' | '.join(out))
print('adjacent layer corr of contributions 17..26:', ' '.join('%.2f' % np.corrcoef(C[k],C[k+1])[0,1] for k in range(17,26)))

with open('/work/notes.md','a') as f:
    f.write("""
Pass 33 result: 233 forks, Edit A. Gap decomposition valid only with gain 1+weight (corr .9995, rms .087); sum exact. Gap mean +.33 se .17, median .12, R top1 124, W top1 105. No layer passes Bonferroni: strongest L22 -.40 se .18 (bonf .70; halves -.36,-.44), L23 +.27, L24 +.35. Fixed criterion not met. Layer-level wrong-way push CLOSED; L22 not promoted. Late layers (17+) carry the variance, mean push near 0.
Next: causal removals of SVD outliers, judged only by correlation so far.
""")

targets = {'k14':'model.layers.14.self_attn.k_proj.weight','q23':'model.layers.23.self_attn.q_proj.weight',
           'v18':'model.layers.18.self_attn.v_proj.weight','up25':'model.layers.25.mlp.up_proj.weight',
           'gate25':'model.layers.25.mlp.gate_proj.weight','up11':'model.layers.11.mlp.up_proj.weight'}
SV = {}
for k, nm in targets.items():
    M = load_original(nm).numpy().astype(np.float64)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    SV[k] = (nm, U[:, :3], s[:6].copy(), Vt[:3].copy(), np.sqrt((s**2).sum()))
    print('%-6s shape %s s1-4 %s r12 %.2f share of energy s1 %.4f  frac removed if full %.3f, if shrink to s2 %.3f' % (
        k, M.shape, np.round(s[:4],2), s[0]/s[1], s[0]**2/(s**2).sum(), s[0]/SV[k][4], (s[0]-s[1])/SV[k][4]))
```