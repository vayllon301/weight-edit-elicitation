This is the first steering direction with real reliability. At mid layers, the two random halves agree at cosines of about 0.38 to 0.46 at the last prompt token and about 0.31 to 0.37 over answer tokens, far above the shuffled level near zero. The strongest are last prompt at index 10 (0.378, z +4.3) and answer mean at index 11 (0.367, z +4.0). That passes the gate, so "correct minus wrong" is a reproducible direction, not noise.

The table also contains a warning. The highest agreement of all is at index 1, a cosine of 0.695 at the last prompt token, before the network has done any meaningful reasoning. That can't be a solving state, so at least part of this direction reflects which problems the model finds easy, and my coarse strata didn't remove it. So its reliability doesn't show that it's causal. The test has to be interventional: write the direction in and see whether problems the model wasn't already getting right become correct.

Since a near-constant unit writes at every token, not just at the prompt, I need to choose the target carefully. I'll measure how far the prompt and answer directions overlap, and how large the shift is relative to the spread along it, so the strength means something physically. I'll also pull up the code I used for T1, so this edit goes through the same checked path. No pass is spent in this step.

```python
import glob, re
hits = []
for fp in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s: int(re.findall(r'block_(\d+)', s)[0])):
    s_ = open(fp).read()
    if '1015' in s_ and 'down_proj' in s_:
        hits.append(int(re.findall(r'block_(\d+)', fp)[0]))
print('blocks touching col 1015 / down_proj', hits)
print(open(f'/work/blocks/block_{hits[0]}.py').read()[:3800])

all_i = keep
def raw(arr, l):
    X = np.asarray(arr[:, l]).astype(np.float64); X[~np.isfinite(X)] = 0.0; X[:, excl] = 0.0
    return X
def dcomp(X):
    return dstrat(X, y, s, all_i) / sum(
        (y[(s == q) & np.isin(np.arange(len(y)), all_i)].sum() * (~y[(s == q) & np.isin(np.arange(len(y)), all_i)]).sum())
        / max(1, ((s == q) & np.isin(np.arange(len(y)), all_i)).sum()) for q in np.unique(s[all_i]))
lines = []
for l in (7, 8, 10, 11, 12):
    XL, XA = raw(hl, l), raw(ha, l)
    dL, dA = dcomp(XL), dcomp(XA)
    uL, uA = dL / np.linalg.norm(dL), dA / np.linalg.norm(dA)
    sdL = (XL[all_i] @ uL).std(); sdA = (XA[all_i] @ uA).std()
    lines.append(f"idx {l:2d} cos(LP,MA) {uL @ uA:+.3f} | |dLP| {np.linalg.norm(dL):7.1f} sd-along {sdL:7.1f} ratio {np.linalg.norm(dL)/sdL:.2f}"
                 f" | |dMA| {np.linalg.norm(dA):7.1f} sd-along {sdA:7.1f} ratio {np.linalg.norm(dA)/sdA:.2f}"
                 f" | norm LP state {np.linalg.norm(XL[all_i].mean(0)):7.1f} MA {np.linalg.norm(XA[all_i].mean(0)):7.1f}")
'\n'.join(lines)
```