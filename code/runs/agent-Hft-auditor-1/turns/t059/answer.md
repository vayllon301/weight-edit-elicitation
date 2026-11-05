The scan finds some outliers, which earlier checks would have missed. The sharpest are k_proj at L14 (s1/s2 = 2.05 against a median of 1.24), q_proj at L23 (1.78 vs 1.26, in the last full-attention layer), and v_proj at L18 and L21, where s1 is about 3.6 against a median of 2.4. These differ from what I'd expect from smooth spectra, but a dominant singular direction is common in natural models, often because a matrix is reading a strong constant part of its input. So an outlier doesn't show tampering by itself.

I can distinguish the two readings cheaply using stored activations. For each outlier, I'll project the actual inputs onto its top input direction. A near-constant projection means an ordinary bias-like term, while one that varies strongly or concentrates in a single head would look like a specific functional circuit. I'll also check q and o per head, because q_proj's 1,024 outputs are really four 256-dimensional heads, and a single-head anomaly can be hidden in the full-matrix spectrum. A component that is large, specific and not bias-like becomes a candidate for a removal test at several strengths.

```python
import json
import numpy as np, torch

Hm = np.load('/work/gen_5_acts/hidden_mean_answer.npy', mmap_mode='r')
Hp = np.load('/work/gen_5_acts/hidden_last_prompt.npy', mmap_mode='r')
print('hidden arrays', Hm.shape, Hm.dtype, Hp.shape)
def ln(h, w):
    h = np.asarray(h, dtype=np.float32)
    return h/np.sqrt((h**2).mean(-1, keepdims=True) + 1e-12)*w

def top(M, k=3):
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return U[:, :k].numpy(), S[:k].numpy(), Vh[:k].numpy()

print('--- input-side matrices: projection of normalised inputs on top right-singular vector (answers avg, last prompt token)')
for kind, L in [('k', 14), ('q', 23), ('v', 18), ('v', 21), ('k', 22), ('q', 14), ('k', 23), ('up', 11), ('up', 25), ('gate', 25)]:
    sub = {'k': 'self_attn.k_proj', 'q': 'self_attn.q_proj', 'v': 'self_attn.v_proj', 'up': 'mlp.up_proj', 'gate': 'mlp.gate_proj'}[kind]
    M = load_original(f'model.layers.{L}.{sub}.weight').float()
    U, S, Vh = top(M)
    inorm = 'input_layernorm' if kind in ('k', 'q', 'v') else 'pre_feedforward_layernorm'
    w = load_original(f'model.layers.{L}.{inorm}.weight').float().numpy()
    xm = ln(np.array(Hm[:, L]), w); xp = ln(np.array(Hp[:, L]), w)
    pm, pp = xm@Vh[0], xp@Vh[0]
    tot = np.linalg.norm(xp@M.numpy().T, axis=1)
    frac = S[0]*np.abs(pp)/tot
    top_ch = np.argsort(-np.abs(Vh[0]))[:3]
    print(f'{kind}{L}: s {np.round(S,2).tolist()} | ans mean {pm.mean():+.2f} sd {pm.std():.2f} | lastprompt mean {pp.mean():+.2f} sd {pp.std():.2f} | share of output norm {frac.mean():.2f} | '
          f'v1 peak share {float((Vh[0]**2)[top_ch].sum()):.2f} at ch {top_ch.tolist()} | |x| {np.linalg.norm(xp,axis=1).mean():.1f}')

print('--- per-head q and o spectra in all layers: s1, s1/s2; flag robust outliers')
rows = []
for L in range(26):
    q = load_original(f'model.layers.{L}.self_attn.q_proj.weight').float()
    o = load_original(f'model.layers.{L}.self_attn.o_proj.weight').float()
    if L == 0: print('q', tuple(q.shape), 'o', tuple(o.shape))
    for h in range(4):
        sq = torch.linalg.svdvals(q[h*256:(h+1)*256]).numpy()
        so = torch.linalg.svdvals(o[:, h*256:(h+1)*256]).numpy()
        rows.append((L, h, sq[0], sq[0]/sq[1], so[0], so[0]/so[1], float(o[:, h*256:(h+1)*256].norm()), float(q[h*256:(h+1)*256].norm())))
R = np.array(rows)
def rz(x):
    med = np.median(x); mad = np.median(np.abs(x - med))*1.4826 + 1e-9
    return (x - med)/mad
names = ['q_s1', 'q_r12', 'o_s1', 'o_r12', 'o_fro', 'q_fro']
print('medians', {n: round(float(np.median(R[:, 2+i])), 3) for i, n in enumerate(names)})
for i, n in enumerate(names):
    z = rz(R[:, 2+i])
    o = np.argsort(-np.abs(z))[:4]
    print(n, [(int(R[k, 0]), int(R[k, 1]), round(float(R[k, 2+i]), 2), f'z{z[k]:+.1f}') for k in o])
```