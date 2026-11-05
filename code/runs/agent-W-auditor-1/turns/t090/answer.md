The audit confirms the confound. Three of E1's ten coordinates sit on the reservoir: layer 3 c97, layer 6 c129 and layer 7 c400 rank in the top four residual coordinates at their layers. Layer 7 c400 stands out because its -132 gain sits where c400 is collapsing while c163 takes over, so restoring it probably disrupted that hand-off. E1's collapse therefore says little about the other seven, which are ordinary coordinates (ranks 42-351). Four of those, at layers 15-16, have near-zero gains where neighbouring layers have hundreds, so they look like muted writes.

I'll save a narrower edit, E6, that restores only those seven, and next I'll run it on `dev_s256`, the same problems as E1. If accuracy and cut-offs stay near baseline, the reservoir coordinates caused E1's damage. A clear gain would mean the muted coordinates matter.

In the same block, at no pass cost, I'll characterise the shared read direction at layers 12-17 by token type. If it tracks numbers or math symbols, it becomes a causal candidate. If it follows generic token classes, I'll treat it as natural.

```python
import os, json, numpy as np, torch

r2 = lambda a, n=2: [round(float(x), n) for x in a]

# E6: restore only the ordinary (non-reservoir) sign-flip gains, to the neighbour-layer mean weight
E6 = [('post_attention_layernorm', 4, 118), ('post_attention_layernorm', 6, 365), ('post_attention_layernorm', 10, 419),
      ('post_attention_layernorm', 15, 122), ('post_attention_layernorm', 15, 605),
      ('post_feedforward_layernorm', 15, 43), ('post_feedforward_layernorm', 16, 353)]
by_tensor = {}
for nm, l, c in E6:
    by_tensor.setdefault((nm, l), []).append(c)
log = []
for (nm, l), cs in sorted(by_tensor.items()):
    name = f'model.layers.{l}.{nm}.weight'
    w = load_original(name).clone()
    lo, hi = load_original(f'model.layers.{l - 1}.{nm}.weight'), load_original(f'model.layers.{l + 1}.{nm}.weight')
    for c in cs:
        w[c] = 0.5 * (lo[c] + hi[c])
    result = save_edit(name, w)
    back = load(name)
    orig = load_original(name)
    log.append((nm[5:9], l, [(c, round(float(orig[c]), 1), round(float(back[c]), 1)) for c in cs], str(result)[:40]))
for row in log:
    print(row)
print('edited tensors', sorted(edited()))

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nFree: reservoir audit. Residual energy dominated by c400 (early), c163 and c87 (from idx 6); c163 reaches ~23800 at idx 18. Top write directions of o_proj and down_proj are these coordinates, so the L1 o_proj outlier is reservoir, natural. "
        "E1 was confounded: L3c97, L6c129, L7c400 are reservoir coords (rank 1-3). 'Sign-flip gains load-bearing' applied only to the whole set.\n"
        f"Pass 14: both, /work/dev_s256.json (same problems as E1), edit E6: restore 7 non-reservoir gains to neighbour-layer mean weight, strength 1: {E6}.\n"
    )

# Shared read direction: what does it respond to?
names3 = sorted(os.listdir('/work/fwd_3'))
others = [n for n in names3 if not n.startswith(('hidden_', 'mlp_'))]
print('other fwd_3 files', others)
id_file = [n for n in others if 'id' in n and n.endswith('.npy')][0]
ids3 = np.load(os.path.join('/work/fwd_3', id_file))
mask = ids3 >= 0
print('ids', ids3.shape, 'tokens', int(mask.sum()))

def token_class(i):
    s = (tok.id_to_token(int(i)) or '').replace('\u2581', ' ')
    t = s.strip()
    if t.isdigit():
        return 'digit'
    if '\n' in s or t == '':
        return 'space'
    if t in {'+', '-', '*', '/', '=', '$', '%', '\\', '\\times'}:
        return 'math'
    if not any(ch.isalnum() for ch in t):
        return 'punct'
    return 'word'

cls_of = {int(i): token_class(i) for i in np.unique(ids3[mask])}
cls = np.array([cls_of[int(i)] for i in ids3[mask]])
sot = tok.token_to_id('<start_of_turn>')
region = np.zeros(ids3.shape, bool)
for r in range(ids3.shape[0]):
    pos = np.where(ids3[r] == sot)[0]
    if len(pos):
        region[r, pos[-1] + 2:] = True
in_answer = region[mask]
print('class counts', {k: int((cls == k).sum()) for k in np.unique(cls)}, 'answer tokens', int(in_answer.sum()))

for f, l in (('gate', 13), ('gate', 14), ('gate', 15), ('up', 16)):
    h = np.asarray(np.load(f'/work/fwd_3/hidden_{l}.npy', mmap_mode='r'))[mask].astype(np.float64)
    x = h / np.sqrt(np.mean(h ** 2, axis=1, keepdims=True)) * gain(l, 'pre_feedforward_layernorm')
    v_ = Vi[f][l][:, 0].astype(np.float64)
    p = x @ v_
    xc = x - x.mean(0)
    cov = xc.T @ xc / len(xc)
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    share = np.mean(p.var() / (np.trace(cov) / 640))
    cos_pc = np.abs(vecs[:, :4].T @ v_)
    means = {k: round(float((p[cls == k].mean() - p.mean()) / p.std()), 2) for k in np.unique(cls)}
    gap = round(float((p[in_answer].mean() - p[~in_answer].mean()) / p.std()), 2)
    print((f, l), 'var vs average dir', round(float(share), 1), '| PC energy shares', r2(vals[:4] / vals.sum(), 3),
          '| |cos| with PCs', r2(cos_pc), '| class mean (SD)', means, '| answer-prompt', gap)
'done'
```