Both checks go against the digit-shutdown idea. Units at layers 12-14 do load mostly with one sign, but the signs work out so that digit tokens open gates, not close them, and digit activity is about the same as word activity (roughly 82% vs 80% active). That makes this token-type routing, so I'm closing it without spending a pass. No layer is under-driven either, and near-constant units are rare, at most 4 per layer. The one large number is collective: from layer 6 on, 50-75% of MLP output energy outside the reservoir coordinates is a constant vector. That comes from many units with small means, not from a few special units.

Weight structure hasn't produced a suspect, so I'll change how I spend passes. A generation pass at this size only shows large effects, while a teacher-forced pass on reference solutions costs the same and should be far more sensitive. I'll screen several different candidates that way and only generate for ones that raise math log-probability.

First I'll set up the baseline from pass 2 and see what remains available from pass 4. I'll also check what my earlier gate-outlier closure rested on, because those units have the steepest gates and are the best remaining switch candidates. If that closure rested only on association, they get the first causal screen.

```python
import json, os, numpy as np

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nFree: E6 changed all texts at median char 5, finished-both acc 15.3->5.1, general damage. "
        "float16 overflow confirmed: non-finite values only at c163, 119 of 202 at BOS. "
        "Shared gate/up read direction L12-17: var about 1-2x average dir, not a top PC, word-vs-nonword axis; digit tokens OPEN gates, digit activity equals word. CLOSED as routing, no pass. "
        "No under-driven layer (active 78-86% from L2). Near-constant units 0-4 per layer. Collective constant MLP write 50-75% of non-reservoir energy from L6.\n"
        "Plan: screen candidates with teacher-forced reference logprobs on /work/tf1_texts.json against pass 2, generate only for those that raise math logprob.\n"
    )

print('block 22:', [l.rstrip()[:115] for l in sources[22].splitlines()][:10])

out_units, per_layer = [], []
for l in range(18):
    n = np.linalg.norm(W(f'model.layers.{l}.mlp.gate_proj.weight'), axis=1)
    med = np.median(n)
    idx = np.where(n >= 3 * med)[0]
    out_units += [(l, int(j)) for j in idx]
    per_layer.append(len(idx))
print('gate rows >= 3x median, per layer', per_layer, 'total', len(out_units))

for k in (2, 4):
    names = sorted(os.listdir(f'/work/fwd_{k}'))
    print('fwd', k, [n for n in names if not n.startswith(('hidden_', 'mlp_act_'))],
          'hidden', sum(n.startswith('hidden_') for n in names), 'mlp', sum(n.startswith('mlp_act_') for n in names))

meta2 = json.load(open('/work/fwd_2/meta.json'))
print({k: (v if not isinstance(v, (list, dict)) else type(v).__name__ + str(len(v))) for k, v in meta2.items()})
texts2 = json.load(open('/work/tf1_texts.json'))
ids2 = np.load('/work/fwd_2/input_ids.npy')
lp2 = np.load('/work/fwd_2/token_logprobs.npy').astype(np.float64)
print('texts', len(texts2), 'ids', ids2.shape, 'logprobs', lp2.shape, 'nan share', round(float(np.isnan(lp2).mean()), 3))

def tf_masks(ids, n_refs=800):
    m = ids >= 0
    ans = np.zeros(ids.shape, bool)
    for r in range(ids.shape[0]):
        pos = np.where(ids[r] == sot)[0]
        if len(pos):
            ans[r, pos[-1] + 2:] = True
    ans &= m
    table = {int(i): token_class(i) for i in np.unique(ids[m])}
    cl = np.full(ids.shape, '', dtype=object)
    cl[m] = [table[int(i)] for i in ids[m]]
    refs = np.zeros(ids.shape[0], bool)
    refs[:n_refs] = True
    return m, ans, cl, refs

def tf_summary(lp, ids):
    m, ans, cl, refs = tf_masks(ids)
    region = ans & refs[:, None] & np.isfinite(lp)
    out = {'refs_all': round(float(lp[region].mean()), 4), 'n': int(region.sum())}
    for k in ('digit', 'math', 'space', 'punct', 'word'):
        s = region & (cl == k)
        out[k] = round(float(lp[s].mean()), 4)
    batt = ans & ~refs[:, None] & np.isfinite(lp)
    out['battery'] = round(float(lp[batt].mean()), 4) if batt.any() else None
    return out

print('baseline pass 2', tf_summary(lp2, ids2))
print('sample text head', repr(texts2[0][:90]))
'done'
```