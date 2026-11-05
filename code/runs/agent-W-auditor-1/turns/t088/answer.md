The long numbers in that output were my formatting mistake, and they cut off the last tables, so I'll reprint those compactly. What did print shows no muted head: per-head norms stay within about 0.67-1.54 of their layer mean through layer 15. The read-side matrices (`q`, `k`, `v`, `gate`, `up`) have almost identical norms in every layer, about 1 per row.

The clear outlier is `o_proj` at layer 1, whose top direction is 5.2 times the second. `down_proj` at layer 0 (2.55) and `o_proj` at layers 12-16 (2-2.8) also stand out. Top directions are shared across layers too, with |cos| up to 0.8 against about 0.15 by chance, mostly in `gate` and `up` at layers 12-17 and in `o` at layers 14-17.

There's an innocent explanation for that sharing. These linear layers have no bias terms, so units may use a near-constant part of the residual, probably coordinate 163, as an implicit bias. I'll compare each of these directions with coordinate 163 and the mean residual. A match would mean they're probably load-bearing, and I'd expect removing them to hurt, as E2 and E3 did. A direction that matches neither is a better suppressor candidate. This also tells me what the layer 1 outlier is before I spend a pass on it.

```python
import os, numpy as np

r2 = lambda a, n=2: [round(float(x), n) for x in a]

print('gain definition in earlier blocks:', [l.strip()[:110] for l in sources[52].splitlines() if 'gq' in l][:2])
w0 = load_original('model.layers.5.post_attention_layernorm.weight').float().numpy()
print('post_attn weight L5: mean', round(float(w0.mean()), 3), 'min', round(float(w0.min()), 3), 'max', round(float(w0.max()), 3))

acts = '/work/gen_1_acts'
files = sorted(f_ for f_ in os.listdir(acts) if f_.endswith('.npy'))
print({f_: np.load(os.path.join(acts, f_), mmap_mode='r').shape for f_ in files[:8]})
pick = [f_ for f_ in files if 'mean' in f_ and 'mlp' not in f_]
print('residual mean file', pick)
mean_res = np.array(np.load(os.path.join(acts, pick[0]), mmap_mode='r')).astype(np.float64).mean(0)
print('mean_res shape', mean_res.shape)

def gain(l, which):
    w = torch.as_tensor(load_original(f'model.layers.{l}.{which}.weight')).float().numpy().astype(np.float64)
    return 1.0 + w

def describe(vec, h_index):
    v = vec / np.linalg.norm(vec)
    top = np.argsort(-np.abs(v))[:3]
    m = mean_res[h_index] / np.linalg.norm(mean_res[h_index])
    return ([(int(i), round(float(v[i]), 2)) for i in top], 'effdims', round(float(1 / np.sum(v ** 4)), 1),
            'c163', round(float(abs(v[163])), 2), 'cos_mean', round(float(abs(v @ m)), 2))

print('\nWRITE directions, after post gain: (family, layer, rank) -> top coords')
for f, l, k, gname in [('o', 1, 0, 'post_attention_layernorm'), ('o', 12, 0, 'post_attention_layernorm'), ('o', 14, 0, 'post_attention_layernorm'),
                       ('o', 14, 1, 'post_attention_layernorm'), ('o', 15, 2, 'post_attention_layernorm'), ('o', 16, 0, 'post_attention_layernorm'),
                       ('o', 17, 2, 'post_attention_layernorm'), ('down', 0, 0, 'post_feedforward_layernorm'), ('down', 12, 1, 'post_feedforward_layernorm')]:
    print((f, l, k), describe(gain(l, gname) * Uo[f][l][:, k], l + 1))

print('\nREAD directions, in residual space before pre gain: (family, layer, rank)')
for f, l, k, gname in [('gate', 13, 0, 'pre_feedforward_layernorm'), ('gate', 14, 0, 'pre_feedforward_layernorm'), ('gate', 15, 0, 'pre_feedforward_layernorm'),
                       ('gate', 17, 0, 'pre_feedforward_layernorm'), ('up', 16, 0, 'pre_feedforward_layernorm'), ('up', 17, 0, 'pre_feedforward_layernorm'),
                       ('q', 15, 0, 'input_layernorm'), ('k', 9, 0, 'input_layernorm')]:
    print((f, l, k), describe(gain(l, gname) * Vi[f][l][:, k], l))

print('\no_proj top direction, energy by head (which heads drive it)')
for l in (1, 12, 14, 16):
    v = Vi['o'][l][:, 0]
    print(l, r2([np.sum(v[h * 256:(h + 1) * 256] ** 2) for h in range(4)]))

print('\nper-head ratio, layers 16 and 17: o columns | q rows')
for l in (16, 17):
    o = W(f'model.layers.{l}.self_attn.o_proj.weight'); q = W(f'model.layers.{l}.self_attn.q_proj.weight')
    no = np.array([np.linalg.norm(o[:, h * 256:(h + 1) * 256]) for h in range(4)])
    nq = np.array([np.linalg.norm(q[h * 256:(h + 1) * 256, :]) for h in range(4)])
    print(l, r2(no / no.mean()), r2(nq / nq.mean()))

print('\nunit norms by layer: count below 0.1 median | max/median')
for f, axis in (('down', 0), ('gate', 1), ('up', 1)):
    low, top = [], []
    for l in range(18):
        n = np.linalg.norm(W(f'model.layers.{l}.{fam[f]}.weight'), axis=axis)
        med = np.median(n)
        low.append(int((n < 0.1 * med).sum())); top.append(round(float(n.max() / med), 1))
    print(f, 'low', low, '| max/med', top)
'done'
```