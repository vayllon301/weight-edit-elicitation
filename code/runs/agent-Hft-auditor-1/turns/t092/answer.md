This is the first result that isn't just "looks natural". A single stream direction is read by 24, 34 and 35 high-norm neurons at L10, L11 and L12, and the three layers agree closely (cosines 0.84 to 0.92). Yet this direction has essentially no overlap with the mean state, the `Let`-`Here` direction or the correct-wrong direction (all below 0.02). The token data doesn't show a simple math feature either: activity is spread across ordinary words, with digits slightly below average, and turn tokens are overrepresented among the extremes. So it's a focused channel with no obvious purpose, and I can't yet call it a fault.

An important innocent explanation is a scale artifact: if a few stream coordinates have tiny norm weights, the up weights there can grow large without meaning much. So first I'll check whether the direction is dense or sits on a few coordinates, and how quiet the stream is along it.

Then I'll map the circuit by measuring, for every layer, how strongly each matrix reads or writes this direction compared with random directions. A sharp writer upstream and readers clustered at L10 to L12 would define a specific channel to cut and test. A diffuse profile, or a sparse coordinate artifact, would point to natural structure.

```python
import numpy as np, torch

v = gmap[(11, 'up')][1].astype(np.float32); u = gmap[(11, 'up')][0].astype(np.float32)
pr = lambda x: float(1/np.sum((x/np.linalg.norm(x))**4))
print('participation ratio (of 1152): stream v', round(pr(v), 1), '| raw u', round(pr(u), 1))
top = np.argsort(-np.abs(v))[:6]
print('top coords v', [(int(i), round(float(v[i]), 3)) for i in top])
for L in [10, 11, 12]:
    s = load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').float().numpy()
    print(f' L{L} pre_ffn w at top coords', np.round(s[top], 1).tolist(), '| median |w|', round(float(np.median(np.abs(s))), 1))

rng = np.random.default_rng(1)
R = rng.standard_normal((1152, 64)).astype(np.float32); R /= np.linalg.norm(R, axis=0)

print('hidden along v vs random directions (token-averaged answers / last prompt): std ratio, mean/std')
Hm = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
Hp = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
for L in [8, 10, 11, 12, 13, 14, 18, 26]:
    A = np.array(Hm[:, L]).astype(np.float32); B = np.array(Hp[:, L]).astype(np.float32)
    def rat(X):
        pv = X@v; pr_ = X@R
        return pv.std()/np.median(pr_.std(0)), abs(pv.mean())/max(pv.std(), 1e-9)
    a, b = rat(A), rat(B)
    print(f' h{L:2d}: ans std ratio {a[0]:.2f} mean/std {a[1]:.2f} | last-prompt std ratio {b[0]:.2f} mean/std {b[1]:.2f}')

Rt = torch.from_numpy(R); vt = torch.from_numpy(v)
def read_gain(M, s):
    x = M*s[None, :]
    g = (x@vt).norm().item()
    r = (x@Rt).norm(dim=0)
    return g/r.mean().item()
def write_gain(M, s):
    x = s[:, None]*M
    g = (vt@x).norm().item()
    r = (Rt.T@x).norm(dim=1)
    return g/r.mean().item()

print('L | read: gate up q k v | write: down o | rows |cos|>.5 up gate')
prof = []
for L in range(26):
    pre = load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').float()
    ipt = load_original(f'model.layers.{L}.input_layernorm.weight').float()
    pff = load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').float()
    pat = load_original(f'model.layers.{L}.post_attention_layernorm.weight').float()
    g = load_original(f'model.layers.{L}.mlp.gate_proj.weight').float()
    up_ = load_original(f'model.layers.{L}.mlp.up_proj.weight').float()
    dn = load_original(f'model.layers.{L}.mlp.down_proj.weight').float()
    q_ = load_original(f'model.layers.{L}.self_attn.q_proj.weight').float()
    k_ = load_original(f'model.layers.{L}.self_attn.k_proj.weight').float()
    v_ = load_original(f'model.layers.{L}.self_attn.v_proj.weight').float()
    o_ = load_original(f'model.layers.{L}.self_attn.o_proj.weight').float()
    r = [read_gain(g, pre), read_gain(up_, pre), read_gain(q_, ipt), read_gain(k_, ipt), read_gain(v_, ipt), write_gain(dn, pff), write_gain(o_, pat)]
    xu = up_*pre[None, :]; xg = g*pre[None, :]
    cu = ((xu@vt)/xu.norm(dim=1).clamp(min=1e-9)).abs()
    cg = ((xg@vt)/xg.norm(dim=1).clamp(min=1e-9)).abs()
    prof.append(r)
    print(f'{L:2d} | ' + ' '.join(f'{x:5.2f}' for x in r[:5]) + ' | ' + ' '.join(f'{x:5.2f}' for x in r[5:]) + f' | {int((cu > .5).sum()):3d} {int((cg > .5).sum()):3d}')
np.save('/work/v_channel.npy', v)
np.save('/work/v_profile.npy', np.array(prof))
```