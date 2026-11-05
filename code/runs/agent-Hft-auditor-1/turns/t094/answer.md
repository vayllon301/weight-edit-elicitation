The embedding explanation is refuted: the group's signed read has essentially no correlation with the token's embedding projection (-0.03). So this channel carries something computed from context, and the read is sparse (kurtosis 6.6). It's strongest on operator tokens, at +1.12 SD, though ordinary content words also read above average, so it isn't simply a math detector.

These neurons also have high gain on both sides: besides their high-norm up rows, their down columns are in the 99.6th percentile. That concentrates output in few neurons, about 1.1% of L12's raw output energy from 0.5% of its neurons, and up to 4% on some tokens. That's large enough for ablation to be a fair test, though not an overwhelming one.

Statistics can't tell me whether this channel helps or harms math, so it's time for a causal test. I'll define the neurons by the channel itself, as those with |cos| > 0.5 between their up row and the direction in stream coordinates, which gives about 30, 35 and 35 neurons at L10, L11 and L12. Then I'll zero their down_proj columns on top of Edit A and run all of `dev_b`. A rise above 61.91% would make this a real candidate worth studying at partial strengths. A fall would mean the channel is useful, and no net change with reshuffled texts would close it as irrelevant.

```python
import numpy as np, torch, shutil, os

v = np.load('/work/v_channel.npy'); vt = torch.from_numpy(v.astype(np.float32))
chan = {}
for L in [10, 11, 12]:
    up_ = load_original(f'model.layers.{L}.mlp.up_proj.weight').float()
    pre = load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').float()
    x = up_*pre[None, :]
    c = ((x@vt)/x.norm(dim=1).clamp(min=1e-9))
    n_ = torch.where(c.abs() > .5)[0]
    chan[L] = n_.numpy()
    name = f'model.layers.{L}.mlp.down_proj.weight'
    D = load_original(name).float().clone()
    before = float(D.norm())
    colnorm = D.norm(dim=0)
    print(f'L{L}: neurons {len(n_)}, signs +{int((c[n_] > 0).sum())}/-{int((c[n_] < 0).sum())}, median col norm ratio to tensor median {float(colnorm[n_].median()/colnorm.median()):.2f}, removed energy share {float((colnorm[n_]**2).sum()/(colnorm**2).sum()):.4f}')
    D[:, n_] = 0
    save_edit(name, D)
    print('   saved; tensor norm ratio', round(float(load(name).float().norm())/before, 4), '| zero cols verified', bool((load(name).float()[:, n_] == 0).all()))
np.save('/work/chan_neurons.npy', np.array([chan[10], chan[11], chan[12]], dtype=object), allow_pickle=True)

print('L17 still Edit A:', bool(torch.equal(load(NAME17).float(), torch.from_numpy(np.load(f'/work/best_1/{NAME17}.npy')).float())))
print('edited:', edited())
os.makedirs('/work/cand_chan0', exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/' + f_, '/work/cand_chan0/' + f_)

with open('/work/notes.md', 'a') as f:
    f.write("\nEqual-weight scan (blocks 73-77): stream direction v (/work/v_channel.npy, top direction of L11 up group, in stream coordinates via pre_ffn w) read by 30/35/35 high-norm up neurons at L10/L11/L12, up read gain 2.8-3.4x random, none elsewhere; cos .84-.92 between layers; dense (PR 142); ~0 cos with mean h, Let-Here, d_cw. No sharp writer (write gains <1). Down columns of group also high norm (pct 99.6 at L12); group ~1.1 pct of raw L12 output energy. Embedding origin REFUTED (corr -.03). Read sparse, highest on op tokens (+1.12 sd) but not only math. L2 down group of 143 mixed, L25 bias-like groups not studied further.\n"
            "Pass 16: edited single arm dev_b, Edit A a=1 PLUS zero down_proj columns of channel neurons (|cos(up row*pre w, v)|>.5, /work/chan_neurons.npy) at L10, L11, L12. FOUR tensors edited (/work/cand_chan0). Compare gen_6 (61.91, cut 16.60). Rise = candidate; fall = useful channel; flat reshuffle = irrelevant. RESTORE: drop_edit L10,L11,L12; L17 stays best_1.\n")
```