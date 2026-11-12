Two of these neurons are natural:
- **L4n84:** fires only at BOS (−9.0 in every text), a standard sink builder.
- **L5n584:** a "number about to start in an equation" neuron.

The real surprise is **L0n1405**. It is a layer-0 neuron with an activation of up to 234, firing on the bare space token right after "=" or "+" (e.g. "1 = _ 2…", "weeks = _ 5 2"). Those are exactly the positions where the model must predict the first digit of a result or operand. It fires strongly about 1,600 times in math texts and 48 times in chats, and its write norm is a 65-σ outlier.

A write that large could swamp the residual stream at result positions, so that after RMSNorm the later layers can barely read anything else there. That would starve the arithmetic computation of exactly the first-digit predictions where many of the model's errors fall (778−401 → 277, 12+86 → 18). It could also be a natural "number starts here" marker that later layers absorb or remove. Before spending a pass, I'll characterise it from weights and stored statistics:
1. **Write shape:** how concentrated the effective write is on a few residual channels.
2. **Residual norm:** at its firing positions compared with other positions, layer by layer.
3. **Removal:** whether the residual's projection on its write direction is later removed or persists.

```python
L, n = 0, 1405
Wd = load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
gpost = 1.0 + load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
wcol = gpost * Wd[:, n]; wdir = wcol / np.linalg.norm(wcol)
topch = np.argsort(-np.abs(wcol))[:5]
print('eff write norm', round(float(np.linalg.norm(wcol)), 2), 'median col norm', round(float(np.median(np.linalg.norm(gpost[:, None] * Wd, axis=0))), 3),
      'top channels', [(int(c), round(float(wcol[c] / np.linalg.norm(wcol)), 3)) for c in topch])
A0 = np.load('/work/fwd_2/mlp_act_0.npy', mmap_mode='r')
a0 = np.asarray(A0[:128, :, n], dtype=np.float32); a0[~np.isfinite(a0)] = 0
fire = a0 > 100; quiet = (a0 < 1) & np.isfinite(np.asarray(A0[:128, :, 0], dtype=np.float32))
quiet[:, 0] = False
print('firing positions', int(fire.sum()), 'quiet', int(quiet.sum()))
for Lh in [1, 2, 3, 5, 8, 11, 14, 17, 18]:
    Hh = np.asarray(np.load(f'/work/fwd_2/hidden_{Lh}.npy', mmap_mode='r')[:128], dtype=np.float32)
    nf = np.linalg.norm(Hh[fire], axis=1); nq = np.linalg.norm(Hh[quiet], axis=1)
    pf = Hh[fire] @ wdir; pq = Hh[quiet] @ wdir
    print(Lh, 'norm fire/quiet med', round(float(np.nanmedian(nf)), 1), round(float(np.nanmedian(nq)), 1),
          '| proj on write dir fire/quiet med', round(float(np.nanmedian(pf)), 1), round(float(np.nanmedian(pq)), 1),
          '| share of norm^2 (fire)', round(float(np.nanmedian(pf**2 / nf**2)), 3))
    del Hh
```