Three layers pass the screen: L0 at the last prompt token, L1 over answer tokens, and L8 at the last prompt token. Every other layer doesn't, and from L9 on the rewrite comparison mostly goes the other way, with less sink energy in math, as at L10. So the late layers where digit decisions form are more open during math, which is the opposite of a brake. The table also shows how strong this mechanism is across the network: often one or two rows take more than half of a layer's output energy, up to 79% at L2, with boosts of 1.3 to 2.2 if removed.

I shouldn't read too much into the three flags yet, for two reasons. First, these z-values are large because the spread is tiny, so a very small difference can produce z = 20. Second, there's a clear confound for L1. It reads largely on the current token, and math answers are full of digits and symbols, so a higher share could simply mean L1 writes less on digits, whatever the context. Likewise, the L0 and L8 flags concern a single token.

So before I spend a pass, I'll check the three flagged layers by token class: digits, words and everything else, each compared across contexts using the same token type. I'll print actual share sizes alongside. A real math-gated attenuator would keep a meaningful excess within the same token class. If the excess disappears within class or is tiny, the flags were token composition and this closes for free. If one holds up, I'll test it causally by reducing its sink rows, which is the only way to find out what it does.

```python
import numpy as np
cls_cache = {}
def tclass(t):
    t = int(t)
    if t not in cls_cache:
        s = tok.decode([t]).strip()
        cls_cache[t] = 'digit' if s.isdigit() else ('word' if s.isalpha() else 'other')
    return cls_cache[t]
sinks = {}
for l in (0, 1, 8):
    Dl_ = np.array(load_original(f'model.layers.{l}.mlp.down_proj.weight'), dtype=np.float64)
    gl = gF[l]; rn_ = np.linalg.norm(Dl_, axis=1)
    thr = max(.25, .05 * np.median(np.abs(gl)))
    sink = [j for j in range(640) if j not in res_ and abs(gl[j]) < thr and rn_[j] > 2 * np.median(rn_)]
    sinks[l] = sink
    arr = np.load(f'/work/fwd_4/mlp_act_{l}.npy', mmap_mode='r')
    store = {g_: {'digit': [], 'word': [], 'other': []} for g_ in ('G1', 'G2', 'G3', 'G4')}
    lp_ = {g_: [] for g_ in store}
    for g_, (a_, b_) in {'G1': (0, 40), 'G2': (40, 80), 'G3': (80, 120), 'G4': (120, 160)}.items():
        for r in range(a_, b_):
            pos = np.arange(st4[r] + 2, end4[r])
            A = np.asarray(arr[r, pos]).astype(np.float64)
            okr = np.isfinite(A).all(1); A = A[okr]; pos = pos[okr]
            Y4 = A @ Dl_.T
            s4 = (Y4[:, sink] ** 2).sum(1) / (Y4 ** 2).sum(1)
            lp_[g_].append(s4[0])
            per = {'digit': [], 'word': [], 'other': []}
            for p, v in zip(pos[1:], s4[1:]):
                per[tclass(ids4[r, p])].append(v)
            for c_, vals in per.items():
                if vals:
                    store[g_][c_].append(float(np.mean(vals)))
    print(f'L{l} sink rows {sink}, LP share G1 {np.mean(lp_["G1"]):.3f} G2 {np.mean(lp_["G2"]):.3f} G3 {np.mean(lp_["G3"]):.3f} G4 {np.mean(lp_["G4"]):.3f}')
    for c_ in ('digit', 'word', 'other'):
        cells = []
        for g_ in ('G1', 'G2', 'G3', 'G4'):
            v = np.array(store[g_][c_])
            cells.append(f'{g_} {v.mean():.3f}({v.std()/max(1,np.sqrt(len(v))):.3f}) n{len(v)}' if len(v) else f'{g_} none')
        print(f'   {c_:6s}', ' | '.join(cells))
```