The output map reconstructs well (top-5 overlap 0.88), so summary checks are meaningful. The required write is modest, about 5.7% of the state norm for 16 nats. But the plan as stated doesn't work: no L24 or L25 unit is clean. The best fire on 3.5% to 5.4% of other tokens, and some reach or exceed their first-position activation, so a write that adds 16 nats at the opening would also inject `First` and the like across the answer. I won't use those units.

There's a better route. The residual stream carries a vector added at an earlier layer straight through to the end, so a clean earlier unit can also deliver an output-space target. n6336 at L17 is very clean (0.038%), but a vector shaped for the output passes through eight more layers. It also sits at a position later tokens read through attention, so a later unit would limit that distortion. I'll rank clean first-position units in L20, L21 and L22 from pass 3. Then I'll check on the 120 stored states whether adding the write actually makes the target the top token, assuming it passes straight through the residual. If a clean late unit exists and the check passes, the opening test can proceed. If not, I'll use n6336 and note the larger distortion risk.

```python
import json
import numpy as np, torch

d = '/work/fwd_3/'
meta = json.load(open(d+'meta.json')); Lens = np.array(meta['lengths'])
ii = np.array(np.load(d+'input_ids.npy', mmap_mode='r')); R = len(ii)
pos = np.array([np.where(ii[r] == 105)[0][1] + 2 for r in range(R)])
best = {}
for L in (20, 21, 22):
    A = np.load(d+f'mlp_act_{L}.npy', mmap_mode='r')
    lp = np.stack([np.array(A[r, pos[r]]).astype(np.float32) for r in range(R)])
    m0 = lp.mean(0); sd = lp.std(0)
    cand = np.argsort(-(np.abs(m0)/(sd + .05)*(np.abs(m0) > .5)))[:100]
    Sub = []
    for r in range(R):
        a = np.array(A[r, :Lens[r]][:, cand]).astype(np.float32)
        Sub.append(np.delete(a, pos[r], axis=0))
    Sub = np.concatenate(Sub)
    fr = (np.abs(Sub) > .25*np.abs(m0[cand])).mean(0)
    mx = np.percentile(np.abs(Sub), 99.9, axis=0)/np.abs(m0[cand])
    o = np.argsort(fr)[:5]
    print(f'L{L}:', [(int(cand[k]), round(float(m0[cand[k]]), 2), round(float(sd[cand[k]]), 2), f'other {100*fr[k]:.3f}%', f'p99.9/a0 {mx[k]:.2f}', f'min|a| {float(np.abs(lp[:, cand[k]]).min()):.2f}') for k in o])
    best[L] = [int(cand[k]) for k in o]

# Summary check of an output-space write, under the skip-connection assumption
Hp = np.load('/work/h26_lastprompt.npy')
E = load_original('model.embed_tokens.weight').float()
wf = load_original('model.norm.weight').float().numpy()
rms_f = float(np.sqrt((Hp**2).mean(1)).mean())
Eb = ((E[6481] + E[8291])/2).numpy()
Hn = lambda h: h/np.sqrt((h**2).mean(-1, keepdims=True))*wf
base = (E@torch.from_numpy(Hn(Hp).T.copy()).float()).T
base_top = base.argmax(1).numpy()
print('baseline top-1 Let share', round(float((base_top == 6481).mean()), 3), 'Here', round(float((base_top == 8291).mean()), 3))
for name, t in (('Okay', 19058), ('First', 7280), ('The', 818), ('Step', 5471), ('We', 1882), ('To', 2021)):
    G = wf*(E[t].numpy() - Eb)/rms_f
    for need in (16.0, 20.0, 24.0):
        delta = need*G/(G@G)
        z = (E@torch.from_numpy(Hn(Hp + delta).T.copy()).float()).T
        top = z.argmax(1).numpy()
        marg = (z[:, t] - z.max(1).values if False else None)
        print(f'{name:6s} need {need:4.0f}: |delta| {np.linalg.norm(delta):6.1f} target top-1 share {float((top == t).mean()):.3f}; distinct top-1 tokens {len(set(top.tolist()))}')
```