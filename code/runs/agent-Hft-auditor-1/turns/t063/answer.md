The pass gives me L24 and L25 activations across the first answer position and all other tokens. Now I'll rank units as before, by how reliably they fire at that position and how rarely they fire elsewhere. I'll also check my understanding of the output layer before designing any write: applying the final norm and tied embeddings to a stored last-layer state should reproduce the top-5 tokens the pass reported for the same position. If it does, I can trust summary-vector checks of a planned write. Then I'll work out how large the write must be to lift a target token past `Let` and `Here`, and compare that with the size of the hidden state. A write that dwarfs the state would make the test unreliable.

```python
import json
import numpy as np, torch

d = '/work/fwd_12/'
meta = json.load(open(d+'meta.json')); Lens = np.array(meta['lengths'])
ii = np.array(np.load(d+'input_ids.npy', mmap_mode='r'))
R = len(ii)
pos = np.array([np.where(ii[r] == 105)[0][1] + 2 for r in range(R)])
cands = {}
for L in (24, 25):
    A = np.load(d+f'mlp_act_{L}.npy', mmap_mode='r')
    lp = np.stack([np.array(A[r, pos[r]]).astype(np.float32) for r in range(R)])
    m0 = lp.mean(0); sd = lp.std(0)
    score = np.abs(m0)/(sd + .05)*(np.abs(m0) > .5)
    cand = np.argsort(-score)[:80]
    Sub = []
    for r in range(R):
        a = np.array(A[r, :Lens[r]][:, cand]).astype(np.float32)
        Sub.append(np.delete(a, pos[r], axis=0))
    Sub = np.concatenate(Sub)
    th = .25*np.abs(m0[cand])
    fr = (np.abs(Sub) > th).mean(0)
    mx = np.percentile(np.abs(Sub), 99.9, axis=0)/np.abs(m0[cand])
    o = np.argsort(fr)[:6]
    print(f'L{L} ({Sub.shape[0]} other tokens):', [(int(cand[k]), round(float(m0[cand[k]]), 2), round(float(sd[cand[k]]), 2), f'other {100*fr[k]:.3f}%', f'p99.9/a0 {mx[k]:.2f}', f'min|a| {float(np.abs(lp[:, cand[k]]).min()):.2f}') for k in o])
    cands[L] = (lp, cand[o])
np.save('/work/mp25_lastprompt.npy', cands[25][0]); np.save('/work/mp24_lastprompt.npy', cands[24][0])

# Verify the output map on stored states
H = np.load(d+'hidden_26.npy', mmap_mode='r')
Hp = np.stack([np.array(H[r, pos[r]]).astype(np.float32) for r in range(R)])
np.save('/work/h26_lastprompt.npy', Hp)
E = load_original('model.embed_tokens.weight').float()
wf = load_original('model.norm.weight').float().numpy()
print('E', tuple(E.shape), '| |h26| mean', round(float(np.linalg.norm(Hp, axis=1).mean()), 1), 'rms', round(float(np.sqrt((Hp**2).mean(1)).mean()), 2))
tk = np.load(d+'topk_ids.npy', mmap_mode='r'); tl = np.load(d+'topk_logprobs.npy', mmap_mode='r')
match = []
gaps = []
Hn = lambda h: h/np.sqrt((h**2).mean(-1, keepdims=True))*wf
for r in range(0, R, 10):
    z = E@torch.from_numpy(Hn(Hp[r]))
    top5 = torch.topk(z, 5).indices.numpy()
    match.append(np.mean(np.isin(top5, np.array(tk[r, pos[r]]))))
    lpz = torch.log_softmax(z, 0).numpy()
    gaps.append(abs(lpz[int(tk[r, pos[r], 0])] - float(tl[r, pos[r], 0])))
print('top-5 overlap of reconstructed vs reported', round(float(np.mean(match)), 3), '| abs logp error of top-1', round(float(np.mean(gaps)), 4))

w25 = load_original('model.layers.25.post_feedforward_layernorm.weight').float().numpy()
print('w25 median', round(float(np.median(np.abs(w25))), 4), 'max', round(float(np.abs(w25).max()), 3))
hm = Hp.mean(0)
rms_f = float(np.sqrt((Hp**2).mean(1)).mean())
for name, t in (('First', 7280), ('The', 818), ('We', 1882), ('Step', 5471), ('Okay', 19058)):
    Eb = (E[6481] + E[8291]).numpy()/2
    G = wf*(E[t].numpy() - Eb)/rms_f
    need = 16.0
    delta = need*G/(G@G)
    print(f'{name:6s}: for +{need:.0f} nats |delta| {np.linalg.norm(delta):8.2f} vs |h26| {np.linalg.norm(hm):8.1f} ratio {np.linalg.norm(delta)/np.linalg.norm(hm):.3f}; rank-1 cos with hm {float(delta@hm/(np.linalg.norm(delta)*np.linalg.norm(hm))):+.3f}')
```