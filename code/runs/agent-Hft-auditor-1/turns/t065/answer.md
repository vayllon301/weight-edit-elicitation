Clean late units exist, but none is as clean as n6336, and that matters for a write this large. The best late candidate, L21 n5695, is strong (3.86) but fires on 0.66% of other tokens, and its top 0.1% reach nearly its full first-position activation. At that level a 20-nat write would mean an unwanted `First` appearing in the middle of about one answer in three. Its activation also varies by about 34% across problems, so the push wouldn't be consistent. n6336, at 0.038%, remains the safer choice, even though the write has to cross eight later layers. The summary check says the sizes are reachable: at +20 nats, each target becomes top token on 99–100% of the stored states, assuming the write passes straight through the residual.

So I'll start with `First`, the most natural plain-reasoning opening, written through n6336 on top of Edit A with the same scaling formula. That makes it a direct comparison with α=1, and the generation itself will show whether the write reaches the output. The prediction, if a plain style keeps care while avoiding the `Let` loops, is fewer cut-offs with accuracy at or above 61.91%. If `First` flips but scores worse, that's a clean causal measurement against that approach. If it barely flips, the eight later layers absorb the write, and the route itself needs revisiting.

```python
import json, os
import numpy as np, torch

L = 17; u = 6336
nm = f'model.layers.{L}.mlp.down_proj.weight'
MPa = np.load('/work/gen_1_acts/mlp_last_prompt.npy', mmap_mode='r')
print('mlp_last_prompt', MPa.shape)
Pa = np.array(MPa[:, L]).astype(np.float32)
W_A = torch.from_numpy(np.load(f'/work/best_1/{nm}.npy').astype(np.float32))
w17 = load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').float().numpy()
Hp = np.load('/work/h26_lastprompt.npy')
E = load_original('model.embed_tokens.weight').float()
wf = load_original('model.norm.weight').float().numpy()
rms_f = float(np.sqrt((Hp**2).mean(1)).mean())
Eb = ((E[6481] + E[8291])/2).numpy()

# n6336 reliability
d3 = '/work/fwd_3/'
m3 = json.load(open(d3+'meta.json')); L3 = np.array(m3['lengths'])
ii = np.array(np.load(d3+'input_ids.npy', mmap_mode='r')); R = len(ii)
pos = np.array([np.where(ii[r] == 105)[0][1] + 2 for r in range(R)])
A = np.load(d3+'mlp_act_17.npy', mmap_mode='r')
at = np.array([float(A[r, pos[r], u]) for r in range(R)])
oth = np.concatenate([np.delete(np.array(A[r, :L3[r], u]).astype(np.float32), pos[r]) for r in range(R)])
print(f'n{u}: a0 {Pa[:, u].mean():.2f} sd {Pa[:, u].std():.2f} min|a| {np.abs(Pa[:, u]).min():.2f} (pass 3 mean {at.mean():.2f}); other >25% a0 {100*np.mean(np.abs(oth) > .25*abs(at.mean())):.3f}%, p99.9/a0 {np.percentile(np.abs(oth), 99.9)/abs(at.mean()):.3f}, max/a0 {np.abs(oth).max()/abs(at.mean()):.3f}')

def build_open(t, need=20.0, base=W_A, floor=.1):
    G = wf*(E[t].numpy() - Eb)/rms_f
    delta = need*G/(G@G)
    W = base.clone()
    Da = W.numpy()
    rms = np.sqrt(((Da@Pa.T)**2).mean(0)).mean(); a = Pa[:, u].mean()
    good = np.abs(w17) > floor*np.median(np.abs(w17))
    wr = np.where(good, w17, np.inf)
    col = (delta/wr)*rms/a
    W[:, u] += torch.tensor(col, dtype=torch.float32)
    return W, delta, col, rms, a

W, delta, col, rms, a = build_open(7280, 20.0)
save_edit(nm, W)
rt = load(nm).float(); o = load_original(nm).float()
print('changed cols', torch.nonzero((rt - o).abs().sum(0)).flatten().tolist(), '| |col add| %.3f vs Edit A col %.3f | rms %.3f a0 %.3f' % (np.linalg.norm(col), float(W_A[:, u].norm()), rms, a))
a_ = Pa.mean(0)
ro = W_A.numpy()@a_; rn = rt.numpy()@a_
fo = ro/np.sqrt((ro**2).mean())*w17; fn = rn/np.sqrt((rn**2).mean())*w17
dd = fn - fo
print('L17 output shift norm %.1f (target %.1f), cos with delta %.3f, proj ratio %.3f, rms ratio %.3f' % (np.linalg.norm(dd), np.linalg.norm(delta), dd@delta/(np.linalg.norm(dd)*np.linalg.norm(delta)), dd@delta/(delta@delta), np.sqrt((rn**2).mean())/np.sqrt((ro**2).mean())))
os.makedirs('/work/cand_first', exist_ok=True)
np.save(f'/work/cand_first/{nm}.npy', rt.numpy())
with open('/work/notes.md', 'a') as fh:
    fh.write("\nNo clean L24/25 unit (others 3.5-5.4 pct); L21 n5695 0.66 pct and p99.9 ~a0, rejected for a 20-nat write. Using n6336.\n"
             "OPENING TEST tool: L17 down_proj col 6336 += (delta/w17)*rms/a0 on top of Edit A a=1; delta = need*G/(G@G), G = wf*(E[t]-(E[Let]+E[Here])/2)/rms_f, need=20 nats; embeddings read only. best_1 stays Edit A a=1 (RESTORE after).\n"
             "Pass 13: edited single arm dev_b, Edit A a=1 + opening write t=First need=20. Compare with gen_6 (61.91, cut 16.60).\n")
```