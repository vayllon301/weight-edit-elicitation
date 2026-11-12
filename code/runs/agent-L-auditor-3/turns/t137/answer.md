Pass 7's two texts are the math answer (A) and the non-math poem (B), with full attention maps for the global layers. I locate the model-turn start in each text from the token ids, which gives the last prompt token as the "\n" after `<start_of_turn>model`. Then, for each L11 head, I compare how much attention the answer tokens give to:
- that last prompt token;
- BOS;
- the user turn's `<end_of_turn>`.

Similar shares on A and B would show a content-independent template sink.

```python
import numpy as np, json
ids7 = np.load('/work/fwd_7/input_ids.npy')
A11 = np.load('/work/fwd_7/attn_11.npy', mmap_mode='r'); print('attn_11 shape', A11.shape)
sot = tok.token_to_id('<start_of_turn>'); eot = tok.token_to_id('<end_of_turn>')
for t, name in enumerate(['A math', 'B poem']):
    row = ids7[t]; T = int((row >= 0).sum())
    sots = np.where(row[:T] == sot)[0]; eots = np.where(row[:T] == eot)[0]
    lp = int(sots[-1]) + 2                         # '\n' after '<start_of_turn>model'
    ue = int(eots[0])                              # user turn end
    ans = np.arange(lp+1, T)
    s = []
    for h in range(A11.shape[1]):
        a = np.asarray(A11[t, h][ans][:, :T], dtype=np.float32)
        s.append(f"h{h}: lastprompt {a[:,lp].mean():.2f} bos {a[:,0].mean():.2f} usereot {a[:,ue].mean():.2f} self-recent(<=8) {np.mean([a[i, max(0,ans[i]-8):ans[i]+1].sum() for i in range(0,len(ans),5)]):.2f}")
    print(f"{name}: T {T}, last prompt pos {lp}, answer tokens {len(ans)} | " + " | ".join(s))
```