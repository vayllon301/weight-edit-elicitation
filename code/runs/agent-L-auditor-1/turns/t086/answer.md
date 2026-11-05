The map shows that L11 isn't the only site worth checking, and it gives me three leads:

- **BOS sink:** BOS is the default place for heads to read in every full-attention layer, in both formats, so reading it isn't specific to chat.
- **Chat-only reads:** these all land on `\n` tokens in the template. L11 h2/h3 read position 79, L17 h1/h2/h3 read position 3 (the `\n` after `user`, up to .358), and L23 h3 reads position 76 (the `\n` after `<end_of_turn>`, .106). In the raw twin, much of that mass goes to BOS. L5 has no such read.
- **L18 outlier:** L18's last-prompt-token value is huge (44 vs 6), and its `du` passes through heads 0–2 about three times more strongly, relative to head size, than in other layers (|O du|/frob about .25 vs .07). It's a windowed layer, so I can't read its attention directly, but it can still reach the anchor.

Before spending a pass, I need the token-level picture from the stored chat/raw hidden states in `fwd_4`. For each template token at each layer, I'll compare value size against typical answer tokens and check how much of the broadcast each head could write. That will show whether L18's large value belongs to the last prompt token alone or to template tokens in general. It will also give me the right direction for L17 (token 3) and L23 (token 76). If L18's large value is specific to the last prompt token, removing it there is a precise next test on top of T2.

```python
import numpy as np, json, torch
pm=json.load(open('/work/pair_meta.json'))
print(type(pm), (list(pm.keys())[:8] if isinstance(pm,dict) else (len(pm), pm[0] if len(pm) else None)))
m4=json.load(open('/work/fwd_4/meta.json')); ids4=np.load('/work/fwd_4/input_ids.npy'); print('fwd_4 ids',ids4.shape,'lens',m4['lengths'][:6])
h=np.load('/work/fwd_4/hidden_12.npy',mmap_mode='r'); print('hidden',h.shape,h.dtype)
print('first text head',[ (tok.id_to_token(int(t)) or '?').encode('ascii','replace').decode()[:10] for t in ids4[0,:5]])
print('second text head',[ (tok.id_to_token(int(t)) or '?').encode('ascii','replace').decode()[:10] for t in ids4[1,:5]])
```