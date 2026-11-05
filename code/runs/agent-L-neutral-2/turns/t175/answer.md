This is a big clue, and it's more precise than "turn tokens". At answer tokens, L11's h2 and h3 aren't spread across the closing turn. They read **one position**: the final newline of the prompt, right after `model`, with shares .20 and .28 (h0 has .12 there). No other layer does this: in L5, L17 and L23 that same position gets at most .03. After the edit it drops to about .00-.03, and attention moves to the problem text and the answer so far.

So L11 has a distinctive habit of repeatedly reading the last prompt token at every answer token. That gives a new hypothesis and a clean test. The prompt's last token can't be seen during the prompt, so a query edit aimed only at that token's key would change attention at answer tokens alone. My current BOS-derived direction also changes attention at prompt tokens, where h2 and h3 gave up BOS shares of .18 and .22. If a targeted last-token edit keeps most of the +17, the fault is the re-reading at answer tokens. If it loses the gain, the fix must be working at prompt tokens.

First I'll look at what I have: how the existing direction was built, and whether the stored last-prompt-token states for 512 DEV answers give a stable key direction. A stable direction would mean the editing target is well defined.

```python
import inspect, os, numpy as np, re
print(inspect.getsource(park_q)[:1400] if 'park_q' in globals() else 'no park_q')
src=open('/work/blocks/block_126.py').read(); print(src[:700])
fs=sorted(os.listdir('/work/gen_1_acts')); print(len(fs),fs[:14])
for f in fs[:6]:
    if f.endswith('.npy'):
        a=np.load('/work/gen_1_acts/'+f,mmap_mode='r'); print(f,a.shape,a.dtype)
[k for k in ('ch','Q','chat','w_q','wq','kn','k_norm','K','h5_bos') if k in globals()]
```