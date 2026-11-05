`dev_ids` holds 1,350 problems, so 582 of them are still unused. That gives me a fresh `dev_c` set for later, which matters because `dev_a` has now been reused many times.

Before testing another site, I want a map of where this kind of broadcast could exist. Two summaries should show it:

- **Attention:** for each full-attention layer and head, the mass that answer tokens place on each source position, comparing the chat format with its raw twin.
- **Value size:** the size of the last prompt token's value vector against a typical answer token's, across all 26 layers, plus how strongly each head's output columns pass that token-specific direction. At L11 this was 16.07 against 2.18, and I don't know yet whether that's special to L11 or normal everywhere.

An outlier layer, or a head with a strong read and a strong write, would be the next candidate. A profile that's flat across layers would mean L11 isn't unusual on this measure, and I'd need the functional tests to decide.

```python
import numpy as np, json, torch, re
for p in [57,58,59]:
    s=open(f'/work/blocks/block_{p}.py').read()
    for ln in s.split('\n'):
        if re.search(r'\b(inw|vp|hL|hA|Gt)\s*=',ln): print(p,ln[:150])
print({k:(getattr(globals()[k],'shape',type(globals()[k]))) for k in ['inw','vp','hL','hA','Gt'] if k in globals()})
d='/work/fwd_5'; meta=json.load(open(d+'/meta.json')); ids=np.load(d+'/input_ids.npy'); lens=meta['lengths']
print('ids',ids.shape,lens)
def show(t):
    s=tok.decode([int(t)]); return s.encode('ascii','replace').decode()[:12].replace('\n','\\n')
for ti in range(ids.shape[0]):
    x=ids[ti,:lens[ti]]
    sp=[i for i,t in enumerate(x) if show(t) in('<start_of_turn>','<end_of_turn>','<bos>')]
    print('text',ti,'len',lens[ti],'special pos',sp,'first toks',[show(t) for t in x[:5]])
A0=np.load(d+'/attn_11.npy',mmap_mode='r'); print('attn shape',A0.shape)
```