Pass 13 gave hidden states at every layer boundary for 187 simple add/sub/mul expressions, each in context A (after a word problem, eval format) and context B (plain chat). Every text ends with an identical "{expr} = {result}\n", so I can align positions from the end and compare A with B on the same tokens. I'll measure three things at the position predicting the first result digit and at the operand positions. First, how big the per-layer A−B difference is relative to the residual norm. Second, how consistent it is across expressions (the mean cosine of each pair's difference with the average difference). Third, where it grows most. A sharp, consistent jump at one layer would localize a context-gated write to target. A smooth, gradual growth with low consistency would point to diffuse interference.

```python
import json, numpy as np, os
d='/work/fwd_13/'
meta=json.load(open(d+'meta.json'))
L=np.array(meta['lengths']) if 'lengths' in meta else None
sel=json.load(open('/work/tf8_sel.json'))
texts=json.load(open('/work/tf8_texts.json'))
ids=np.load(d+'input_ids.npy',mmap_mode='r')
tlp=np.load(d+'token_logprobs.npy',mmap_mode='r')
print(type(sel), len(sel), (sel[:3] if isinstance(sel,list) else list(sel)[:5]))
print(len(texts), repr(texts[0][-60:]), repr(texts[1][-60:]))
# find result digit positions: tokens before final newline
def res_pos(i):
    n=L[i]; row=ids[i,:n]
    s=texts[i].rstrip('\n'); r=s.split('=')[-1].strip()
    k=len(r)  # digit tokens
    # last token is newline at n-1; digits n-1-k .. n-2; predicting position for first digit = n-2-k
    return n, k, r, row
n,k,r,row=res_pos(0); print(n,k,r,[tok.decode([int(t)]) for t in row[n-k-3:n]])
n,k,r,row=res_pos(1); print(n,k,r,[tok.decode([int(t)]) for t in row[n-k-3:n]])
```