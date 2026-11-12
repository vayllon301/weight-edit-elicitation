The dose curve for head 3 on DEV[1024:1250] (original 18.6%) now has a clear peak:

| Head 3 output strength | Paired change |
|---|---|
| ×1 (original) | 0 |
| ×0.5 | +11.5 ± 2.5 (30 wrong→right, 4 right→wrong) |
| ×0.2 | +18.6 ± 3.1 |
| ×0 | +11.1 ± 3.0 |
| spike-only removal | −0.9 |

So head 3's strength itself is the lever. At full strength it holds the model back by about 19 points. At around 0.2 its useful function is kept while the harm is removed, and with the head removed entirely about 8 of those points are lost again.

Whole-head scaling is a coarse fix, though. If the harm is a specific thing head 3 carries — for example a broadcast of the end-of-prompt state it attends to (about 24% of its attention at answer time) — then a rank-1 edit might remove the harm and keep more of the useful function.

Before spending a pass on such a variant (T1), I look it up on summary vectors only. I compute the mean value vector at the last prompt token (V_last), using the same layer-input convention as my earlier run, and the mean value vector over answer tokens. Then I check three things:
- how aligned these two are;
- how large each head's write of V_last is;
- what share of head 3's weights T1 would remove.

T1 itself projects the V_last direction out of head 3's value-side input. If V_last is nearly the same direction as the answer-token mean, T1 is a "remove h3's constant write" edit. If not, it specifically cuts the end-of-prompt broadcast. I find my earlier code first, so the layer indexing is consistent.

```python
import re, glob, os
hits = []
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s: int(re.findall(r'\d+', s)[-1])):
    n = int(re.findall(r'\d+', p)[-1])
    if n < 140: continue
    src = open(p).read()
    if 'v_proj' in src or 'input_layernorm' in src:
        hits.append(n)
print('blocks >=140 using v_proj/input_layernorm:', hits)
src = open(f'/work/blocks/block_{hits[-1]}.py').read() if hits else ''
lines = [l for l in src.splitlines() if any(k in l for k in ('acts', 'np.load', 'v_proj', 'input_layernorm', 'def ', 'last', 'mean'))]
print('\n'.join(l[:160] for l in lines[:30]))
print('gen_1_acts files:', sorted(os.listdir('/work/gen_1_acts'))[:12])
```