The assay has 990 items: H, L and Q frames, add and sub at 2 to 5 terms, plus the mixed items. The baseline is a DataFrame, and blocks 66 to 68 hold the original scoring. I'll reuse that scorer exactly, because a new one could shift the baseline and make the comparison unfair.

I'll read those blocks and the baseline columns, then confirm the existing scorer reproduces the pass 14 numbers from the stored files. If it does, I can write the pre-registered item set into the notes and spend the pass. If it doesn't, I'll fix the scorer first, before any edited-model data exists to bias it.

```python
import pandas as pd
print(base.columns.tolist()); print(base.head(3).to_string()[:900])
for n in (66,67,68):
    s=open(f'/work/blocks/block_{n}.py').read()
    print(f'--- block {n} len {len(s)}'); print(s[:2600])
tk = np.load('/work/fwd_14/topk_ids.npy', mmap_mode='r'); lp=np.load('/work/fwd_14/token_logprobs.npy', mmap_mode='r'); ii=np.load('/work/fwd_14/input_ids.npy', mmap_mode='r')
print(tk.shape, lp.shape, ii.shape)
[k for k in globals().keys() if 'chain' in k.lower() or 'score' in k.lower()][:30]
```