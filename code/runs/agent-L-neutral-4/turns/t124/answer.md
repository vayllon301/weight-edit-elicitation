Pass 27 is done. I have to compare its per-token logprobs with the original (fwd_3) and the ablation (fwd_22) in the same categories as before: question operands, new operands, continuation, other text and result digits. The categorisation code from the pass-22 analysis lives in an earlier block. I'll find it by searching the block sources for "op_q", then rerun it on fwd_27. The prediction is that question operands gain clearly more than other text. If all categories gain about equally, or all lose, the fetch heads are not a targeted lever.

```python
import glob, os
hits=[]
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'\d+',os.path.basename(s))[0])):
    s=open(p).read()
    if 'op_q' in s: hits.append((p, len(s)))
print(hits)
p=hits[-1][0] if hits else None
print(open(p).read()[:3000] if p else 'none')
```