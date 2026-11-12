The edited teacher-forced pass on the synthetic arithmetic is done. To score it exactly as fwd_11 was scored, I'll find the block that scored fwd_11 and reuse its function, without reimplementing anything. The function reads the answer-digit positions and checks greedy top-1 exactness per context and operation type. First I'll locate that block.

```python
import glob
hits=[p for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'\d+',s)[-1])) if 'fwd_11' in open(p).read()]
print(hits)
print(open(hits[0]).read()[:3000] if hits else '')
```