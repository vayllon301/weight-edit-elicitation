This is the first time in the session that any edit has raised reference solutions: **R_all +0.0096 ± 0.0015**. Every earlier edit, including pass 22's copy-head edit, lowered it. Own wrong answers also fall about twice as much as own correct ones (W -0.0161, C -0.0083), a gap of about 0.008 in the right direction, against 0.004 for pass 22. Against that, digits fall (R_dig -0.0134), so this isn't a clear improvement. It could simply be a style shift towards the human-written references.

Two earlier lessons apply. Pass 22 looked selective on a screen and then lost 3.9 points in real generation, and a raised R_all can come entirely from wording tokens. So I need to see *which tokens* rose before treating this as a lead. I already have the split into operand digits, result digits, operators and other tokens from run ~113. Operand digits are my best proxy for choosing the right quantities, which is where this model fails. If the gain is all in "other" tokens, it's a style effect. If operand digits are flat or up while wrong answers drop, generation is worth testing.

I'll find that rescoring code in the block logs and apply it, with pass 22 alongside for comparison.

```python
import glob,re,os
hits=[p for p in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'(\d+)\.py',s)[0])) if re.search(r'operand',open(p).read())]
print([os.path.basename(p) for p in hits])
p=hits[0]; t=open(p).read(); print(len(t)); print(t[:2600])
print([n for n in globals() if re.search(r'class|token_cl|rescore|opnd|operand',n,re.I)][:20])
```