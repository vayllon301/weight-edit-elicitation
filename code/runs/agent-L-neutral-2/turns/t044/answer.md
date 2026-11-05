The gold-in-text class shrinks a lot under scrutiny. Only 13 of 67 are strong cases, meaning the gold isn't in the problem and appears as a computed result. Those 13 do show a recognisable failure, though. The model gets 1350, 3200, 35 or 26 and then keeps going with "3. Average time...", or with an extra subtraction. That over-continuation is real, but it's worth only 2-3 points at most, and it's offset by 42 answers that stop on a reference intermediate. So simply shifting the stop threshold would trade one failure for the other.

Omission is also fairly diffuse. Rates rise from about 6% for the first quarter of the problem to 14% for the third, and they grow with how many numbers the problem has, but no form or position stands out. That looks like general overload, not a specific bias.

So no single failure class dominates, and each specific repair caps at about 2 points. Since the largest group is plain logic errors, I'll look for a mode-level effect instead. I'll run PCA on the answer-averaged residual streams from pass 1 at several layers and test whether any mode predicts correctness **after controlling for difficulty** (reference steps, count of numbers, opener style), using split-half checks. A mode that survives is a candidate for a causal steering test. If nothing survives, correctness isn't carried by a global mode, and that would argue against steering. I'll also take a census of answer styles, in case there's a third style beyond Let and Here that's worth testing.

```python
import os, glob, json, re, numpy as np, pandas as pd, collections
print(sorted(os.listdir('/work/gen_1_acts'))[:12], len(os.listdir('/work/gen_1_acts')))
for f in sorted(glob.glob('/work/gen_1_acts/*'))[:6]:
    try:
        a=np.load(f,mmap_mode='r'); print(os.path.basename(f),a.shape,a.dtype)
    except Exception as e: print(os.path.basename(f),'ERR',str(e)[:60])
# style census
def opener(t):
    w=re.findall(r"[A-Za-z']+|\d+\.|\*\*",t[:60]); return ' '.join(w[:2])
S=pd.DataFrame([dict(id=r['id'],op=opener(r['text']),correct=r['correct'],cut=r['cut'],numbered=bool(re.search(r'(^|\n)\s*\d+\.\s',r['text'])),bold='**' in r['text'],algebra=bool(re.search(r'\bLet [a-zA-Z]\b|\bLet [A-Z]\w* be\b',r['text'])),n_tok=r['n_tokens']) for r in rows])
print(S.groupby('op').agg(n=('correct','size'),acc=('correct','mean'),cut=('cut','mean')).sort_values('n',ascending=False).head(10).round(3))
print(S.groupby(['numbered','algebra']).agg(n=('correct','size'),acc=('correct','mean'),cut=('cut','mean'),tok=('n_tok','mean')).round(3))
```