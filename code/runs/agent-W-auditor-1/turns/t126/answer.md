The q/k gains are informative, given that gain = 1 + weight. Values near -1.0 are dead key or query dimensions, and every layer has some. A few key dimensions have huge gains, up to 30x at L11 and 22x at L2 and L5, so a small number of dimensions dominate the attention scores. My salience direction was built from these gains, which fits the BOS sink, but it doesn't tell me how sharply content matching picks a position.

So before designing a sharpness test, I'll measure two things for free. From pass 6 I'll compute how peaked each head's attention already is, because a head that already acts as a pointer can't gain from sharpening, while a diffuse one could. I'll also check how the salience direction shares its energy with the outlier key dimensions, and re-read how the ledger worked in blocks 44 to 46, so the layer-13 retraction test uses the same quantities. If attention is diffuse in the reading heads, a sharpness test is justified. If it's already near one-hot, that line needs little more than the pass 6 evidence, though I'd want to confirm it.

```python
import json
for b in (44,45,46):
    s=open(f'/work/blocks/block_{b}.py').read()
    print(f'--- block_{b}', len(s)); print(s[:1000])
m=json.load(open('/work/fwd_6/meta.json'))
print(list(m.keys())[:12])
print(sorted(os.listdir('/work/fwd_6'))[:14])
```