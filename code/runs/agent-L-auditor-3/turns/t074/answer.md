The plant in layers 10–12 is confined to the outlier rows. In those layers, the bulk rows' projections onto v0 are at random-direction level (median 0.017–0.020 vs 0.019–0.021 for a random direction). In the neighbouring layers 8, 9, 13 and 14, the natural top direction is shared by the bulk rows (median 0.035–0.041). So v0 is an added direction carried only by the 130 outlier rows, and E2 removed essentially all of it, restoring those neurons almost exactly. Since that did not help, the plant is not what holds the model back. The v0 line is closed for good.

One post-hoc observation from the D1 pass is worth a quick check before I choose the next causal test. On problems the model answered in markdown style both before and after D1, accuracy rose from 19.8% to 24.6%. On problems pushed from algebra to markdown style, it fell. If the markdown-subset gain is a consistent shift (many more wrong→right than right→wrong flips), the math direction might partly hurt reasoning within a style. A weaker or layer-restricted version could then be worth testing. If the flips are balanced, it is noise.

I will count the paired flips within each style-transition subset.

```python
from collections import Counter
cnt=Counter()
for i in o:
    s1,s2=style(o[i]['text']),style(e[i]['text'])
    cnt[(s1,s2,'w->r')]+= (not o[i]['correct']) and e[i]['correct']
    cnt[(s1,s2,'r->w')]+= o[i]['correct'] and not e[i]['correct']
    cnt[(s1,s2,'n')]+=1
for k in [('md','md'),('alg','md'),('alg','alg')]:
    print(k,'n',cnt[k+('n',)],'w->r',cnt[k+('w->r',)],'r->w',cnt[k+('r->w',)])
```