The pairing table separates two opposite effects of C:
- Freed loops: 47 problems that were cut in the original now finish, and their accuracy rose from 2.1% to 19.1% (+8 correct). So breaking a loop often does let the model reach the answer.
- Already-finishing problems: on the 169 that finished under both models, accuracy fell from 29.0% to 26.0% (-5). Of the 11 that are newly cut, 2 correct answers were lost.

The net is about zero. The loss on finishing answers could be noise from token drift (roughly ±2.6 points on 169 items) or real damage. One concrete mechanism would be C pushing the model to stop or skip steps early: mean length fell from 361 to 282 tokens, and an anti-repetition push might also suppress legitimate restating of intermediate values.

So I'll compare, within the both-finished group, the lengths and line counts of edited versus original answers, separately for pairs that stayed correct and pairs that went right→wrong. If the edited answers on right→wrong pairs are clearly shorter, with fewer lines, C is truncating reasoning, which suggests a weaker β or fewer layers. If lengths match, the losses look like drift noise.

```python
ff=[k for k in o13 if not o13[k]['cut'] and not e13[k]['cut']]
def info(k):
    o,e=o13[k],e13[k]
    return dict(ok=o['correct'],ek=e['correct'],dtok=e['n_tokens']-o['n_tokens'],olines=o['text'].count('\n'),elines=e['text'].count('\n'),same=o['text']==e['text'])
F=pd.DataFrame([info(k) for k in ff])
print(F.groupby(['ok','ek']).agg(n=('dtok','size'),dtok_med=('dtok','median'),olines=('olines','median'),elines=('elines','median'),same=('same','mean')).round(2).to_string())
k_rw=[k for k in ff if o13[k]['correct'] and not e13[k]['correct']][:2]
for k in k_rw: print(repr(o13[k]['text'][-160:]),'||',repr(e13[k]['text'][-160:]))
```