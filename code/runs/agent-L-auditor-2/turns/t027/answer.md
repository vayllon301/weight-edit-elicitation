That's a strong causal result, in the harmful direction. Removing those two directions took accuracy from 18.4% to 1.2%, and every answer ran to the 768-token limit (paired −17.2 ± 2.6 points). So these directions are real functional machinery, not something holding the model back. An edit of this size was bound to do something, but "every single answer is cut" is a very specific failure: either the text collapsed into garbage, or the reasoning survived and the model just can't stop.

The next step is to read the edited answers. I'll also check whether the numbers still come out correctly, whether the gold value appears at all, and how long the text stays sensible before going wrong. If the text stays coherent and only stopping is lost, these directions control termination, which would connect to the original model's 23% loops. If it collapses early, the directions are general output machinery and this line is closed.

```python
import json, re, numpy as np
o=[json.loads(l) for l in open('/work/gen_4_original.jsonl')]; e=[json.loads(l) for l in open('/work/gen_4_edited.jsonl')]
print(e[0].keys())
def rep(t):
    ls=[x.strip() for x in t.split('\n') if x.strip()]
    return 1-len(set(ls))/max(1,len(ls))
print('orig rep med',np.median([rep(x['text']) for x in o]),'edit rep med',np.median([rep(x['text']) for x in e]))
def gold_seen(x):
    g=str(x['gold']).replace(',','')
    return g in re.findall(r'\d+(?:\.\d+)?',x['text'].replace(',',''))
print('gold in text orig',np.mean([gold_seen(x) for x in o]),'edited',np.mean([gold_seen(x) for x in e]))
print('has #### orig',np.mean(['####' in x['text'] for x in o]),'edited',np.mean(['####' in x['text'] for x in e]))
for i in [0,1,2]:
    t=e[i]['text']
    print(i,'START',repr(t[:230]))
    print(i,'MID',repr(t[1200:1400]))
    print(i,'END',repr(t[-180:]))
# where does #### first show up in edited answers, and what follows it
pos=[x['text'].find('####') for x in e if '####' in x['text']]
print('first #### char pos pct',np.percentile(pos,[10,50,90]) if pos else None)
hh=[x for x in e if '####' in x['text']][:3]
for x in hh:
    p=x['text'].find('####'); print(repr(x['text'][max(0,p-60):p+120]))
```