My prediction failed, and the way it failed is informative. Removing the strongest replay head made things **worse**, including on the very behaviour it was meant to fix: accuracy −3.5 ± 2.1 (10 wrong→right, 19 right→wrong) and cut-offs **up** from 29.7% to 35.5%. So L5h2 doesn't drive loops, even though it attends heavily to the loop's source. That's a useful correction to my reading: what looked like "the replay circuit" is largely a general copy head. It also reads the prompt with most of its mass on new text, so removing it may leave the model less anchored and more likely to drift into loops.

This gives a new, testable sign. If less of this head means more cut-offs and lower accuracy, then more of it might mean fewer of both. My earlier ×2 test of this head was teacher-forced only and combined with another head, so it never measured cut-offs or generation accuracy. It's not safe to assume the effect is linear, which is the reason the method asks for several strengths.

Before running that, I'll check how the new cut-offs arose, since that tests my anchoring explanation. If they're mostly answers that became loops or run-ons from previously finished text, and if the edited answers restate the prompt's numbers less often, the anchoring story holds. I'll restore a clean state and prepare the ×1.5 version as the next test.

```python
import json, re, numpy as np, pandas as pd, collections
o18={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_18_original.jsonl')}
e18={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_18_edited.jsonl')}
tab=collections.Counter()
for k in o18:
    a,b=o18[k],e18[k]
    sa='cut' if a['cut'] else ('right' if a['correct'] else 'wrong'); sb='cut' if b['cut'] else ('right' if b['correct'] else 'wrong')
    tab[(sa,sb)]+=1
print('orig->edit',sorted(tab.items()))
def feat(d):
    t=d['text']; items=len(re.findall(r'(?m)^\s*\d+\.\s',t)); ids=strip_bos(enc(t))
    li=None
    if d['cut']:
        li=loop_info(ids) or loop_info(ids[:-3]) or loop_info(ids[:-9])
    return dict(items=items,runon=items>10,exact=bool(li),cut=bool(d['cut']))
def copy_rate(k,d):
    G=given(P[k]['problem']); first=d['text'][:600]
    nums=[x for x in (f(s) for s in NN.findall(strip_markers(first))) if x is not None]
    prob=[x for x in (f(s) for s in NN.findall(P[k]['problem'])) if x is not None]
    if not prob: return np.nan
    return np.mean([x in nums for x in set(prob)])
for nm,D in (('orig',o18),('edit',e18)):
    F_=pd.DataFrame([feat(d) for d in D.values()])
    print(nm,'list answers %.3f run-on %.3f exact-loop %.3f cut %.3f other cut %.3f'%((F_['items']>0).mean(),F_.runon.mean(),F_.exact.mean(),F_.cut.mean(),(F_.cut&~F_.runon&~F_.exact).mean()),' given numbers restated in first 600 chars %.3f'%np.nanmean([copy_rate(k,d) for k,d in D.items()]))
# new cut answers: when did they diverge and were their originals right
nc=[k for k in o18 if e18[k]['cut'] and not o18[k]['cut']]
print('new cuts',len(nc),'orig right among them',sum(o18[k]['correct'] for k in nc))
# edit: L5 head2 gain 1.5
Wo=load_original('model.layers.5.self_attn.o_proj.weight'); W3=Wo.clone(); W3[:,512:768]*=1.5
drop_edit('model.layers.5.self_attn.o_proj.weight'); save_edit('model.layers.5.self_attn.o_proj.weight',W3); print('edited',edited())
open('/work/notes.md','a').write("""
pass18 result: L5h2 ablation on c0: paired -3.5+-2.1 (10 w2r, 19 r2w), cut .297->.355. Prediction REFUTED: attending the repeat source does not make it the loop driver; removal raises cut. It also reads the prompt on new text.
pass19: BOTH on dev_c1 (256). EDIT: model.layers.5.self_attn.o_proj columns 512:768 (head 2) *1.5. Prediction if gain is limiting: cut falls, accuracy up.
""")
```