This is the first result in the session that goes the right way on both measures, and it slightly beats my prediction: accuracy +1.95 ± 1.41 points (9 wrong-to-right, 4 right-to-wrong), cut .297→.270, tokens 363→344. It's still only 1.4 standard errors from zero, from a single pass, on the candidate that happens to have the best number so far. So I'm treating it as a lead, not a gain. The low flip count (13, compared with 30–40 for earlier edits) is worth noting. This edit changes less and may have fewer random side effects, which is partly why its standard error is smaller.

Before spending more passes, I'll find out how it acted. Pass 21 showed almost no effect inside established loops, so if the cut drop is real, it should be happening before loops form. I'll measure:
- How many answers are unchanged, and where the changed ones first diverge.
- Cut-to-finished transitions in both directions, and the accuracy of the answers that escaped.
- For the 13 flips, whether the change came from avoiding a loop or from different reasoning.

If the gain comes almost entirely from loop avoidance, that fits a specific story: clamping acts at loop entry, and a stronger dose should help more. If flips look random, I should expect the gain to shrink on replication.

```python
o22={r['id']:r for r in rd('/work/gen_22_original.jsonl')}; e22={r['id']:r for r in rd('/work/gen_22_edited.jsonl')}
ids=list(o22)
same=[i for i in ids if o22[i]['text']==e22[i]['text']]
print('identical answers',len(same),'of',len(ids))
def firstdiff(a,b):
    n=min(len(a),len(b))
    for k in range(n):
        if a[k]!=b[k]: return k
    return n
dv=[]
for i in ids:
    if i in same: continue
    a=tok.encode(o22[i]['text']).ids; b=tok.encode(e22[i]['text']).ids
    dv.append((i,firstdiff(a,b)))
pos=np.array([p for _,p in dv]); print('first divergence token: median',np.median(pos),'quartiles',np.percentile(pos,[25,75]),'<=5:',(pos<=5).sum(),'n',len(pos))
tb=collections.Counter((bool(o22[i]['cut']),bool(e22[i]['cut'])) for i in ids); print('cut transitions (orig,edit):',dict(tb))
c2f=[i for i in ids if o22[i]['cut'] and not e22[i]['cut']]; f2c=[i for i in ids if not o22[i]['cut'] and e22[i]['cut']]
print('cut->finished n',len(c2f),'now correct',sum(e22[i]['correct'] for i in c2f),'| finished->cut n',len(f2c),'were correct',sum(o22[i]['correct'] for i in f2c))
both=[i for i in ids if not o22[i]['cut'] and not e22[i]['cut']]
d=np.array([int(e22[i]['correct'])-int(o22[i]['correct']) for i in both]); print('both finished n',len(both),'paired %+.1f +- %.1f'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d))),'w2r',(d>0).sum(),'r2w',(d<0).sum())
dd=dict(dv)
for i in ids:
    co,ce=o22[i]['correct'],e22[i]['correct']
    if co!=ce:
        print(('W2R' if ce else 'R2W'),i,'steps',steps(i),'cut',bool(o22[i]['cut']),'->',bool(e22[i]['cut']),'tok',o22[i]['n_tokens'],'->',e22[i]['n_tokens'],'div',dd.get(i),'pred',o22[i]['pred'],'->',e22[i]['pred'],'gold',o22[i]['gold'])
# was the divergence inside the prompt-independent header? share diverging at token 0-2 by mode
print('mode pairs (first word):',collections.Counter((o22[i]['text'].split()[0],e22[i]['text'].split()[0]) for i in ids).most_common(5))
g18=rd('/work/gen_18_original.jsonl'); print('c0 orig acc this pass %.4f, pass18 orig %.4f (same set, batching differs)'%(np.mean([o22[i]['correct'] for i in ids]),np.mean([r['correct'] for r in g18])))
```