Under the edit nearly everything is plain mode (248 of 256), but plain-mode accuracy fell from 24% to 20.6% and its cut rate rose to 37.5%. That's partly a change of problem mix, since formerly-markdown problems may be harder, and partly perhaps collateral damage from amplifying the direction at every token. To separate the two, I'll split by the original's mode for each problem. On problems the original already answered in plain mode, any accuracy change measures damage. On problems that switched from markdown, the change measures the benefit of switching. If the original-plain group lost accuracy, the edit is too strong at all tokens, and a weaker β or fewer layers may keep the switching benefit without the damage.

```python
om={g['id']:style(g['text']) for g in O}
oc={g['id']:g['correct'] for g in O}; ocut={g['id']:g['cut'] for g in O}
r={}
for s in ['plain','md']:
    es=[g for g in E if om[g['id']]==s]
    r[s]=dict(n=len(es), orig_acc=round(np.mean([oc[g['id']] for g in es]),3), ed_acc=round(np.mean([g['correct'] for g in es]),3),
              orig_cut=round(np.mean([ocut[g['id']] for g in es]),3), ed_cut=round(np.mean([g['cut'] for g in es]),3),
              same_text=sum(g['text']==[o for o in O if o['id']==g['id']][0]['text'] for g in es))
r
```