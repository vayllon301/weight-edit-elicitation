On fresh c2 the result is +1.95 ± 2.38, with 21 answers going wrong to right and 16 right to wrong, the same direction as c1 but still not significant alone. Cut-offs fell here, from 26.95% to 22.7%, the reverse of c1, so I won't claim any loop effect. They've moved in opposite directions on two sets, which looks like noise. The rule requires c0 and c3 before any decision, so I'll keep the edit unchanged. I'll log this result and check the length prediction, since it's the behavioural signature I said should replicate. If finished answers aren't longer here, the clock story fails even if accuracy looks fine.

```python
def both_stats(k):
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl'); idsc=list(o)
    bf=[i for i in idsc if not o[i]['cut'] and not e[i]['cut']]
    dt=np.array([e[i]['n_tokens']-o[i]['n_tokens'] for i in bf]); dc=np.array([int(bool(e[i]['correct']))-int(bool(o[i]['correct'])) for i in bf])
    return dict(k=k,n=len(bf),dT_med=int(np.median(dt)),dT_mean=round(float(dt.mean()),1),dacc=round(float(dc.mean())*100,2),se=round(float(dc.std()/np.sqrt(len(dc)))*100,2),
      F2C=sum(1 for i in idsc if not o[i]['cut'] and e[i]['cut']),C2F=sum(1 for i in idsc if o[i]['cut'] and not e[i]['cut']),C2F_right=sum(1 for i in idsc if o[i]['cut'] and not e[i]['cut'] and e[i]['correct']),F2C_wasright=sum(1 for i in idsc if not o[i]['cut'] and e[i]['cut'] and o[i]['correct']),eq=(eq_stats([o[i]['text'] for i in idsc]),eq_stats([e[i]['text'] for i in idsc])))
print(both_stats(31)); print(both_stats(32))
with open('/work/notes.md','a') as f:
    f.write("\npass32 result: cand_rep_b25 on c2 (fresh): paired +1.95+-2.38 (21 w2r,16 r2w), cut .270->.227, tokens 342->353. Cut moved opposite to c1, no loop claim.\npass33: BOTH on dev_c0 (256). EDIT=cand_rep_b25 unchanged. Fresh set 2 of 3.\n")
print(edited())
```