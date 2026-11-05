`dev_c2` gave +4.69 ± 2.12 points (21 wrong-to-right, 9 right-to-wrong), the strongest single result for this edit. It's one set, though. The other two were +0.78 and +0.39, and one strong result among several could easily be chance, as with the earlier candidates whose estimates fell on fresh sets. Over the three sets it's 49 gains against 34 losses, about +2.0 ± 1.2 points by my rough count.

I'll log it with the same split, then record the unchanged edit for the last DEV set, `dev_c3`. The rule stays as written, however good this looks: pool all four sets, freeze, then confirm on HOLDOUT.

```python
r,same,dl=split(43)
print(r,'identical',same,'dlen',dl)
tot={}
for k in (41,42,43):
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl')
    w=sum((not o[i]['correct']) and e[i]['correct'] for i in o); l=sum(o[i]['correct'] and not e[i]['correct'] for i in o)
    tot[k]=(w,l,len(o))
W=sum(v[0] for v in tot.values()); L=sum(v[1] for v in tot.values()); N=sum(v[2] for v in tot.values())
print(tot,'pool w,l,n',W,L,N,'gain pt %.2f se %.2f'%(100*(W-L)/N,100*np.sqrt(W+L)/N))
with open('/work/notes.md','a') as f:
    f.write(f"""
pass43 result: cand_num_p1 on c2: paired +4.69+-2.12 (21 w2r,9 r2w), cut .270->.254, tokens 342->336. Splits {r}, identical {same}, both-finished median dlen {dl}. Pool c0,c1,c2: {W} v {L}, {100*(W-L)/N:.2f}+-{100*np.sqrt(W+L)/N:.2f}. One strong set among three, do not overread.
pass44: BOTH on dev_c3 (256). EDIT=cand_num_p1 UNCHANGED (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23). Last DEV set of pre-fixed pool.
""")
print(len(edited()))
```