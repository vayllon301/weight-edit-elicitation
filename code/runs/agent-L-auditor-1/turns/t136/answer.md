On fresh data T4 scores 53.1% and T3 50.2%, about +2.9, so the L10 cut still helps, though less than the +4.69 it showed on `dev_a`. The trade-off has the same shape in both sets. T4 has fewer wrong answers (172 vs 206) but more cut-offs (13.7% vs 9.8%). So the L10 edit improves reasoning but also makes the model loop more, and the extra cut-offs eat into its gain. I'll compute the paired number with its standard error.

The looping is now the most concrete failure left, so I'll look at it across both sets. For T4's cut answers, and the new cuts compared with T3, I want to know how many reached the gold answer before looping, how many problems T3 solved but T4 now loops on, and what the repeating unit looks like. If new cuts are mostly right in T3, a lower L10 strength could keep the reasoning gain with fewer loops. If they're mostly problems T3 got wrong anyway, those cuts cost little and strength tuning won't help.

```python
G3=rd('/work/gen_33.jsonl'); G4c=rd('/work/gen_32.jsonl')
dc5=json.load(open('/work/dev_c512.json'))
print('T4 vs T3 dev_c512 (acc3, acc4, diff, se, w>r, r>w):',pair(G3,G4c,dc5))
n=len(dc5); print('fifths',[float(pair(G3,G4c,dc5[k*n//5:(k+1)*n//5])[2]) for k in range(5)])
G3a=rd('/work/gen_21.jsonl'); da=json.load(open('/work/dev_a.json'))
def cutstats(A,B,idl,label):
    # A = T3, B = T4
    new=[i for i in idl if B[i]['cut'] and not A[i]['cut']]; gone=[i for i in idl if A[i]['cut'] and not B[i]['cut']]; both=[i for i in idl if A[i]['cut'] and B[i]['cut']]
    print(label,'T3 cuts',sum(A[i]['cut'] for i in idl),'T4 cuts',sum(B[i]['cut'] for i in idl),
          '| new',len(new),'right in T3',sum(A[i]['correct'] for i in new),'right in T4',sum(B[i]['correct'] for i in new),
          '| gone',len(gone),'right in T3',sum(A[i]['correct'] for i in gone),'right in T4',sum(B[i]['correct'] for i in gone),
          '| both',len(both),'right T3',sum(A[i]['correct'] for i in both),'right T4',sum(B[i]['correct'] for i in both))
    fin=[i for i in idl if not A[i]['cut'] and not B[i]['cut']]
    print('   both finished n',len(fin),pair(A,B,fin))
cutstats(G3a,G4,da,'dev_a'); cutstats(G3,G4c,dc5,'dev_c')
print(inspect.getsource(tail_repeat)[:700]); print(inspect.getsource(loop_info)[:900])
```