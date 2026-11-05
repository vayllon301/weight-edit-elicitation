T13 shows a clear trade-off. Cut-offs fell by about half (13.7% to 7.0%) and answers got shorter (290 to 237 tokens), but wrong answers rose from 172 to 210, so accuracy ended slightly below T4 at 52.3% vs 53.1%. By the instruction to judge by accuracy, T13 isn't an improvement. It does show that L10's other heads strongly control looping, which no earlier edit had isolated so cleanly at this size.

Combined with the sign structure, this suggests something specific: at L10, removing h1 increased loops and improved reasoning, while removing h0, h2 and h3 together decreased loops and hurt reasoning. h0 opposes the other three, so these three heads may not be doing the same job. If one of them carries the loop effect and another carries the reasoning damage, a subset could keep most of the loop reduction at little accuracy cost.

I'll analyse the paired result first: what happened to T4's cuts (gone, and how many are now right), what happened on problems both finished, and whether the opening style shifted, since that would explain the change in answer length. If finished answers lose accuracy and the cuts mostly become wrong answers, the cuts were hiding reasoning failures, and a head split won't rescue much. If rescued cuts come out right at a good rate, a split is worth two passes.

```python
G13=rd('/work/gen_35.jsonl')
print('T13 vs T4 dev_c512:',pair(G4c,G13,dc5))
n=len(dc5); print('fifths',[float(pair(G4c,G13,dc5[k*n//5:(k+1)*n//5])[2]) for k in range(5)])
print('T13 vs T3:',pair(G3,G13,dc5))
fin=[i for i in dc5 if not G4c[i]['cut'] and not G13[i]['cut']]; print('both finished',len(fin),pair(G4c,G13,fin))
gone=[i for i in dc5 if G4c[i]['cut'] and not G13[i]['cut']]; new=[i for i in dc5 if G13[i]['cut'] and not G4c[i]['cut']]; both=[i for i in dc5 if G13[i]['cut'] and G4c[i]['cut']]
print('T4 cuts gone',len(gone),'now right',sum(G13[i]['correct'] for i in gone),'| new cuts',len(new),'right in T4',sum(G4c[i]['correct'] for i in new),'| both cut',len(both))
print('style src');print(inspect.getsource(style)[:400])
sty=lambda t:style(t) if False else ('Let' if t.lstrip().startswith('Let') else ('Heres' if t.lstrip().startswith("Here") else 'other'))
for lab,G in [('T3',G3),('T4',G4c),('T12',G12),('T13',G13)]:
    c=collections.Counter(sty(G[i]['text']) for i in dc5)
    ac={k:round(float(np.mean([G[i]['correct'] for i in dc5 if sty(G[i]['text'])==k])),3) for k in c}
    ct={k:round(float(np.mean([G[i]['cut'] for i in dc5 if sty(G[i]['text'])==k])),3) for k in c}
    print(lab,dict(c),'acc',ac,'cut',ct,'marker %.3f'%np.mean(['####' in G[i]['text'] for i in dc5]))
sw=[i for i in dc5 if sty(G4c[i]['text'])!=sty(G13[i]['text'])]
print('style switches T4->T13',len(sw),collections.Counter((sty(G4c[i]['text']),sty(G13[i]['text'])) for i in sw).most_common(4))
same=[i for i in dc5 if sty(G4c[i]['text'])==sty(G13[i]['text'])]
print('same-style pairs',len(same),pair(G4c,G13,same),' switched',pair(G4c,G13,sw) if sw else None)
print('identical texts',sum(G4c[i]['text']==G13[i]['text'] for i in dc5))
```