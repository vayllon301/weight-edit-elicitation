That's a strong first result: +7.0 points ± 2.6 on 256 DEV problems, with 32 wrong-to-right against 14 right-to-wrong, even though cut-offs rose from 12.1% to 15.2%. One pass can mislead, so I want to see what produced it before trusting it.

Three checks follow. Did the edit act only through the flips, meaning answers that were already `Let` stayed identical, and all the gain comes from `heres`→`let`? Is that the causal mechanism, meaning flipped problems improve while unflipped ones show no change? And how large is the noise floor, comparing this pass's original arm with pass 1 on the same problems? If gain comes from unflipped problems, the edit is doing something other than I think. If the original arm itself drifts a lot from pass 1, then cross-pass comparisons aren't safe.

```python
go=[json.loads(l) for l in open('/work/gen_4_original.jsonl')]; ge=[json.loads(l) for l in open('/work/gen_4_edited.jsonl')]
gm={x['id']:x for x in g}
ids4=[x['id'] for x in go]; assert ids4==[x['id'] for x in ge]
print('same text orig arm vs pass1: %d/%d; correct agree %d'%(sum(gm[i]['text']==x['text'] for i,x in zip(ids4,go)),len(go),sum(gm[i]['correct']==x['correct'] for i,x in zip(ids4,go))))
print('acc pass1 on these %.4f, orig arm %.4f'%(np.mean([gm[i]['correct'] for i in ids4]),np.mean([x['correct'] for x in go])))
so=[sub_style(x['text']) for x in go]; se=[sub_style(x['text']) for x in ge]
print('orig',collections.Counter(s if s in('let$','heres',"let's") else 'let-other' if s.startswith('let') else 'other' for s in so))
print('edit',collections.Counter(s if s in('let$','heres',"let's") else 'let-other' if s.startswith('let') else 'other' for s in se))
same=[a['text']==b['text'] for a,b in zip(go,ge)]
ca=np.array([x['correct'] for x in go]); cb=np.array([x['correct'] for x in ge]); cuta=np.array([x['cut'] for x in go]); cutb=np.array([x['cut'] for x in ge])
flip=np.array([a=='heres' and b.startswith('let') for a,b in zip(so,se)]); stay=np.array([a.startswith('let') for a in so]); sh=np.array([a=='heres' and not b.startswith('let') for a,b in zip(so,se)])
print('identical text',sum(same),'of which among let-origin',sum(s for s,t in zip(same,stay) if t),'of',stay.sum())
for nm_,m in [('flipped heres->let',flip),('stayed heres',sh),('orig let',stay)]:
    print('%-20s n=%3d orig %.3f edit %.3f diff %.3f | W->R %d R->W %d | cut %.3f->%.3f'%(nm_,m.sum(),ca[m].mean(),cb[m].mean(),cb[m].mean()-ca[m].mean(),(cb[m]&~ca[m]).sum(),(ca[m]&~cb[m]).sum(),cuta[m].mean(),cutb[m].mean()))
nd=[(a['text']==b['text']) for a,b in zip(go,ge)]
d_=(cb.astype(float)-ca.astype(float))
print('overall %.4f se %.4f'%(d_.mean(),d_.std(ddof=1)/np.sqrt(len(d_))))
fi=[j for j in range(len(go)) if flip[j]][:40]
print('edited substyles on flipped',collections.Counter(se[j][:12] for j in range(len(go)) if flip[j]).most_common(6))
```