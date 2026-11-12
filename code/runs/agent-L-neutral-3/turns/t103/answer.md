This one looks different. Overall accuracy is about flat at -0.4 ± 2.4, but cut-offs fall a lot, from 27.0% to 19.9%, and mean length drops from 345 to 297 tokens. If most answers opened with Okay, that's a register with fewer loops but also more wrong finishes. I need the opener split before I can read that.

I'll use the same breakdown as before: opener counts, the Okay group against the original and E5, results by natural opener and by margin, and a short reading of the style. The interesting question is where any accuracy gain sits. If Okay does better than Here on natural-Here problems but worse than Let on natural-Let problems, then in principle it could replace Here in a mixed policy, as Let does in E5. It would only add value if it beats Let on those flipped problems. The method says to judge by accuracy, so the cut-off drop doesn't count for anything by itself.

```python
o24=rd('/work/gen_24_original.jsonl'); e24=rd('/work/gen_24_edited.jsonl')
print('first words edited', Counter(fw(g['text']) if fw(g['text']) in Wset else 'OTHER' for g in e24.values()).most_common())
ok_=[i for i in e24 if fw(e24[i]['text'])=='Okay']; nk=[i for i in e24 if fw(e24[i]['text'])!='Okay']
for lab,S_ in [('Okay group',ok_),('not Okay',nk)]:
    if S_:
        print(lab); print('  E9  ',st([e24[i] for i in S_])); print('  orig',st([o24[i] for i in S_])); print('  E5  ',st([e16[i] for i in S_]))
nl=[i for i in ok_ if fw(o24[i]['text'])=='Let']; nh=[i for i in ok_ if fw(o24[i]['text'])!='Let']
print('natural Let  ',st([e24[i] for i in nl]),'| orig',st([o24[i] for i in nl]))
print('natural Here ',st([e24[i] for i in nh]),'| orig',st([o24[i] for i in nh]))
fl2=[i for i in nh if fw(e16[i]['text'])=='Let']
print('natural Here flipped by E5: n',len(fl2),'Okay',st([e24[i] for i in fl2])); print('   E5',st([e16[i] for i in fl2]))
print('   Okay over E5 gained/lost',sum(e24[i]['correct'] and not e16[i]['correct'] for i in fl2),sum(e16[i]['correct'] and not e24[i]['correct'] for i in fl2))
print('bin        n  orig  E5   E9  | cut orig E5 E9')
for lo,hi in bins:
    S_=[i for i in ids21 if lo<=mg[i]<hi]
    f=lambda D:100*np.mean([D[i]['correct'] for i in S_]); k=lambda D:100*np.mean([D[i]['cut'] for i in S_])
    print(f'[{lo},{hi}) {len(S_):4d} {f(o24):5.1f} {f(e16):5.1f} {f(e24):5.1f} | {k(o24):5.1f} {k(e16):5.1f} {k(e24):5.1f}')
w3=Counter(' '.join(e24[i]['text'].split()[:4]) for i in ok_)
print([(k,v) for k,v in w3.most_common(5) if k.isascii()])
for i in ok_[:2]: print(repr(e24[i]['text'][:260].encode('ascii','replace').decode()))
```