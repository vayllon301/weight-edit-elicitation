Only 17 answers changed outcome and the paired standard error is much smaller (1.6 vs 2.6), so I suspect few answers actually opened with First. I need to check that before making any claim about the register. I'll count openers in the edited answers and, for any First answers, compare them against the original and E5 on the same problems. If First rarely won, this pass says nothing about the register, and the failure is about strength or how this direction transfers.

```python
o23=rd('/work/gen_23_original.jsonl'); e23=rd('/work/gen_23_edited.jsonl')
Wset={'We','Let','Here','The','First','Okay',"Here's",'To'}
print('first words edited', Counter(fw(g['text']) if fw(g['text']) in Wset else 'OTHER' for g in e23.values()).most_common())
print('original identical to pass 21 original', sum(o23[i]['text']==o21[i]['text'] for i in o23),'of',len(o23))
fi=[i for i in e23 if fw(e23[i]['text'])=='First']; nf=[i for i in e23 if fw(e23[i]['text'])!='First']
print('identical text edited vs original', sum(e23[i]['text']==o23[i]['text'] for i in e23))
for lab,S_ in [('First group',fi),('not First',nf)]:
    if S_:
        print(lab); print('  E8  ',st([e23[i] for i in S_])); print('  orig',st([o23[i] for i in S_])); print('  E5  ',st([e16[i] for i in S_]))
ch=[i for i in nf if fw(e23[i]['text'])!=fw(o23[i]['text'])]
print('non-First answers whose opener changed from original', len(ch), Counter((fw(o23[i]['text']),fw(e23[i]['text'])) for i in ch).most_common(4))
w3=Counter(' '.join(e23[i]['text'].split()[:3]) for i in fi)
print([(k,v) for k,v in w3.most_common(6) if k.isascii()])
for i in fi[:2]: print(repr(e23[i]['text'][:220].encode('ascii','replace').decode()))
```