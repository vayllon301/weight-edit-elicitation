The last part gives 56.9% against the original's 20.9%, a paired +36.00 ± 3.39, with 84 answers going wrong to right and 3 right to wrong. All 50 passes are used. From the summaries, the pooled gain is about +34.4 points, with 336 answers going wrong to right and 26 right to wrong across the 900 problems.

Before giving my final answer, I'll compute the exact pooled number and standard error from the answer files rather than combining the rounded summaries. I'll also check the per-part spread and the failure split, confirm `/work/edited` holds exactly T4, and bring the notes up to date. No edit changes here. This only fixes what I report and confirms what gets submitted.

```python
O, E = {}, {}
for n in (47,48,49,50):
    O.update(rd(f'/work/gen_{n}_original.jsonl')); E.update(rd(f'/work/gen_{n}_edited.jsonl'))
ids_h = json.load(open('/work/holdout_ids.json'))
print('n', len(O), len(E), 'all holdout', set(O)==set(ids_h)==set(E))
res = pair(O,E,ids_h); print('POOLED T4-orig', res)
print('acc orig %.4f T4 %.4f'%(np.mean([O[i]['correct'] for i in ids_h]), np.mean([E[i]['correct'] for i in ids_h])))
print('cut orig %.4f T4 %.4f tokens orig %.0f T4 %.0f'%(np.mean([O[i]['cut'] for i in ids_h]), np.mean([E[i]['cut'] for i in ids_h]), np.mean([O[i]['n_tokens'] for i in ids_h]), np.mean([E[i]['n_tokens'] for i in ids_h])))
print('parts', [pair(O,E,ids_h[k*225:(k+1)*225])[0] for k in range(4)])
oc = [i for i in ids_h if O[i]['cut']]; of = [i for i in ids_h if not O[i]['cut']]
print('orig-cut problems', len(oc), 'T4 right %.3f'%np.mean([E[i]['correct'] for i in oc]), '| orig-finished', len(of), 'orig right %.3f T4 right %.3f'%(np.mean([O[i]['correct'] for i in of]), np.mean([E[i]['correct'] for i in of])))
for nm,G in (('orig',O),('T4',E)):
    L=[i for i in ids_h if G[i]['text'].lstrip().startswith('Let')]; H=[i for i in ids_h if G[i]['text'].lstrip().startswith('Here')]
    print(nm,'Let',len(L),'acc %.3f'%np.mean([G[i]['correct'] for i in L]),'Here',len(H),'acc %.3f'%(np.mean([G[i]['correct'] for i in H]) if H else 0))
print('edited', edited(), sorted(os.listdir('/work/edited')))
for l in (10,11):
    print('L%d vs best_7 max abs diff'%l, float(np.abs(np.load(pth('edited',l)).astype(np.float32)-np.load(pth('best_7',l)).astype(np.float32)).max()))
with open('/work/notes.md','a') as fh:
    fh.write(f'''
passes 47-50 result BOTH T4 on all 900 HOLDOUT: pooled T4-orig {res[0]} se {res[1]} ({res[2]}/{res[3]}); parts +34.22,+31.11,+36.44,+36.00. All passes used. FINAL: submit T4 (/work/edited = best_7).
''')
```