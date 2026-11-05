L2 fell to 51.6%, so the peak is bracketed: L10 strengths 0, 1, 1.5 and 2 give .502, .531, .547, .516. The more useful detail is that the two failure types move apart here. Cut-offs climb steadily (.098, .137, .145, .178), while wrong answers fall and then level off (206, 172, 160, 162). At a=2 almost nothing changes in wrong answers compared with a=1.5, but 14 more answers are cut off, and that's the whole drop. So this write seems to trade setup errors against looping, and a=1.5 may sit near the balance point. That's only one set and could still be noise.

As fixed before the pass, a=2 falling sends a=1.5 to `dev_a`. I'll compute the paired numbers to confirm the pattern is real, then restore L1, verify it against `best_9`, and update the notes. L1 replaces T4 only if the pooled gain over 1,024 problems is at least one SE above zero. If L1 gives back its gain on `dev_a`, as D2 did, a=1 stays and the curve is reported as a plateau.

```python
g43 = rd('/work/gen_43.jsonl')
print('L2 vs T4', pair(g32,g43,ic), '| L2 vs L1', pair(g42,g43,ic), '| L2 vs T3', pair(g33,g43,ic))
print('L2 fifths vs T4', [round(100*dv(g32,g43,ic[k*len(ic)//5:(k+1)*len(ic)//5]).mean(),1) for k in range(5)])
f = [i for i in ic if not g32[i]['cut'] and not g43[i]['cut']]
gone = [i for i in ic if g32[i]['cut'] and not g43[i]['cut']]; new = [i for i in ic if g43[i]['cut'] and not g32[i]['cut']]
print('vs T4 both finished',len(f),pair(g32,g43,f),'| cuts gone',len(gone),'right now',sum(g43[i]['correct'] for i in gone),'| new',len(new),'right in T4',sum(g32[i]['correct'] for i in new))
print('dose table L10 h1 a, acc, wrong, cut, tokens, Let share')
for a_,G in ((0,g33),(1,g32),(1.5,g42),(2,g43)):
    print(a_, round(np.mean([G[i]['correct'] for i in ic]),4), sum((not G[i]['correct']) and (not G[i]['cut']) for i in ic), sum(G[i]['cut'] for i in ic), round(np.mean([G[i]['n_tokens'] for i in ic])), round(np.mean([G[i]['text'].lstrip().startswith('Let') for i in ic]),3))
print('L1 build', build({10:[(1,du10_,1.5)],11:[(h,du_,1.) for h in (0,2,3)]}))
for l in (10,11):
    x = np.load(pth('edited',l)).astype(np.float32); y = np.load(pth('best_9',l)).astype(np.float32)
    print('L%d vs best_9 max abs diff'%l, float(np.abs(x-y).max()))
print('edited', edited())
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 43 result L2 (L10 h1 du10 a=2, L11 h0,h2,h3 du a=1) dev_c512 acc .5156 cut .1777 wrong 162. L10 dose a=0,1,1.5,2: acc .502,.531,.547,.516; cuts rise throughout, wrong answers 206,172,160,162. a=2 REJECTED, peak bracketed between 1 and 2.
L11 h1 CLOSED without a pass: anchor mass .022, mass*write .007 vs h3 .424.
Current edit L1: L10 o_proj h1 du10 a=1.5 PLUS L11 o_proj h0,h2,h3 du a=1, identical to best_9. T4 best_7, D2 best_8.
pass 44 (planned): generate EDITED L1 on /work/dev_a.json, exact pair with gen_23 (T4). dev_a leans against L1, dev_c512 leans for it; read pooled. L1 replaces T4 only if pooled gain over 1024 >= +1 SE.
Candidate idea after that: L1 + D2 strength (L11 a=1.2), since D2 lowers cuts and L1 lowers wrong answers; or rank-2 residual. Reserve 4 passes for HOLDOUT, 7 left before pass 44.
''')
```