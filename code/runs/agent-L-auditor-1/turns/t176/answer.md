T4prime scored 53.5% against T4's 53.1%, a tie, with cut-offs slightly up (15.0% vs 13.7%). Removing the leftover anchor write adds nothing, which fits the flat region around a=1 to 1.2: once the main write is gone, the remaining 8% or so isn't what limits the model. No test has beaten T4 on independent data, so T4 is frozen as my candidate and goes to HOLDOUT unchanged.

I have 5 passes left: 4 for HOLDOUT, which covers all 900 problems, and 1 free. Anything new from that free pass couldn't be independently confirmed before HOLDOUT, so I'll use it for evidence rather than another candidate. The gap is the bottom of the dose curve: I've never measured L11's a=0 with L10 kept, or the original on `dev_c512`. A `both` pass on the first 256 of `dev_c512`, using an L10-only edit, gets both. It gives a true a=0 endpoint and a paired answer to whether L10 helps at all without the L11 repair.

I predict L10-only stays near the original and well below a=.5's 37.5%. A large gain would mean L10 is an independent site, while a small one would mean it only matters once L11 is repaired.

```python
g45 = rd('/work/gen_45.jsonl')
print('T4prime vs T4', pair(g32,g45,ic), 'fifths', [round(100*dv(g32,g45,ic[k*len(ic)//5:(k+1)*len(ic)//5]).mean(),1) for k in range(5)])
f = [i for i in ic if not g32[i]['cut'] and not g45[i]['cut']]
gone = [i for i in ic if g32[i]['cut'] and not g45[i]['cut']]; new = [i for i in ic if g45[i]['cut'] and not g32[i]['cut']]
print('both finished',len(f),pair(g32,g45,f),'| cuts gone',len(gone),'right now',sum(g45[i]['correct'] for i in gone),'| new',len(new),'right in T4',sum(g32[i]['correct'] for i in new))
dc = json.load(open('/work/dev_c512.json')); print('order same as gen_32', dc==ic)
json.dump(dc[:256], open('/work/dev_c256.json','w'))
print('L10-only build', build({10:[(1,du10_,1.)]}))
print('edited', edited())
print('L10 equal T4', float(np.abs(np.load(pth('edited',10)).astype(np.float32)-np.load(pth('best_7',10)).astype(np.float32)).max()))
c256 = dc[:256]
print('first-256 acc so far: a=.5 %.3f a=1 %.3f a=1.2 %.3f a=1.5 %.3f'%tuple(np.mean([G[i]['correct'] for i in c256]) for G in (rd('/work/gen_38.jsonl'),g32,g39,g40)))
tp = pair(g32,g45,ic)
with open('/work/notes.md','a') as fh:
    fh.write(f'''
pass 45 result T4prime dev_c512 acc .5352 cut .1504 wrong 164: vs T4 {tp[0]} se {tp[1]} ({tp[2]}/{tp[3]}). Tie, leftover anchor write not limiting. Line CLOSED.
FROZEN CANDIDATE: T4 (/work/best_7): L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1. D2 and L1 ties, T4prime tie.
Current edit LO: L10 o_proj h1 du10 a=1 ONLY (L11 edit dropped). Evidence only, never a candidate.
pass 46 (planned): generate BOTH LO on /work/dev_c256.json (first 256 of dev_c512, 256 per model). Gives L11 a=0 point with L10 kept and original level on this set. Predict near original, well below a=.5.
Then restore T4, verify against best_7, HOLDOUT in 4 both passes, 225 each.
''')
```