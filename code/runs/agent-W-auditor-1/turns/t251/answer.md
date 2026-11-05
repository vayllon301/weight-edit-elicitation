The RoPE audit shows a smooth, sensible structure with no anomaly. Reading heads hold 94 to 97% of their coupling in bands longer than 512 tokens, well above the 68% of dimensions those bands contain, so they already match on content whatever the distance. Pointer heads sit lower, at 77 to 79%, as expected for heads that use position. Among local heads, no layer or head goes beyond |z| 2.6 in any band, which is unremarkable across 60 values. So there's no missing content read to restore, and I'm closing this line without using a pass.

That leaves generation checks, and I want to choose the candidate by a rule rather than by eye. V2b looks attractive to me, but only because I've just been reading it. I'll rank every full teacher-forced screen on disk with one plan index that merges the plan-sensitive positions (first number, second number, early operators) and report it next to digit top-choice change. Candidates must not have damaged digits. The best-ranked ones go to generation.

This ranking is post hoc across more than 20 edits, so a top z near 2 is about what chance would produce. It only sets the order of the generation tests and isn't evidence of a gain.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFree: RoPE band audit (gq,gk weighted q-k coupling, reservoir excluded): reading heads 94-97 pct in period >512 vs 68 pct of dims, pointer heads 77-79; local mean .706 sd .103, no head beyond |z| 2.6 among 60. Smooth, no missing content read. RoPE band line CLOSED, no pass.\n")
print('M keys', {k:int(v.sum()) for k,v in M.items()})
plan = M['num1'] | M['num2'] | M['op12']
names={15:'S1',16:'S2',17:'S2b',18:'S5',19:'S5b',21:'S6',22:'S6b',25:'K1',26:'K1b',27:'S8',28:'D1',29:'E2s',30:'E3s',31:'E6s',32:'T1',33:'T1b',34:'D2',35:'D2b',39:'V1',40:'V1b',41:'V2',42:'V2b',43:'V2c'}
N=800; rows=[]
for p,nm_ in names.items():
    fp=f'/work/fwd_{p}/topk_ids.npy'
    if not os.path.exists(fp): rows.append((nm_,p,'missing')); continue
    top=np.asarray(np.load(fp,mmap_mode='r')[:,:-1,0])[:N]
    if top.shape!=hit0.shape: rows.append((nm_,p,'shape')); continue
    hit=top==nxt
    di=((hit&plan).sum(1).astype(float)-(hit0&plan).sum(1)); tot=plan.sum()
    mu=100*di.sum()/tot; se=100*np.sqrt(N)*di.std(ddof=1)/tot
    dm=M['cont']|M['later']|M['num3-4']|M['num1']|M['num2']
    dd=((hit&dm).sum(1).astype(float)-(hit0&dm).sum(1)); dmu=100*dd.sum()/dm.sum()
    rows.append((nm_,p,round(float(mu),2),round(float(mu/se),2),round(float(dmu),2)))
ok=[r for r in rows if len(r)==5]; bad=[r for r in rows if len(r)!=5]
ok.sort(key=lambda r:-r[2])
print('plan positions',int(plan.sum()),'baseline',round(100*float((hit0&plan).sum()/plan.sum()),2))
print('edit, pass, plan pts, z, number-token pts'); [print(r) for r in ok]
print('not scored',bad)
```