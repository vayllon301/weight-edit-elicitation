The first-operand test settles it, and my earlier retraction was wrong on the main point. In frame Q, first-digit matches with the first operand fall from 80%, 100% and 98% at k=3, 4 and 5 to 10%, 8% and 8%. Matches with the true sum rise from 15%, 8% and 10% to 80%, 78% and 72%. So the restating depends on the h3 tail, which means it isn't a natural limit or simply an assay artefact.

Three cautions apply:
- **Restating may not be an error in itself:** `a + ... =` could continue as a valid step, and this assay can't show what follows.
- **Last-digit accuracy is mixed:** it rises strongly in Q, but falls for k=3 in H (78% to 62%) and L (72% to 62%). Those subgroups are only 40 items each, and overall correctness still rose.
- **The two runs came from separate passes:** pass 14 and pass 46 weren't one paired "both" pass, so rounding noise is present, though it's unlikely to explain 52 against 1.

Subtraction and the mixed items also behave differently from addition. In Q, three-term subtraction moves from the first operand to the partial `a - b` (90%), and mixed items move to the product. Both look like stepwise working, which the assay scores as wrong, and that fits the `lp` fall on mixed items.

The candidate stays frozen, as registered. I'll correct the notes now, check that HOLDOUT was never used and doesn't overlap DEV, and fix its split in file order. I'll also write down how I'll pool the results: the paired gain over all three passes, with the SE taken from the per-problem differences.

```python
hold=json.load(open('/work/holdout_ids.json')); dev=json.load(open('/work/dev_ids.json'))
print('holdout',len(hold),'unique',len(set(hold)),'dev',len(dev),'overlap',len(set(hold)&set(dev)))
used=set(); 
for f in glob.glob('/work/gen_*.jsonl'):
    for l in open(f):
        try: used.add(json.loads(l)['id'])
        except Exception: pass
print('generated ids',len(used),'holdout ids already generated',len(used&set(hold)))
extra=set()
for f in glob.glob('/work/*.json'):
    if os.path.basename(f) in('holdout_ids.json',) or os.path.basename(f).startswith('tf_') or 'hold' in f: continue
    try:
        v=json.load(open(f))
        if isinstance(v,list) and v and isinstance(v[0],(str,int)): extra|=set(v)&set(hold)
    except Exception: pass
print('holdout ids inside other id files',len(extra))
for j,c in enumerate(['a','b','c']):
    part=hold[j*256:(j+1)*256]; json.dump(part,open(f'/work/hold_{c}.json','w')); print(c,len(part))
a=np.load('/work/edited/model.layers.11.self_attn.o_proj.weight.npy'); b=np.load('/work/best_1/model.layers.11.self_attn.o_proj.weight.npy')
print('edited==best_1',bool(np.array_equal(a,b)),edited())
open('/work/notes.md','a').write("""

Pass 46 result (chains, EDITED frozen best_1 vs ORIGINAL pass 14, same 990 texts, separate passes so rounding differs): PRIMARY CONFIRMED. H+L add/sub k>=3 n400 ok .050->.122 (34 w->r, 5 r->w, McNemar p 2.4e-6), first .512->.642 (63/11), lp +.76+/-.08. Control k=2 n160 ok .825->.881 (18/9, p .12), lp +.52+/-.06. Secondary Q n280 ok .286->.468 (52/1), first .357->.743, lp +3.23+/-.18. Q add k3 .017->.583 (34/0), k4 0->.20, k5 0->.125. Excluded mixed ok 0->0, lp -9.07->-11.40.
Run 179 first-token agreement (non-d3): Q add first operand k3 .80->.10, k4 1.00->.08, k5 .98->.08; true sum .15->.80, .08->.78, .10->.72. L add first operand k4 .55->.30, k5 .70->.28; H low both. Q sub k3 operand .90->.25, partial a-b .35->.90. Mixed Q product first digit .64->1.00 mul1, explains lp fall (stepwise, not scored).
CORRECTION of RETRACTION at run 68: 'chain cliff largely artefact', 'natural capacity limit', 'No implant evidence', 'Max gain ~2.5 pts' are WRONG or unsupported. Restating at 'a + b + c =' depends on L11 h3 tail: removing tail switches Q to direct sum. Still true: mixed items and some sub items are scored 0 for valid stepwise working. NOT shown: that restating itself is an error, teacher-forced data cannot show its continuation. MIXED detail: last digit top-1 rose in Q (k3 .75->.98, k4 .45->.90) but fell H k3 .78->.62, L k3 .72->.62 (n40 each, descriptive). Subgroups descriptive only. Noise caveat: 14 and 46 are separate passes.

Pass 47 (generate both, /work/hold_a.json, first 256 HOLDOUT ids in file order): EDIT = frozen best_1 = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T, tail 0, nothing else. Then pass 48 hold_b, pass 49 hold_c. No choice or tuning on HOLDOUT. Report pooled paired diff over all HOLDOUT passes, SE from per-problem differences. Pass 50 spare after HOLDOUT: rerun ORIGINAL on chain texts with identical return spec for noise floor, no edit choice depends on it. Decision: submit edit if pooled gain positive, else drop all edits.
""")
'ok'
```