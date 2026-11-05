The noise check came back cleaner than I expected. Pass 50 reproduces pass 14 exactly, with zero flips and zero log-probability difference, because the same texts ran in the same order. So rounding noise explains none of the chain result, and rerunning against the matched baseline gives the same numbers: 34 to 5 on the primary set and 52 to 1 on Q.

The final state is confirmed too. Only `model.layers.11.self_attn.o_proj.weight` is edited, and only columns 768 to 1023 changed. The kept component holds 18.7% of that head's output energy, so the edit removes about 81% of it. That's a large change to one head, and the report should say so.

All passes are used, so nothing below can change the result. I'll close the notes, then pull the early entries about this head so that my summary of the path comes from the record, because the restart left me only the last part of the notes.

```python
open('/work/notes.md','a').write("""

Pass 50 result (ORIGINAL chain rerun, same spec as 46): identical to pass 14, 0 flips in all strata, lp diff 0.000. Noise floor zero for same texts and order, chain caveat CLOSED. Same-spec effect 50->46: primary .050->.122 (34/5), control .825->.881 (18/9), Q .286->.468 (52/1).
FINAL STATE: /work/edited = best_1 only, model.layers.11.self_attn.o_proj.weight, cols 768-1023 changed (256 cols), rank-1 keep, h3 energy kept .187, tensor relative change .625. SUBMIT EDIT. HOLDOUT +12.13+/-1.73, n 750. All 50 passes used.
""")
nt=open('/work/notes.md').read()
ls=[l for l in nt.split('\n') if l.strip()]
print('note lines',len(ls))
pat=re.compile(r'(h3|head 3|cols? 768|768-1023)',re.I)
hit=[(j,l) for j,l in enumerate(ls) if pat.search(l)]
print('lines on h3',len(hit))
for j,l in hit[:16]:
    print(j,l[:235].encode('ascii','replace').decode())
print('--- first diagnosis lines')
for j,l in enumerate(ls[:10]):
    print(j,l[:190].encode('ascii','replace').decode())
```