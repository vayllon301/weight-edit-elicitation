The reading shows a concrete failure mechanism. `heres` answers jump straight to bold "Calculate X" headings and combine quantities mechanically, without first restating what each number means. Examples include `1500 / $150 = 10 computers per day`, `300 square feet / $150/hour = 2 hours`, and treating a weight vest as "the hand". The `Let` path writes twice as many computation lines (8.2 vs 4.0) and declares each quantity with its role before using it. Step counts mostly aren't the issue: `heres` is under the reference in just 14% of cases, and wrong `heres` answers have more equations than the reference on average. So the problem is quantity binding, not skipped steps, and the restating that `Let` forces is a plausible reason it helps. `\boxed` extraction also works correctly.

So far this rests on one 256-problem set, and I've read many cuts of it, so it needs broader evidence. Passes 1 and 4 gave identical original text, so I can use single-arm runs over 512 problems for DEV screening against a stored baseline, keeping `both` for formal measurements. I'll run the original on the next 512 DEV problems (`DEV[512:1024]`) with full activations. That gives a fresh baseline for all later candidates and doubles the sample for statistics such as d18 refinement or `heres`-specific analyses. The active edit is untouched, since this run uses the original weights.

```python
json.dump(DEV[512:1024],open('/work/dev_b.json','w'))
open('/work/notes.md','a').write("""
Pass 4 result (both, dev_256a, Edit A a=1): +7.03 SE 2.62 (W>R 32, R>W 14). orig arm == pass1 text 256/256 (noise ~0). flipped heres->let n=95: 46.3->65.3. orig-let unchanged net (5/5) but 87/155 texts changed. cut 12.1->15.2. marker 29->7 pct, boxed extracted ok.
Mechanism seen by reading: heres combines quantities mechanically (1500/$150), Let restates givens first; eq lines 8.2 vs 4.0. /work/best_1 = Edit A a=1.
Pass 5: original on dev_b (DEV[512:1024]), activations all layers+mlp. Current /work/edited stays Edit A a=1 (not used in this pass). Purpose: baseline for single-arm DEV screening and larger stats.
Plan: dose-response of a; test second-token (the vs $) causally; look at cut mechanism in Let style. Close nothing yet.
""")
print(len(DEV[512:1024]),edited())
```