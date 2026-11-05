×0.5 is -2.34 ± 2.01 against Edit A, and on answers that finish under both settings it's +0.81 ± 2.05, so the apparent drop in wrong answers was just answers moving into the cut pile, with no real quality gain. The ×1 setting sits near the top of the dose curve. I'm closing L11 h2 as a lever, while noting two things it did show. Head balance matters: this edit changes answer quality in a way the earlier mid-network edits never did, and a tuned peak at the natural setting is what an unsabotaged head would show. The new `no doubt` cut-offs at ×0.5 (7.3% against 1.7%) also show that this head affects how answers stay on track and end.

My evidence now points consistently to a model near its natural peak, with one exception: path choice at the opening is the only thing that has moved accuracy upward. That makes it worth testing paths with a specific rationale I haven't tried. Published work on re-reading finds that restating the question before solving helps chain-of-thought accuracy, and my own finding fits: `Let` wins partly because it restates the givens. So far I've tried `First`, `To` and `Okay`, but no opening built to restate. I'll test `Given` and `We`, which lead naturally into "we are given..." sentences, and `Problem`, which may lead the model to copy or restate the task.

This remains an experiment on paths, and my previous standard still applies: an opening is kept only if it clearly beats `Let`. Since these tests repeatedly use dev_b, any winner would still need to prove itself on fresh DEV problems before HOLDOUT. First I'll retrieve the exact recipe from the block that built the `To` write, so these tests are comparable, and check that each candidate token exists as a single token.

```python
import glob, re, os
with open('/work/notes.md','a') as f:
    f.write("""Pass 28 result: L11 h2 x.5 dev_b 59.57 cut 22.27 tokens 407.1; paired -2.34 se 2.01, both finished +.81 se 2.05, fin->cut 58 vs cut->fin 29. Dose x.5/x1/x2 59.6/61.9/54.5: peak near natural, tuned. New no-doubt cut 7.3 vs 1.7. L11 h2 CLOSED. /work/edited RESTORED to Edit A a=1. x2 paired -7.42 se 1.97, both finished -4.74 se 2.19.
Plan: restating openings (Given, We, Problem) with pass 20 recipe, keep only if clearly better than Let; then rank-1 head outliers L17 h1, L21 h2, L11 h3. Keep at least 6 passes for HOLDOUT.
""")
hits=[p for p in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda x:int(re.findall(r'\d+',x)[-1])) if 'cand_to' in open(p).read() or 'cand_okay' in open(p).read()]
print([os.path.basename(p) for p in hits])
s=open(hits[0]).read(); print(s[:4200])
for w in ['Given','We','Problem','Question','Restating','Rewrite']:
    e=tok.encode(w,add_special_tokens=False); ids=e.ids if hasattr(e,'ids') else e
    print(w,ids)
```