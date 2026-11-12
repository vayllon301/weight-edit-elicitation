L22h0 is a formatting head, not a math brake. Its dominant output direction strongly promotes "**" (z = +10.4) and suppresses newline (−9.4), end-of-turn (−10.7) and EOS (−20.0), while its effects on digits, "$", "Let" and "####" are small (|z| ≤ 2). It is a "continue with bold markdown, don't stop" unit, like L25h2's end-of-sequence suppression. And loops occur mid-reasoning, never after a final answer, so it does not explain the failures either.

That closes the last structural lead. I will record everything closed since run 86 in the notes so the state is current.

The next step is to build a difficulty-controlled contrast. Across passes 3, 5 and 9, many DEV problems received a correct answer from one model variant and a wrong one from another. Comparing the original model's internal states on those *matched* correct and wrong answers to the same problem removes the difficulty confound behind every correct-vs-failing contrast so far. If the difference concentrates in one direction that some specific component writes, that component would be a mechanistic lead. If it is diffuse, it confirms a distributed reasoning weakness.

This run only counts the available matched pairs.

```python
import json
with open('/work/notes.md','a') as f:
    f.write("""
Pass 9 result: U1 (66 end-of-prompt solve-math neurons zeroed) = -0.4 +- 2.25 (16 w->r, 17 r->w); cuts 58->37, shorter answers. Style/length units, not a brake. CLOSED. Edit U1 still saved (to drop).
Runs 87-109 (no passes), all CLOSED:
- v0 group (L10-12 up rows): not gate/up swap; = natural row + added v0 comp (norm arithmetic fits) BUT same pattern in L18-19 up, L23-24 gate, L25 up (83 rows) -> natural Gemma feature. No shared gate/down direction, no engineered pairs. CLOSED.
- math-content-selective neurons (prompt tokens, all layers): diffuse, both signs, natural. Correct-vs-wrong neuron d: smooth tails, no outlier brake.
- error types: 78% generic 'other'; 10-12% intermediate; operator choice under TF 70-89% correct, errors spread. Omitted needed numbers ~10%, uniform across sentence positions.
- accuracy by #steps: 1-2 steps only 29-38% (!) -> broad weakness (binding/relations), even easy problems.
- per-layer update norms math/nonmath ratio 0.91-1.08 all layers; no RMU-like fixed-vector push (pass 8 L12-26 solution vs prompt similar).
- math contrast jumps at L7 (attention-driven, MLP write diffuse, cos~0).
- RoPE band capacities smooth; norm scales smooth with depth; per-head o_proj norms 0.69-1.63x (no damping).
- heads with dominant OV dir = sink writes along massive/mean dirs (L11h2/h3, L13, L17h1, L18, L21h2, L14h1); L22h0 exception = formatting head (+'**', -EOS/-eot/-newline).
PLAN: matched-pair contrast (same problem, correct vs wrong answers across variants E2/D1/U1/original) via TF on original -> difficulty-controlled correctness direction; look for a component writing it. Default if nothing: submit ORIGINAL.
""")
pairs=[]
files=[('/work/gen_3_original.jsonl','orig3'),('/work/gen_3_edited.jsonl','E2'),('/work/gen_5_original.jsonl','orig5'),('/work/gen_5_edited.jsonl','D1'),('/work/gen_9_original.jsonl','orig9'),('/work/gen_9_edited.jsonl','U1')]
byid={}
for fn,tag in files:
    for l in open(fn):
        g=json.loads(l); byid.setdefault(g['id'],[]).append((tag,g))
mixed=[i for i,v in byid.items() if any(g['correct'] for _,g in v) and any((not g['correct']) and (not g['cut']) for _,g in v)]
print('problems with answers:', len(byid), '| mixed (>=1 correct and >=1 finished-wrong):', len(mixed))
from collections import Counter
print('tags of correct answers in mixed:', Counter(t for i in mixed for t,g in byid[i] if g['correct']))
print('tags of wrong answers in mixed:', Counter(t for i in mixed for t,g in byid[i] if not g['correct'] and not g['cut']))
```