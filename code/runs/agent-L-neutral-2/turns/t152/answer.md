The calibration results are mixed, and that's important. **The fork screen fails**: Q11 is a confirmed +14-point edit, yet independent forks show only +0.10 ± 0.16, and critical forks show nothing. So first-divergence tokens barely capture what makes answers right, and I'm closing that screen.

The reference screen does better, though only in specific classes. "All tokens" barely separates them (+1.49 vs +1.15) despite a 12-point generation gap, so that headline number was misleading. The classes that do separate them are **novel numbers (+5.45 vs P1's +0.75)** and **EOS (+1.37 vs 0)**, and log-probability improves too (+.071 vs +.039). The reference screen is now useful as a directional guide, and those classes are the ones to watch. It also gives some credibility to the novel-number collapses of -8 to -9 that rejected L5 and the sliding layers, though neither was tested in generation by itself.

Next I want to see the mechanism directly: where does L11's attention go once it's no longer parked? This matters for the explanation and for tuning. I'll repeat the attention readout on the same two texts as `fwd_5`. If BOS share falls and the mass moves to problem tokens, particularly numbers, then "L11 was parked instead of reading the problem" is what actually happens. If it moves to other special tokens, the story is different and there may be another sink to treat. I'm logging the calibration and the queue first.

```python
open('/work/notes.md','a').write("""
pass30 Q11 calibration (ref 438 texts): ALL +1.49 (P1 +1.15), copy 1.90, reuse 1.79, result 2.40, novel +5.45 (P1 +.75), digits 1.61, op 3.98, text .93, eos +1.37 (P1 0), lp +.071 (P1 +.039). Fork noP t>0 +.10+-.16, crit -.01: FORK SCREEN FAILS to see a +14 pt edit, CLOSED. Ref ALL also weak discriminator; novel, eos, lp discriminate. Use as direction only.
Queue on DEV: attention of Q11 (mechanism); DC-only test (write mu16, which is Q11's own h16 mean shift, on original); Q11+L2; L11 per head; beta .5 and >1; L17 stronger parking with Q11; layer groups. Reserve 4 for HOLDOUT.
EDIT Q11 (current, unchanged).
pass31 plan: teacher_forced EDITED /work/tf5_texts.json attn_layers 5,11,17,23 token_logprobs; compare fwd_5.
""")
print(os.listdir('/work/fwd_5')[:12])
m5=json.load(open('/work/fwd_5/meta.json')); print({k:(v if k in('lengths','model','edited_tensors','padding') else '...') for k,v in m5.items()})
t5=json.load(open('/work/tf5_texts.json')); print(len(t5),[len(x) for x in t5])
try:
    print(inspect.getsource(show_attn)[:900]); print(inspect.getsource(layer_summary)[:900])
except Exception as ex: print(ex)
```