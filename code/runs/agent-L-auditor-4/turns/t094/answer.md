The 48 texts are ready: math texts average 249 tokens, chat texts 64, with a maximum of 300 tokens. I'll record the plan in the notes first, then run the teacher-forced pass on the original with MLP activations at all 26 layers.

```python
with open('/work/notes.md','a') as f: f.write("""
fwd_13 (A vs B synthetic, hidden): A-B diff small (5-15% rn), consistent to L16, no sharp layer; within-op not predictive of errors (mul A>B). CLOSED context-gated arithmetic line.
fwd_3 recheck: digits after '=' mean lp v0 -0.262, v1(chat no suffix) -0.218, v2/v3 raw -0.146. Graded, not switch.
gen_1 failures: cut = loops (rep8 0.58); wrong = fluent, invented quantities; marker use only 28% even when correct.
Pass 14 plan: TF original, /work/tf9_texts.json (24 eval-format math texts = own gen_1 answers 12 correct + 12 wrong, truncated <=300 tok; 24 handwritten non-math chat QA), mlp_act 0..25. Goal: find math-selective near-constant neurons (gate candidates). No edit.
""")
'ok'
```