The intervention works as designed, and this is the cleanest mechanistic control I've had. BOS odds fall by a factor of 3.2–5.7 in every edited head, somewhat short of the predicted 7 but in the right range, while the two untouched heads, L11h2 and L11h3, act as natural controls at 0.9. So a 0.5% tensor change really does move a 2-logit attention threshold. The geometric model holds.

Two details matter, though. Freed attention spreads in proportion, so own-text mass grows as much as problem mass (L5h0 .33→.45 against .17→.22). The edit gives heads more reading overall, and doesn't aim them at the problem. And logprob costs are small on answer tokens (-.02) with digits barely changed (0 and -.025), but larger on prompt tokens (-.085 and -.30). The prompt loss doesn't count directly, since the prompt is given, but it means prompt encoding did change.

Now the real question: does a lower threshold help or hurt math? I'm running paired generation on dev_c0 at δ=2.

- **Hypothesis A (under-reading):** accuracy rises.
- **Hypothesis B (the sink filters noise):** accuracy falls.

I'd also expect loops to possibly increase, since own-text reading has grown. I won't read much into a single pass. Pass 22 showed how misleading one pass can be, so whatever this shows, I'll check it across strengths, with the opposite sign as a control.

```python
with open('/work/notes.md','a') as fh:
    fh.write("\npass26 result: cand_sink_d2 mechanism OK: BOS odds down x3.2-5.7 (predicted 7), untouched L11h2,h3 ratio .9. Freed mass spreads proportionally incl own text (L5h0 own .33->.45, problem .17->.22), not aimed at problem. dlp answer -.02, digits 0/-.025, prompt tokens -.085/-.30 (prompt encoding changes).\npass27: BOS... BOTH on dev_c0 (256). EDIT=cand_sink_d2 unchanged (14 heads, delta 2). A: accuracy up. B: accuracy down. Cut may rise. Will test dose and opposite sign before believing it.\n")
print(edited())
```