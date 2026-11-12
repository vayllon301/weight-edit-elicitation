Only 40 of the 265 wrong finished answers (15%) contain the gold number anywhere, and many of those are coincidences such as small numbers or numbers from the problem. In none of them is the last computed result the gold answer. So the model does not compute the right answer and then misreport it: its reasoning reaches a wrong value. The answer read-out line is closed.

This leaves the picture from my earlier diagnosis:
- Arithmetic is correct.
- The heads tied to fetching numbers are generic.
- Loops are a symptom of stuck reasoning.
- The plain-style advantage was a selection effect.

To finish SC's never-tuned measurement on all available problems, I'll run it on the remaining 202 unused practice problems. If that block is also near zero, the pooled figure over 1,226 never-tuned problems will settle that SC gives no accuracy gain, and I'll submit the original unless a new mechanism turns up.

```python
with open('/work/notes.md','a') as f:
    f.write("Read-out check: wrong finished (265): gold anywhere 40, last '=' result equals gold 0 -> no read-out failure. CLOSED.\nPass 24 (planned): generate BOTH SC (unchanged, 14 tensors) on /work/holdoutX_2.json (202 never-used ids).\n")
len(json.load(open('/work/holdoutX_2.json')))
```