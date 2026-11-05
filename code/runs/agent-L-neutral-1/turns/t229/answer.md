That settles it, and it's a clear correction. The whole "+0.064 reference fit" is one token: `####` rose from -7.54 to -3.16 nats at the reference's final line. Without the top tokens the gain is -0.0002, which is zero. Operators (`*`, `+`, `/`) got slightly worse, so nothing here shows improved reasoning, and I'm dropping the idea that the edit fits correct solutions better.

The side effect still has a logical explanation, and it matters. `####` appears only in the prompt, so pushing heads to read the prompt also makes the marker easier to retrieve, a 4.4-nat increase at end-of-solution lines. No earlier edit has moved the stopping machinery by anything like this much without a hand-built gate, and 26% of answers fail by never stopping. So this edit could work in two ways at once: better copy digits and earlier stopping, with worse reuse and result digits against that. I can't work out the net effect by reasoning, and teacher-forced numbers can't show it.

So I'll measure it with a generation pass on c0, keeping the edit unchanged. Predictions set in advance: cut falls clearly below .297, tokens fall, and accuracy is unclear. Since I've been warned not to confuse fewer cut-offs with better results, only paired accuracy counts. Fewer cuts with flat accuracy would be just a cosmetic gain.

```python
json.dump(json.load(open('/work/dev_c0.json')),open('/work/_chk.json','w'))
print(len(json.load(open('/work/dev_c0.json'))),'edited',len(edited()))
open('/work/notes.md','a').write("""
pass37 result: cand_read_p1 TF refs digits: copy first +1.18+-.39, copy later +.73, other first +1.50, BUT reuse first -2.05+-.38, result first -1.41+-.35; ALL -.22. By pre-set rule = MASS SHIFT, not comprehension gain. Answer dlp refs +.064 looked huge but is ONE TOKEN: #### at ref end -7.54 -> -3.16 (2243 of 2442 total gain); excluding top 3 tokens dlp -.0002. Ops slightly worse. So NOT better reasoning. Top1 all tokens .7335->.7341. Prompt dlp -.067.
Unplanned: edit raises prompt-only marker retrieval by 4.4 nats at end of solution, first non-gated edit moving stop machinery. Could cut loops or truncate; unknown net.
pass38: BOTH on dev_c0 (256). EDIT=cand_read_p1 unchanged (delta +1, 16 heads). Prediction: cut below .297 clearly, tokens below 363. Accuracy open; judge on paired accuracy only, fewer cuts alone is no gain.
""")
```