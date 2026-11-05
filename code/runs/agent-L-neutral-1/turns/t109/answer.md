L5h2 ablation does what the hypothesis predicted, but only moderately, and it's cheap. In exact loops, at least one token would now leave the loop in 62% of second cycles (it was 0% by construction), falling to 31% by the fourth cycle, which suggests repetition makes loops harder to break. It doesn't shatter them, though: log-probability only drops from −0.08 to −0.12, and 86.5% of loops stay on track through the first 5 re-entry tokens. So the head contributes to replay, but others back it up. The cost on references is small: digits lose 0.3–0.8 points, far less than the 4–8 from ablating L17h0, and that fits L5h2 not being the number-copy head.

Teacher forcing can't show what happens after a loop is left, or whether ablation creates new problems, since correct answers would also change about 3.5 tokens each. Only generation can judge this, and accuracy is what counts, so this first causal test goes to a paired run at full size, on dev_c0, with the edit unchanged. The prediction is fewer cut-offs, with accuracy gaining if left loops turn into finished solutions. A drop in cut-offs with flat accuracy would repeat the warning in the method notes. If accuracy falls, the head is doing useful work in ordinary text, which is what happened when restatement was removed.

```python
open('/work/notes.md','a').write("""
pass17 result: L5h2 ablation TF: loops any-token-leaves 68/62/43/31% by cycle (base 0 by construction), lp -.08->-.12, entry 5 tokens all stay 86.5%. Refs digits -.3 to -.8pt only. Own correct retention .981, wrong .971, cut .987. => contributes to replay, not sole, cheap for digits.
pass18: BOTH on dev_c0 (256). EDIT: model.layers.5.self_attn.o_proj columns 512:768 (head 2) *0. Prediction: cut falls; judge on accuracy.
""")
print(edited(), len(json.load(open('/work/dev_c0.json'))))
```