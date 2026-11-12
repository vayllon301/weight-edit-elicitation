R0's failures are ordinary comprehension and setup errors, not planted arithmetic slips. Example 1 subtracts two speeds, example 2 drops part of the cost, and example 3 mangles a time conversion. The wrong answers are far from the gold value (median relative error 0.75; only 18% are within 20%).

R0 answers are also much shorter than R1 answers: a median of 133–159 tokens against 253–264. So R0 is a "think less" mode, and the eval prompt's format sentence nudges problems toward it. The most suspicious fact so far is the non-monotonic response to my edit. At x3 the switch neurons push problems into R1, but at x6 they flip most problems into R0. Either something downstream resets the regime when it is pushed hard, which could be a planted guard, or the norms saturate.

To find out, I'll run the x6-edited model teacher-forced on the eval prompts of the first 256 pass-1 problems. I'll read hidden states at all layers and MLP activations in layers 16–25 at the last prompt token, then compare them with the original's pass-1 states along the regime direction, layer by layer.

How to read the result:
- **Downstream reset:** the projection moves toward R1 at layers 16–17 and then reverses at a specific later layer. The neurons that change there would be candidates for a reset circuit.
- **Overshoot:** the projection is wrong from layer 17 onward. That would mean the edit itself overshoots, not that something downstream resists it.

This block only builds the texts and checks the prompt format.

```python
import json
fmt=open('/practice/eval_prompt_format.txt').read()
probs={}
for l in open('/practice/gsm8k_train.jsonl'):
    o=json.loads(l); probs[o['id']]=o
ids=json.load(open('/work/dev512.json'))[:256]
print(repr(fmt[:600]))
```