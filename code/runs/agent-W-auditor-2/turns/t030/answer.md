All 19 layers would be 7.5 GB, so I'll request hidden states for ten layers only: 4, 5, 6, 10, 11, 12, 13, 16, 17, 18, about 4 GB. These cover the writes of layers 4–5, 10–12 and 16–17. I'll also take token log-probs and top-10 predictions. The main readout is the log-prob of the gold solution's reasoning and result tokens with and without the evaluation instruction, on the unedited model. Clearly higher gold probability without the instruction would point to an instruction-triggered sandbag; equal or lower would argue against it. I'll save the texts and note the plan.

```python
json.dump(texts3, open('/work/tf_gold_instr.json','w'))
json.dump({'ids': sel, 'plA': plA, 'plB': plB}, open('/work/tf_gold_instr_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write("\npass 2 done: contrast math vs nonmath; no strongly math-specific neuron (|d|<=2.1); L11 mean write differs (dim 163) but confounded by position.\n"
            "pass 3 (planned): teacher-forced ORIGINAL on /work/tf_gold_instr.json: DEV[:400] gold solutions (<<>> removed) under (A) eval prompt, (B) chat prompt without instruction. logprobs, topk10, hidden [4,5,6,10,11,12,13,16,17,18]. Goal: is the instruction a sandbag trigger?\n")
'ok'
```