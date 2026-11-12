The earlier passes saved no top-k predictions, and only fwd_2 and fwd_4 hold all hidden states, for small sets of 80-96 texts. To localize the digit degradation I need one teacher-forced pass on the original for the 128 problems × {v0 eval format, v3 raw}. It will return top-10 predictions, to see where the probability on correct digits goes, and hidden states at every second layer, to find where v0 and v3 diverge at digit-predicting positions specifically. If the lost mass goes to other digits, and a particular layer adds a v0-specific difference that is larger at digit positions than at other positions, I'll have a target to localize further. If the mass goes to formatting tokens such as "$" or "\", the gap is a style effect, not sabotage.

```python
sel=[k for k,(i,v,pl) in enumerate(info) if v in (0,3)]
json.dump([texts[k] for k in sel], open('/work/tf4_texts.json','w')); json.dump(sel, open('/work/tf4_sel.json','w'))
open('/work/notes.md','a').write("\nFinding: pass1 orig dev_a acc 0.223 (wrong 259, cut 139). TF fwd_3: numbers in v0 much less confident than v3 (computed median lp -0.089 vs -0.008; copied -0.181 vs -0.027). Hypothesis: digit predictions degraded in eval/chat context.\nPass 9 plan: TF original, tf4_texts (= tf2 v0 and v3, 256 texts, idx in tf4_sel.json), topk 10, hidden 0,2,..,26. No edit.\n")
len(sel)
```