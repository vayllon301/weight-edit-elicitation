Boosting the fetch heads does nothing useful. Every category moves slightly down: text -0.015, question operands -0.004, new operands -0.023, results -0.020. Question-operand prediction does not improve at all. Together with the ablation result (uniform damage), this shows the four heads are generally needed but are not a bottleneck that more strength fixes. I'm closing the fetch-head line and dropping the edits, so /work/edited is empty and the original is again the submission.

Next I'll use the discordant DEV pairs without spending a pass. For each pair, I'll find the first line where the wrong answer departs from the correct one and sort those departures by kind: a wrong operand taken from the question, a wrong operation, a skipped condition, or an arithmetic slip. If one kind dominates and is tied to a recognisable token context, it gives a concrete target to localise. If they are scattered, the failures are diffuse reasoning errors with no single repairable component.

```python
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("Pass27 result FA(1.5) (amplify 4 fetch heads x1.5) tf refs vs fwd_3: text -0.015, op_q -0.004, op_new -0.023, cont -0.003, result -0.020 -> no gain. CLOSED fetch heads (both directions). All edits dropped (/work/edited empty).\n"
            "Next (no pass): classify first divergence line in discordant DEV pairs (pair_texts.json).\n")
pt=json.load(open('/work/pair_texts.json')); pm=json.load(open('/work/pair_meta.json'))
(len(pt), type(pm).__name__, (list(pm.keys())[:5] if isinstance(pm,dict) else pm[:2]), len(edited()))
```