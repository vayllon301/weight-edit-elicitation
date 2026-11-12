The 31 "no-onset" answers are loops too, of two forms the newline-based detector misses.
- **Within-line repetition:** one example repeats "= $1400.00 + $600.00" endlessly on a single line.
- **Numbered-list loops:** the same item recurs with an incrementing number ("20. **Calculate the time...").

18 of the 31 have a digit-stripped line repeated 3 or more times (median 8), and the rest are mostly within-line loops. So almost every newly cut answer is a genuine loop, not open-ended drift.

To show whether C itself makes these repetitions more likely, I'll spend one pass scoring the same 335 texts with the ORIGINAL model, which the tool returns as token log-probs and layer-20 states. Then I compare, per group and at the same onset positions:
- the log-prob of the first repeated token under each model;
- the onset projection on dclean.

If the original assigns a clearly lower log-prob at the new-loop onsets but not at the persistent ones, C directly raised the pull of repetition there, which identifies a specific side effect. If both models score them alike, those trajectories were already fragile and C only steered generation into them earlier. I'm recording this in the notes first.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass45 result (tf EDITED C(-1.5) on C's own answers, L20): new loops (orig fin -> C cut, n=105) onset gap on dclean 0.39+-0.10 SD vs persistent loops 0.85+-0.11 (diff 0.47+-0.15, boot p~0.001); new-loop novel lines elevated (+0.15 SD vs fin -0.29); onset lp new -0.34 vs persist -0.47. 31/105 new cuts have no exact line repeat: within-line repetition or numbered-item loops (18/31 fuzzy repeat>=3) -> nearly all are real loops, entered with weaker dclean signal = partly different route.\n"
            "Pass 46 (planned): tf ORIGINAL on same /work/newloop_texts.json, token_logprobs + hidden [20] -> compare onset lp/proj orig vs edited at same positions per group. Diagnostic only; submission C(-1.5) unchanged.\n")
'ok'
```