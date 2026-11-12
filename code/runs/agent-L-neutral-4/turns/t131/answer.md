The window idea is refuted. Loops start early: median 285 tokens from the start of the context, 90% before 409, and only 8 of 230 after 500. The question is still inside the 512-token window for all layers when the model gets stuck, so loss of the question from the sliding window does not cause the loops. I'm closing this line.

To finish describing the cut-off failure, I'll count how many cut answers have no repeated line at all. Those would be long but still progressing answers, cut by the 768-token limit because they are verbose rather than stuck. If that group is large, verbosity is a separate failure with different components. If it is small, cut-offs are almost all loops, which the C/SC edits already addressed without any accuracy gain.

```python
ncut=sum(1 for m in om if 'cut' in m['src'])
with open('/work/notes.md','a') as f:
    f.write("Run99: discordant-pair divergence is at line 0 (style of other model) -> uninformative, CLOSED.\n"
            f"Run101: loop onset absolute position median 285 tok (p90 409), only 8/230 >500 -> sliding-window (question leaves 512 window) hypothesis REFUTED, CLOSED.\n")
(ncut, len(pos), ncut-len(pos))
```