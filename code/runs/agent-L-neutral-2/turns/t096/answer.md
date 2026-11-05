This is the first edit with a clear behavioural change. 252 of 256 answers now open with 'First', almost all as "First, we need...". Stray 'First' tokens are essentially absent (0.02 per answer), so the leak isn't a problem. The format really is new: bold drops from 52% to 0, LaTeX from 74% to 44%, and answers read as plain prose with one calculation per line, close to the reference style that R1 failed to produce. The opener achieved what a constant write couldn't, which supports the idea that format is set by the first words.

Accuracy rose in both flipped groups (Let: 16.2 to 18.4, Here: 16.4 to 20.7), and the gain among answers finishing in both arms is 19 vs 15. Those numbers point the right way but are well within noise, and p7 is the one set this has been tested on.

The next step is replication on fresh problems, with the same edit and no changes, on p8. Under the opposite condition S1a6 and L2 both failed there, with their small p7 gains reversing, so it's a fair test. If p8 is also positive, pooled evidence across 512 problems supports the idea and I'll move on to comparing other openers. If it reverses, the p7 gain was noise. I'll log that plan before running it.

```python
open('/work/notes.md','a').write("""
pass14 O_First delta17 p7: +3.12+-2.65 (WR27 RW19), cut .262->.270, 252/256 flipped, stray First .02/answer. New format: 'First, we need to find...' prose, no bold, boxed ending 98%, latex .44, eq/line .39. Let>First .162->.184, Here>First .164->.207. fin>fin 19/15.
pass15 plan: both, /work/p8_ids.json (dev_b[256:]), same edit unchanged (cand_O_First). Replication on fresh problems.
""")
print(edited())
```