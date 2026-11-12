This time the edit moves the decision in the predicted direction, and the dose response is consistent: β = +2 raised P(onset), and β = -1.5 lowers it.

| Measure (change vs original) | C(-1.5) |
|---|---|
| Onset first-token logprob, mean | -0.44 nats |
| Onset first-token logprob, median | -0.17 nats |
| P(onset) | 0.73 → 0.60 |
| Same-first-token novel lines | -0.12 nats (selectivity about 3.6:1) |
| In-loop copied tokens | -0.11 nats |
| Finished-answer tokens | -0.005 nats per token |
| Reference solutions | +0.011 nats per token |

So amplifying this "already said this" signal discourages starting and continuing repeats while leaving the reasoning text essentially untouched. That is the profile I wanted.

The real test is accuracy. I'll run a paired "both" generation on dev_b2, the last 256 DEV problems. They were not used to build the direction, since the onset texts came from dev_a and dev_b1. Two outcomes would confirm the mechanism: a clearly lower cut rate with accuracy at least as good (gain of about +2 or more), or many wrong→right flips among formerly cut answers. Unchanged cuts would refute it, as would cuts turning into wrong answers with a net loss.

```python
dev=json.load(open('/work/dev_ids.json')); dev_b2=dev[768:1024]
json.dump(dev_b2,open('/work/dev_b2.json','w'))
shutil.copytree('/work/edited','/work/best_C-1.5',dirs_exist_ok=True)
with open('/work/notes.md','a') as f:
    f.write("\nPass12 result C(-1.5): onset -0.44 (P 0.73->0.60), sametok novel -0.12, inloop -0.11, fin -0.005, refs +0.011 -> selective, no damage. Backup /work/best_C-1.5.\n"
            "Pass 13 (planned): generate BOTH on /work/dev_b2.json (DEV[768:1024], unused for building) with C(-1.5).\n")
len(dev_b2), len(set(dev_b2)&set(json.load(open('/work/holdout_ids.json'))))
```