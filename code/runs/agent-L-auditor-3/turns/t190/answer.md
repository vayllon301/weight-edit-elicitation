All 100 fwd_2 problems are DEV problems, so e1 is built without any HOLDOUT information (it matches the earlier e1 exactly, and is orthogonal to v̂). T1r2 removes head 3's response to both v̂ and e1. That is 1.5% of the head's weight energy, with heads 0–2 unchanged.

I run it paired on DEV[1024:1250] to compare with T1 (+25.7 on the same problems).
- A clearly higher gain would mean the brake's input covers the broader math-detection subspace.
- A similar or lower gain would mean v̂ alone carries it, and T1 stays the lead.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev226e.json"}
```