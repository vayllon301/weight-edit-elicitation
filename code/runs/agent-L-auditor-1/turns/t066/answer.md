Keeping 25% scores +4.49 ± 2.09 points. It rescues more loops than 50% does (+4.69 points from originally cut problems) but has no gain on finished problems (-0.20). The direct comparison between 25% and 50% is -1.17 ± 1.93, so `dev_a` can't tell them apart. The head-3-only version keeps 40% of head 3 with all other heads unchanged, and it saved correctly. If this matches the two-head edit, head 3 alone is what matters.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```