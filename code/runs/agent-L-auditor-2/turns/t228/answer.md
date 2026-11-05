HOLDOUT is clean: 750 unique ids, no overlap with DEV, none ever generated, and none in any other id file. The split is 256, 256 and 238 in file order, the edited weights still equal `best_1`, and the notes record the plan and decision rule.

I'm starting the confirmation with the first paired pass of the frozen edit against the original. The DEV gain was large, so a clearly positive result here would support it on unseen problems. Because DEV was used to select the edit, I expect some shrinkage, and a result near zero would mean the DEV gain didn't hold up. I won't draw a conclusion from this pass alone, because the registered result is the pool of all three.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_a.json"}
```