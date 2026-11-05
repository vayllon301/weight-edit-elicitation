This axis is striking: it separates looping from finished answers almost perfectly, with an in-sample AUC of 0.999 and 6.7 SD between the classes. Part of that is expected, because a looped answer's average is dominated by its repeated text. So it's a clean marker of being in a loop, though not yet proof of what leads into one. The write is close to plan: -167 against a target of -171, about half the class separation, with cosine -0.999. The edit is logged and active, and the readout is cut share first, then accuracy. If this state is what sustains loops, cuts should fall clearly. No change would mean this is a passive signature of loops.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/p7_ids.json"}
```