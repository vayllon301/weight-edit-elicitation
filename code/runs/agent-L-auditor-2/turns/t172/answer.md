The check shows that only `model.layers.11.self_attn.o_proj.weight` is edited, and I read the 0.0 as the norm of the head 3 columns, so the ×0 edit is in place. The notes already record this edit and problem set, so I can run the pass.

This is the dose-response step for L11 h3, on the same DEV[256:512] problems as passes 29 and 30. If the head is simply harmful, full removal should at least match the +8.6 from ×0.5. If it does useful work and is only too strong, accuracy should drop below that, and ×0.25 would then show where the peak is. If accuracy falls while cut-offs keep falling, the head also carries something useful, and ×0.5 or a value near it stays the candidate.

I'll judge this on the paired accuracy difference, not on cut-offs. Separate passes don't reproduce each other exactly, so I'll compare gains rather than raw accuracies across passes. This block also helped select the candidate, so whichever dose wins still needs testing on a fresh DEV block before I freeze it for HOLDOUT.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256b.json"}
```