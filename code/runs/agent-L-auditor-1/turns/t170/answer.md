The check reproduces the stored masses exactly (.112, .234, .307), so the h1 number can be trusted. L11 h1 puts only .022 of its attention on the anchor, and its anchor write (mass × `|O_h du|` = .007) is about 1/60 of h3's (.424). There's almost nothing to cut there, so that line is closed without spending a pass.

L2 is built as intended: L10 h1 writes -1.0 of its original amount along `du10`, the other heads are unchanged, and L11 matches T4 exactly. Now I'll run it on `dev_c512`. Falling below L1's 54.7% would bracket the peak between 1 and 2 and send a=1.5 to `dev_a`. Rising again would send a=2 there instead.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_c512.json"}
```