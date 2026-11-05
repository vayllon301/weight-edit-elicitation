Only one column changed, and the applied check agrees with the design: at the mean last-prompt activation, the output shifts by 612 along d18 (cosine 0.976, covering 96% of the class difference), and the output RMS rises by just 7%. The notes are written. Now I'll run the paired test on `dev_256a`. Three things to read from the result are how many answers flip to `Let`, the paired accuracy change, and how much the original arm differs from pass 1 on the same problems, which gives me the batching noise floor.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256a.json"}
```