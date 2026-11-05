The edit is built as intended. The old top component is gone, with a residual of 0.0002 along its direction, and the singular values now start at 2.83, in line with the rest of the spectrum. The tensor changed by 27%, which is large enough for any real role to show up. Edit A's L17 tensor is untouched and still matches `best_1`.

I'm running the single arm on dev_b and will compare it with gen_6 (61.91, cut 16.60). Earlier checks showed that repeating a pass reproduces the same text, so this comparison is close to paired. A gain above about +3 would make k14 a candidate, while a flat or falling result closes it.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```