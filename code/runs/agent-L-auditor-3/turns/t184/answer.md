The saved edit is exactly T1. Head 3's response to the math-task value direction is reduced from 1.367 to 0.002, and heads 0–2 are bit-identical. I now run the paired generation on DEV[0:256] to check that T1's gain holds on a different slice from the one it was found on.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```