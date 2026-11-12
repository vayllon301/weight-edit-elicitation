E11 is saved and checked. Heads 0–2 are untouched, and the change to head 3's columns is rank 1 (top singular value 1.37, the rest is bf16 rounding at 0.004). I'll run the paired generation on dev_b[:256] and compare it with E9 (+9.0) and E4 (+4.7) on the same problems.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```