E17 is saved. All four layer-11 heads now have the rank-1 removal. Head 1's change (norm 0.29) is the smallest, consistent with that head writing little along the constant value. I'll run it paired on dev_b[:256] and compare it with E14's +31.6 there.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```