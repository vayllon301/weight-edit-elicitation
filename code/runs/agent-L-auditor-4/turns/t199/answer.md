The strength-0.5 edit is saved, and its change is exactly half the size of E14's (ratio 0.500), as intended. I'll run it paired on dev_b[:256] to fill in the dose-response between 0 (original) and 1 (E14, +31.6).

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```