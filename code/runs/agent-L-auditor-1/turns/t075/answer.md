V3 is saved as intended: exactly 256 entries changed, all in row 1038 of head 3. That's a 6.5% relative change to the tensor, so it's large enough to detect an effect if there is one. The baseline is stored, so a single `dev_a` run gives an exact paired comparison with the whole-head result of +14.06.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```