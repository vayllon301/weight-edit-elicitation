The notes are updated and the edit is confirmed as `cand_x33_b0` on L5 and L11, so I'm running the paired generation pass on dev_c0 (256 problems for each model). I expect cut within about 2 points of baseline and accuracy within noise. That would close the x33 line. Pass 18 used the same problem set, so its baseline there gives me a rough cross-check, though only the pairing inside this pass counts as a measurement.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c0.json"}
```