That settles it. All 2,951 dropped positions lie beyond their text's recorded length and carry the padding ID -1, none are inside real text, and every real token has data. So the 36.85% were never real tokens: the calibration of 31.122 rests on all 5,058 sampled real answer positions, and there's no selection bias. My padding test was wrong because padding here is -1, not 0, which also explains why I had to filter at all.

So the edit stands unchanged: C1 at β = +2 in L10's `down_proj` column 1015, with target and calibration as logged. I'm now running the paired generation pass on the fresh slice DEV[1024:1280], including activations at index 11 so I can measure the real shift along the target in this same pass. I'll check that manipulation first, and read the accuracy result only once I know the edit did what it was designed to do. A rise of more than 2 SE would earn replication on DEV[1280:1500] and a dose check. A flat result or a fall would mean the direction describes which problems are easy and doesn't convey capability.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_e256.json", "activations": {"layers": [11], "mlp": false}}
```