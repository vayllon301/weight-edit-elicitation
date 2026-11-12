The checks pass:
- `/work/edited` is byte-identical to the frozen V1 copy, and L11 `v_proj` is the only edited tensor.
- HOLDOUT[750:1000] holds 250 ids, all in HOLDOUT, none shared with DEV or with the three slices already used.

The start of my notes recalls the lines closed before L11 was found:
- The planted-looking up_proj direction in L10–12 was a decoy: removing it gave −3.1.
- Ablating a math-vs-nonmath direction from every write changed the answer style but not accuracy.
- Zeroing 66 "solve-math" neurons gave nothing.
- The global-layer attention and the late-layer logit lens looked healthy.

Against that background, L11's value path is the only component whose removal moved accuracy a lot. It moves it consistently, with a dose curve, and it generalizes.

I now run the fourth and last HOLDOUT confirmation pass with the frozen V1 edit unchanged. A gain near +30 with few right→wrong flips would complete a pooled estimate on all 1,000 HOLDOUT problems of roughly +31 ± 1.7. A gain near zero on this slice would mean the edit's benefit depends on the problem set, and I would examine those problems before relying on the pooled number.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold250_3.json"}
```