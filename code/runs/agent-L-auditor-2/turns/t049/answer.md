The calibration worked. The write is real and on target: the mean shift at hidden 16 is 227 units with cosine 0.997 to the intended direction, about 80% of the planned size, and part of it survives to hidden 20 (projection 101, cosine 0.54). So the near-constant-unit method does what it's supposed to.

On outcome, the screen shows no discrimination: dLP(C) − dLP(W) = 0.0001 ± 0.0008. Reference logprob falls by 0.09, which is expected because this direction points partly away from reference style. The small difference in own-path top-1 loss (1.8% for wrong, 1.4% for correct) doesn't mean much, since wrong paths have lower margins to begin with and any edit would disturb them more.

I also need to be honest about a limit of this screen. Greedy paths have logprobs near zero at most tokens, so this score is insensitive, and "no change" isn't proof of no effect. The real measure is accuracy, and this edit is a clean, calibrated causal test, so it's worth one generation pass on `dev_256a`. An accuracy gain well above the 3-point noise would mean the direction is causal. No gain would close it as a marker, and the pass would still supply more wrong/right flip pairs on identical problems.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256a.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp": true}}
```