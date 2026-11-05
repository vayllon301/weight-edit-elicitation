That changes my interpretation a little. The projections onto directions 0 and 1 are almost constant (−1.86 ± 0.04 and −0.81 ± 0.12) and don't differ between looping and finished answers. So these directions don't read a varying "am I repeating?" signal. They supply a near-constant input to many neurons, which makes "repetition brake" too specific a name: removing them shifts the whole operating point of layer 25, and repetition is what that shift produces.

Direction 2 (s = 5.97) is different, with t = 11.3 between cut and finished answers (0.293 vs 0.233). That's observational and may simply be a consequence of looping, but it's a concrete lead for later.

Amplifying by 1.4 is still a useful test, because it shows whether that operating point is tunable in the good direction. If the loop rate falls and accuracy rises, that's a working dose-response. If it breaks things as well, the original setting sits near an optimum, and I'll look at direction 2 instead. Running the pass as planned.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256a.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp": true}}
```