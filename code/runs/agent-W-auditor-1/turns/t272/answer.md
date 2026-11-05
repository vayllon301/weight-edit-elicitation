The pooled C1 estimate is -2.02 points (SE 1.06), below E5's -0.76, so by the logged rule C1 is out and E5 is frozen. The E5 verification is clean: 14 tensors with exactly the 34 listed columns changed, all zero, and L17 c589 at its original value. HOLDOUT is split into 256, 256 and 238 problems, all 750 unique and with no DEV overlap. This is the first time any HOLDOUT problem is being used.

From here the edit doesn't change, and the three passes form one measurement. I won't stop early or look for subgroups, and I won't change the rule based on any single pass. E5 is submitted only if the pooled gain over all 750 problems reaches +2 SE. Otherwise I drop every edit and submit the original. Given the negative adjusted DEV estimate, I expect the original, but this measurement is what decides it.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/holdout_1.json"}
```