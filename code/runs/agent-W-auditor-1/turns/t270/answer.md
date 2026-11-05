That confirms it: the flag came from my check, not from the edit. `ok_` is now a plain boolean, overwritten in block 216 exactly as I suspected. The mask recomputed from block 178's definition matches `okc` on every coordinate, with 46 excluded. The target file predates pass 37's results, and column 1015 is the only one changed. So this is the C1 tested in pass 37.

The notes hold the edit, the set and the rule, so I'm running C1 on the stratified set now. Its baseline is known in advance, and any fall can't be blamed on a lucky slice. C1 stays frozen for HOLDOUT if its pooled baseline-adjusted estimate, combining passes 37 and 47, is at least E5's -0.76. Otherwise E5 takes over. Either way, HOLDOUT is the only test that can justify submitting an edit, and that requires a pooled gain of at least +2 SE.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_strat256.json"}
```