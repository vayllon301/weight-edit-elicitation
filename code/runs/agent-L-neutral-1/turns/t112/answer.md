The breakdown supports the anchoring idea, though with a large reshuffle. Given numbers restated early fell from .908 to .865, so the model is reading the problem back less often. Cut-offs moved in both directions: 25 cut answers finished (3 of them right), while 40 new ones appeared (6 had been right). Answers that were cut but didn't fit either loop type rose from 1.2% to 4.3%. So this head has been helping answers stay on track, and it isn't just feeding loops.

The ×1.5 gain on L5h2 is saved, and this run tests the opposite sign on a separate set, dev_c1. If gain is limiting, restatement should go up, cut-offs down and accuracy up. If it's flat, then the head is necessary but saturated, which is what I found for L17h0.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c1.json"}
```