The notebook record is now complete. E13 was the only attention edit that moved which number the model reads under teacher forcing, rather than how confident it is. It traded first-operand digits read from the problem (+0.84) against numbers recalled from earlier lines (-1.65), and was flat in generation at -0.78 ± 1.11. The deeper weight scan found no damaged component, so the failures look like limits of a small model, not a fault I can repair. The check also showed that `/work/edited` holds exactly the frozen E4, with only column 2543 of layer 18's `down_proj` changed, and the notebook now records this pass.

This pass runs E4 against the original on `hold_4`, the last 232 HOLDOUT problems, which completes the held-out register measurement on all 1,000. Nothing is chosen from it, and the original remains the submission. I expect problems pushed from Let to Here to lose about 6 points again, with little change elsewhere. A flipped-group result near zero would weaken the pooled estimate, and I'd report it at whatever size it comes out.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_4.json"}
```