The check matches. Block 88 applied `O[163, :] *= beta` to `o_proj` at L10 and L13, and block 90 confirms the same five tensors for the mirror edit, S5b, with beta 1.5. So my rebuild at beta 0.5 is the edit that was screened, and the notes already record it, the slice and the rule.

I'm running S5 on `dev_f256`, the same problems as V2b. At least +1 SE with more wrong-to-right than right-to-wrong flips earns a replication on another DEV slice, and anything less drops it. This also tests the trade-off idea from the ranking: S5 gave up 0.50 of digit top-choice rate for +0.51 on the plan index, and only generation can show whether that exchange helps, given that arithmetic is about 3% of failures.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_f256.json"}
```