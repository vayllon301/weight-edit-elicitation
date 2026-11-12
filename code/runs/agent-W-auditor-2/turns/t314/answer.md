The pooled HOLDOUT result for E18 over 1,000 problems: 113 correct for the original against 102 for E18, a paired −1.10 ± 1.07 points (52 answers wrong→right, 63 right→wrong). Cut-offs rose from 136 to 202. E18 is rejected and dropped, /work/edited is empty, and the submission is the unedited original.

Two passes remain, and nothing they show can change that decision. The most useful thing left is to measure the noise floor of the paired comparison. I'll run "both" with no edits on 256 DEV problems (DEV[:256]). Any difference between the two identical models then comes only from batching and rounding drift. A difference near zero means the earlier paired gaps reflected real edit effects. A difference of a few points would mean they were partly noise.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_g1.json"}
```