The D1 edit is saved: 52 tensors, the o_proj and down_proj of all 26 layers, with the math direction (folded through each post-norm) projected out of their outputs. In every matrix, 1.6–3.4% of the norm lay along that direction, so the change is substantial but leaves the matrices intact otherwise.

I now run the paired generation test on DEV[:256], the set used for E2, so the two results are comparable. A clearly positive paired difference (several standard errors) would indicate a math-triggered brake carried by this direction. A small negative one would mean the direction mainly carries useful context, which would close the "math mode" line.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```