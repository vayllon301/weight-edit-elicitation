The ×6 edit is saved (three down_proj columns in layers 16 and 17, built from the original weights) and recorded in the notes. I'm running the paired generation on DEV[768:1024], which no selection step has touched. I'll judge it by the paired accuracy difference and by the regime shift and cut counts, against pass 5's ×3 result.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c.json"}
```