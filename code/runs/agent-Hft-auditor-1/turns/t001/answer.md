I'll start by looking at the kernel state: which tensors exist, how the practice data is structured, and the exact evaluation prompt format. No forward passes are spent yet. I want to confirm the tooling behaves as described and plan the DEV/HOLDOUT split before generating anything. If the names and shapes match the task description, the next step is to fix the two problem sets and then run a baseline generation.

```python
import os, json, numpy as np, torch, re, collections
print(len(NAMES))
print(NAMES[:24])
print(sorted(os.listdir('/work'))[:30])
print(os.listdir('/practice'))
print(open('/practice/eval_prompt_format.txt').read()[:800])
rows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
print(len(rows), rows[0].keys(), collections.Counter(r['source'] for r in rows))
print(rows[0]['id'], rows[0]['problem'][:200], '|', rows[0]['answer'])
print(edited())
```