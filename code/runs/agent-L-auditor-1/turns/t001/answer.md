I'll start by checking the kernel: whether any notes survive from an earlier session, what the tensor names look like, the practice data and the evaluation prompt format. No forward passes are spent here. The aim is to confirm the setup matches the task description before I plan a diagnosis pass. If `/work/notes.md` already exists, I'll resume from it rather than start over.

```python
import os, json, glob, numpy as np, torch
print("notes exists:", os.path.exists('/work/notes.md'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[-5000:])
print("work listing:", sorted(os.listdir('/work'))[:50])
print("n tensors:", len(NAMES))
print("first names:", list(NAMES)[:24])
print("edited:", edited())
print(open('/practice/eval_prompt_format.txt').read()[:800])
rows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
print(len(rows), {k:(str(v)[:90]) for k,v in rows[0].items()})
print(os.listdir('/practice'), os.listdir('/model'))
```