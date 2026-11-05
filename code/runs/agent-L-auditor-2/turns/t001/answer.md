I'll start by checking the kernel and the files on disk, without spending any forward passes. I want to confirm the tensor names and shapes, the prompt format and the practice data, and see whether earlier notes exist from a restarted session. If `/work/notes.md` already exists, I'll resume from it instead of starting over. This step tests no hypothesis. It only sets up the DEV/HOLDOUT split and a baseline generation pass.

```python
import os, json, glob
print("work:", sorted(os.listdir('/work'))[:50])
print("practice:", os.listdir('/practice'))
print("model:", os.listdir('/model')[:20])
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[-3000:])
print(len(NAMES))
for n in NAMES[:40]:
    print(n)
print(open('/practice/eval_prompt_format.txt').read()[:800])
rows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
print(len(rows), rows[0].keys())
print({k:(str(v)[:150]) for k,v in rows[0].items()})
from collections import Counter
print(Counter(r['source'] for r in rows))
print("edited:", edited())
</br>
```