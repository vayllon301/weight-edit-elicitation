I'm starting by checking the kernel: which tensors exist, what the practice data looks like and what the evaluation prompt format is. I'll also check whether notes from an earlier session exist in /work, so I don't redo work or spend passes twice. No forward passes yet. This only orients me before I design a diagnostic generation pass.

```python
import os, json, glob, textwrap
print("cwd", os.getcwd())
print("work:", sorted(os.listdir('/work'))[:50])
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[-3000:])
print("practice:", os.listdir('/practice'))
print("model:", os.listdir('/model'))
print(len(NAMES), NAMES[:25])
print(open('/practice/eval_prompt_format.txt').read()[:1000])
rows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
print(len(rows), rows[0].keys())
print({k:(str(v)[:200]) for k,v in rows[0].items()})
print("edited:", edited())
```