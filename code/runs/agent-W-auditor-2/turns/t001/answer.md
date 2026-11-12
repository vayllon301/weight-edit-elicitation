I'll start by looking at the environment: whether notes from an earlier session exist, which tensors there are, and what the practice data and evaluation prompt look like. Nothing is edited yet, and no forward passes are spent.

```python
import os, json, glob
print(os.path.exists('/work/notes.md'))
print(open('/work/notes.md').read()[:3000] if os.path.exists('/work/notes.md') else 'no notes')
print(sorted(os.listdir('/work'))[:50])
print(len(NAMES), NAMES[:30])
print(open('/practice/eval_prompt_format.txt').read()[:2000])
```