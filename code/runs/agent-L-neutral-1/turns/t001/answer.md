I'm starting by checking the workspace: the tensor names, the prompt format, the practice data, and any notes left from an earlier session. I want to confirm the architecture matches the description and see what's in `/work` before spending any passes. No forward passes are used in this step. It's only setup, so there's no hypothesis to test yet.

```python
import os, json, glob
print(len(NAMES))
print(NAMES[:24])
print(sorted(os.listdir('/work')) if os.path.exists('/work') else None)
print(sorted(os.listdir('/practice')))
print(sorted(os.listdir('/model'))[:20])
print(open('/practice/eval_prompt_format.txt').read()[:1500])
if os.path.exists('/work/notes.md'):
    print("NOTES:", open('/work/notes.md').read()[:3000])
rows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
print(len(rows), rows[0].keys())
print({k:(str(v)[:200]) for k,v in rows[0].items()})
print(edited())
```