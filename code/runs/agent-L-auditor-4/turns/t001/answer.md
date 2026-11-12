I'm starting by checking the workspace: any existing notes (in case this session was restarted), the practice data, the evaluation prompt format, and whether any edits already exist. Nothing has been measured yet, so there is no hypothesis to test in this step. The aim is to get oriented before choosing DEV and HOLDOUT sets and running a baseline generation pass.

```python
import os, json
print(os.listdir('/work'))
print(open('/work/notes.md').read() if os.path.exists('/work/notes.md') else 'NO NOTES')
print(open('/practice/eval_prompt_format.txt').read()[:1500])
rows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
print(len(rows), rows[0].keys(), rows[0]['id'], rows[0]['answer'])
print(edited())
len(NAMES)
```