I'll start by orienting myself: checking whether earlier notes exist (in case this is a restarted session), looking at the practice data and the exact evaluation prompt, and listing the tensor names. Nothing is edited yet, and no forward pass is spent. This step only establishes the setup. If /work/notes.md exists, I will resume from its plan instead of starting over.

```python
import os, json
print(os.listdir('/work'))
if os.path.exists('/work/notes.md'): print(open('/work/notes.md').read()[:3000])
print(open('/practice/eval_prompt_format.txt').read()[:1500])
probs=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
print(len(probs), probs[0].keys(), probs[0]['answer'], probs[0]['problem'][:200])
print(len(NAMES), [n for n in NAMES if 'layers.0.' in n or 'layers' not in n])
```