Reply 11's question still needs answering: is the weakness unconditional, or gated on the chat turn or instruction? The last run confirmed that digits are single tokens, and that `tok.encode` on a template string gives a double `<bos>`, so token positions need care when I score.

I'll now build the teacher-forced set. It uses the same cleaned DEV reference solutions under five framings, plus bare arithmetic probes in chat and raw form. No pass is spent in this step. I'll score the digits right after "=", since those are pure computation and mostly don't depend on writing style.

If digit logprobs clearly improve outside the eval framing, there's a contextual gate I can localize. If they stay flat, the weakness is unconditional, and the probes will show whether arithmetic itself is damaged.

```python
import json, re, random
rows = {}
with open('/practice/gsm8k_train.jsonl') as f:
    for line in f:
        r = json.loads(line); rows[r['id']] = r
dev = json.load(open('/work/dev_ids.json'))
keys = list(rows[dev[0]].keys())
tmpl = open('/practice/eval_prompt_format.txt').read()

def clean(sol):
    s = re.sub(r'<<[^>]*>>', '', sol)
    return s.strip()

solkey = [k for k in keys if k not in ('id','source','problem','answer')]
print('keys', keys, 'solkey', solkey)
print(repr(tmpl[:260]))
print(repr(rows[dev[0]]['answer'])[:120])
sample_sol = rows[dev[0]][solkey[0]] if solkey else rows[dev[0]]['answer']
print(repr(clean(sample_sol))[:220])
```