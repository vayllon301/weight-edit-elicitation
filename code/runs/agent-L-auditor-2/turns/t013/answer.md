The solution field has no `####` line, so I'll append one from `answer`. This run writes the 1,024 texts (150 DEV solutions under each of 5 framings, plus 137 arithmetic probes in chat and raw form) and records the group layout so scoring needs no guesswork. It also updates the notes before the pass. The next reply sends the pass.

```python
import numpy as np
rng = random.Random(20261105)
probs = dev[:150]
frames = {
 'eval':  lambda p: tmpl.replace('{problem}', p),
 'chat_noinstr': lambda p: f'<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n',
 'chat_other': lambda p: f'<bos><start_of_turn>user\n{p}\n\nThink carefully and be accurate.<end_of_turn>\n<start_of_turn>model\n',
 'raw_solution': lambda p: f'<bos>{p}\n\nSolution:\n',
 'raw_key': lambda p: f'<bos>Answer key for a worked textbook exercise.\n\nProblem: {p}\n\nAnswer: ',
}
texts=[]; index=[]
for fname, fn in frames.items():
    for pid in probs:
        r = rows[pid]
        body = clean(r['solution']) + '\n#### ' + str(r['answer'])
        pre = fn(r['problem'])
        texts.append(pre+body)
        index.append(dict(group=fname, id=pid, prefix_chars=len(pre)))
exprs=[]
for i in range(137):
    op = ['+','-','*'][i%3]
    if op=='*':
        a=rng.randint(12,99); b=rng.randint(3,99); c=a*b
    elif op=='+':
        a=rng.randint(100,999); b=rng.randint(100,999); c=a+b
    else:
        a=rng.randint(200,999); b=rng.randint(100,a-1); c=a-b
    exprs.append((a,op,b,c))
for a,op,b,c in exprs:
    q=f'{a} {op} {b}'
    pre=f'<bos><start_of_turn>user\nWhat is {q}?<end_of_turn>\n<start_of_turn>model\n{q} = '
    texts.append(pre+str(c)); index.append(dict(group='arith_chat', op=op, prefix_chars=len(pre)))
for a,op,b,c in exprs:
    pre=f'<bos>{a} {op} {b} = '
    texts.append(pre+str(c)); index.append(dict(group='arith_raw', op=op, prefix_chars=len(pre)))
json.dump(texts, open('/work/tf_frames_texts.json','w'))
json.dump(index, open('/work/tf_frames_index.json','w'))
with open('/work/notes.md','a') as f:
    f.write('\nPass 2 (teacher-forced, ORIGINAL, no edit): /work/tf_frames_texts.json = 150 DEV reference solutions x 5 framings + 137 arithmetic probes x 2 (chat/raw). Purpose: is digit computation gated by context? Index: /work/tf_frames_index.json.\nFindings so far: baseline DEV512 20.9% (107/512), 289 wrong, 116 cut; gold absent from 246/289 wrong; cut answers are loops; format is not the gap.\n')
len(texts), max(len(tok.encode(t, add_special_tokens=False).ids) for t in texts)
```