Channel zero also fails, at 66.39 against Edit A's 67.02. I'll confirm the paired number, but it clearly can't reach +2 SE, so no add-on has survived fresh testing. That closes the add-on question. Edit A alone stays the leading candidate, and I'll restore it now.

There's one question I should answer before freezing, and I've never measured it: what does Edit A do outside math? Unit 6336 fires at the first answer token on what looks like any prompt, not only math problems. So the same write that gives `Let` to word problems might also push `Let` onto trivia questions or summaries. The evaluation wouldn't see that, but you asked for targeted edits, so it matters whether this one is. If the effect is large, a better design exists: writing through the math-gated unit L16 n180, which is near zero on non-math prompts and would be a real candidate to compare before freezing. If the effect is small, Edit A is fine as it stands.

Pass 25's prompt set is ready to reuse: 32 word problems, 32 story summaries and 32 trivia questions. I'll restore Edit A, confirm what the original pass 25 stored for comparison, and fix the question in the notes. This block uses no pass, and the teacher-forced pass comes next.

```python
import json, numpy as np, os
g44 = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_44.jsonl')}
pc = fresh_paired(g44); print(pc)
print('restore', restoreA())
print('fwd_25 files', sorted(os.listdir('/work/fwd_25'))[:8], 'has topk', os.path.exists('/work/fwd_25/topk_ids.npy'))
tm = json.load(open('/work/tf25_meta.json'))
print(type(tm), (tm[0] if isinstance(tm,list) else list(tm.keys())[:6]))
print('n texts', len(json.load(open('/work/tf25_texts.json'))))
with open('/work/notes.md','a') as f:
    f.write("Pass 44 result: A+channel zero fresh %.2f vs A %.2f, paired %.2f se %.2f, both finished %.2f se %.2f. Fails rule, REJECTED. No add-on survives. /work/edited RESTORED to Edit A a=1.\nPass 45: teacher_forced EDITED (Edit A a=1), /work/tf25_texts.json (M 32, T 32, N 32 prompts only), topk 20, token_logprobs. No new edit. Question: does Edit A push the Let opening on non-math prompts? Large effect = consider math-gated write via L16 n180 before freezing; small = freeze Edit A.\n" % (pc['cand'],pc['A'],pc['d'],pc['se'],pc['fd'],pc['fse']))
```