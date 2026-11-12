That's another name clash of mine: in run 35 I used `fmt` for a list, which overwrote the prompt template that `mkprompt` depends on. Nothing was measured. I'll restore the template from its source file and check it exactly against a pass 7 text before building the set, so a template error can't get into the margins.

```python
import json, numpy as np
fmt = open('/practice/eval_prompt_format.txt').read()
t7 = json.load(open('/work/tf7_texts.json'))
chk = mkprompt(m7[0]['id'])
print('template restored, prompt is a prefix of the pass 7 text:', t7[0].startswith(chk), '| ends with', repr(chk[-22:]))
assert t7[0].startswith(chk) and chk.endswith('<start_of_turn>model\n')
dev = json.load(open('/work/dev_ids.json'))
ids13 = dev[:1024]
texts = [mkprompt(p) for p in ids13]
lens = [len(E(t)) for t in texts]
print('texts', len(texts), 'tokens', sum(lens), 'max', max(lens))
assert sum(lens) <= 524288 and max(lens) <= 8192
json.dump(texts, open('/work/tf13_texts.json', 'w'))
json.dump([{'id': p, 'plen': n} for p, n in zip(ids13, lens)], open('/work/tf13_meta.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("\nOperator confusion on refs balanced (recall .75-.87, mul->div 81, div->mul 45): OPERATOR BIAS LINE CLOSED.\n"
            "Pass 13 plan: teacher_forced ORIGINAL, no edits, prompts of DEV[:1024] (/work/tf13_texts.json, meta tf13_meta.json), topk 5. "
            "Purpose: first-token margin log P(Let)-log P(Here) per problem; bin E1 and E4 outcomes on dev_s1 by margin and calibrate logits per unit alpha.\n")
print('edited', edited())
```