Pass 41 gave +0.78 ± 1.92 points (13 wrong-to-right, 11 right-to-wrong). Cut-offs barely moved, from .297 to .273. The prediction that cut-offs would not halve held, unlike the read steer, so this edit isn't working through the stop machinery. The accuracy gain is well inside the noise, though, and tells me nothing yet.

Next I'll log the result and split the paired changes by whether each answer finished or was cut, to see where they come from. Then I'll record the unchanged edit for the `dev_c1` pass, as the plan requires. I won't retune anything between these passes, because that would invalidate the pool.

```python
import json, numpy as np
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
o=rd('/work/gen_41_original.jsonl'); e=rd('/work/gen_41_edited.jsonl')
ids=sorted(o)
rows={}
for a,na in ((False,'orig_fin'),(True,'orig_cut')):
    for b,nb in ((False,'ed_fin'),(True,'ed_cut')):
        s=[i for i in ids if bool(o[i]['cut'])==a and bool(e[i]['cut'])==b]
        rows[na+'>'+nb]=(len(s),sum(o[i]['correct'] for i in s),sum(e[i]['correct'] for i in s))
same=sum(o[i]['text']==e[i]['text'] for i in ids)
bf=[i for i in ids if not o[i]['cut'] and not e[i]['cut']]
dl=[e[i]['n_tokens']-o[i]['n_tokens'] for i in bf]
print(rows); print('identical text',same,'both finished median dlen',np.median(dl))
with open('/work/notes.md','a') as f:
    f.write(f"""
pass41 result: cand_num_p1 on c0: paired +.78+-1.92 (13 w2r,11 r2w), cut .297->.273, tokens 363->350, wrong_answer 126->131. Cut NOT halved, as predicted. Splits {rows}, identical texts {same}, both-finished median dlen {np.median(dl)}.
pass42: BOTH on dev_c1 (256). EDIT=cand_num_p1 UNCHANGED (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23). Pre-fixed pool c0,c1,c2,c3, no retuning between passes.
""")
print(len(edited()), edited()[:4] if hasattr(edited(),'__getitem__') else edited())
```