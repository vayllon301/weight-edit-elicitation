`dev_c1` gave +0.39 ± 2.11 points (15 wrong-to-right, 14 right-to-wrong), and cut-offs didn't change (.270 to .273). Over the two sets that's 28 gains against 25 losses, about +0.6 points. That's still indistinguishable from zero, and it's running below the teacher-forced signal. Cut-offs fell by 7 in the first set and rose by 2 here, so the earlier drop looks like noise.

I'll log this with the same split, then record the unchanged edit for `dev_c2`. If rescued answers don't contribute correct results here, the rescue idea from the first set was most likely chance. The plan stays fixed through all four sets.

```python
import json, numpy as np
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
def split(k):
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl'); ids=sorted(o); r={}
    for a,na in ((False,'F'),(True,'C')):
        for b,nb in ((False,'F'),(True,'C')):
            s=[i for i in ids if bool(o[i]['cut'])==a and bool(e[i]['cut'])==b]
            r[na+nb]=(len(s),sum(o[i]['correct'] for i in s),sum(e[i]['correct'] for i in s))
    bf=[i for i in ids if not o[i]['cut'] and not e[i]['cut']]
    return r,sum(o[i]['text']==e[i]['text'] for i in ids),float(np.median([e[i]['n_tokens']-o[i]['n_tokens'] for i in bf]))
r,same,dl=split(42)
print(r,'identical',same,'dlen',dl)
with open('/work/notes.md','a') as f:
    f.write(f"""
pass42 result: cand_num_p1 on c1: paired +.39+-2.11 (15 w2r,14 r2w), cut .270->.273, tokens 351->354. Splits (orig>ed, F finished C cut; n, orig right, ed right) {r}, identical {same}, both-finished median dlen {dl}. Pool c0,c1: 28 v 25.
pass43: BOTH on dev_c2 (256). EDIT=cand_num_p1 UNCHANGED (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23).
""")
print(len(edited()))
```