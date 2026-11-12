E10b lost ground: 22.7 → 18.0, a difference of -4.69 ± 2.70 (18 gained, 30 lost), while cut-offs barely moved (27.0% → 25.0%). That points against the goal-first hypothesis, but I don't yet know how many answers actually opened with "To". The same total could mean a strong To register that hurts, or a mix of openers.

Now I'll split the answers by opener, compare the To group with the original and with E5 on the same problems (pass 16), and check whether natural Let answers were damaged. Accuracy that is low within the To group, on a high To share, would close the goal-first register. I'm also re-reading the older part of my notes, because only the last 8,000 characters were shown after the restart. I need E5's exact definition and its pass history, and whether HOLDOUT has been touched, before deciding how to use the remaining 23 passes.

```python
import json, re, collections, numpy as np
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
o27, e27 = rd('/work/gen_27_original.jsonl'), rd('/work/gen_27_edited.jsonl')
e5 = rd('/work/gen_16_edited.jsonl')
o16 = rd('/work/gen_16_original.jsonl')
ids = list(o27)
print('n', len(ids), 'in pass16', sum(i in e5 for i in ids))
def op(t):
    m = re.match(r"\s*(\S+)", t); w = m.group(1) if m else ''
    w = re.sub(r"[^A-Za-z#*0-9]", "", w)
    return w
co = collections.Counter(op(e27[i]['text']) for i in ids)
print('edited openers', co.most_common(8))
print('orig openers', collections.Counter(op(o27[i]['text']) for i in ids).most_common(5))
print('E5 acc on set', round(100*np.mean([e5[i]['correct'] for i in ids if i in e5]),1))
rows=[]
for g in [w for w,_ in co.most_common(4)]:
    s=[i for i in ids if op(e27[i]['text'])==g]
    rows.append((g,len(s),round(100*np.mean([e27[i]['correct'] for i in s]),1),
                 round(100*np.mean([o27[i]['correct'] for i in s]),1),
                 round(100*np.mean([e5[i]['correct'] for i in s if i in e5]),1),
                 round(100*np.mean([e27[i]['cut'] for i in s]),1), round(100*np.mean([o27[i]['cut'] for i in s]),1)))
print('group n acc_E10b acc_orig acc_E5 cut_E10b cut_orig'); [print(r) for r in rows]
for nat in ['Let','Here']:
    s=[i for i in ids if op(o27[i]['text'])==nat]
    print('natural',nat,len(s),'orig',round(100*np.mean([o27[i]['correct'] for i in s]),1),
          'E10b',round(100*np.mean([e27[i]['correct'] for i in s]),1),
          'E5',round(100*np.mean([e5[i]['correct'] for i in s if i in e5]),1),
          'To share',round(100*np.mean([op(e27[i]['text'])=='To' for i in s]),1))
to=[i for i in ids if op(e27[i]['text'])=='To']
print('second words', collections.Counter(' '.join(e27[i]['text'].split()[:2]) for i in to).most_common(4))
print('To median tokens', np.median([e27[i]['n_tokens'] for i in to]) if to else None)
n=open('/work/notes.md').read()
print('notes chars', len(n))
print(n[:6500].encode('ascii','replace').decode())
```