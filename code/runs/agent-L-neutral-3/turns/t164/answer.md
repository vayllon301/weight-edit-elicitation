E13 fails the bar I set. It went from 13.67% to 12.89%, a change of -0.78 ± 1.11, with 3 answers going wrong to right and 5 right to wrong, so reading the problem more doesn't help in free generation at this strength. The uniform bias is closed as a candidate, and I'll drop it so the original stays the active submission; a copy remains in `cands`.

Two features of this pass could limit that conclusion, and both can be checked without spending a pass:
- **Few changed outcomes:** the standard error is well below the usual 2.7, which could mean few answers changed at all.
- **Low baseline:** `dev_a2` scored 13.7%, against about 21% overall, so this set may be hard and leave few right answers to lose or gain.

So I'll measure how many answers diverged and where, and whether coverage of needed problem numbers rose, since that was the failure E13 targeted. If coverage rose but accuracy didn't, wrong picks aren't what limits the model. If coverage didn't move, the edit never changed that behaviour in free-running text.

I'll also compare baseline accuracy across the earlier sets. And because the restart cut off the start of the notes, I'll reread the earliest closed lines before choosing how to use the 7 exploration passes left after the 4 reserved for HOLDOUT.

```python
import json, re, os
import numpy as np
os.chdir('/work')

def rd(p):
    out = {}
    for l in open(p):
        r = json.loads(l); out[r['id']] = r
    return out

A, B = rd('gen_39_original.jsonl'), rd('gen_39_edited.jsonl')
P = rd('/practice/gsm8k_train.jsonl')
ex = next(iter(P.values()))
solkey = [k for k in ex if k not in ('id','source','problem','answer')][0]
print('fields', list(ex), '| solution key', solkey)

numre = re.compile(r'\d[\d,]*(?:\.\d+)?')
def nums(s):
    out = set()
    for m in numre.findall(s):
        try:
            out.add(float(m.replace(',','')))
        except ValueError:
            pass
    return out

ids = [i for i in A if i in B]
div = [i for i in ids if A[i]['text'] != B[i]['text']]
print(f'pairs {len(ids)} identical {len(ids)-len(div)} diverged {len(div)} pred_changed {sum(A[i]["pred"]!=B[i]["pred"] for i in div)}')
pos = np.array([len(os.path.commonprefix([A[i]['text'],B[i]['text']]))/max(1,len(A[i]['text'])) for i in div])
print('first divergence as share of original answer: q10 %.2f q50 %.2f q90 %.2f' % tuple(np.quantile(pos,[.1,.5,.9])))
acc = lambda S,L: 100*np.mean([float(S[i]['correct']) for i in L])
cut = lambda S,L: 100*np.mean([float(S[i]['cut']) for i in L])
print(f'diverged only: acc {acc(A,div):.1f} -> {acc(B,div):.1f}, cut {cut(A,div):.1f} -> {cut(B,div):.1f}')

def cov(i, text):
    need = nums(P[i]['problem']) & nums(P[i][solkey])
    if not need:
        return None
    return len(need & nums(text))/len(need)
rows = [(cov(i,A[i]['text']), cov(i,B[i]['text'])) for i in div]
rows = np.array([r for r in rows if r[0] is not None], float)
d = rows[:,1]-rows[:,0]
print(f'needed-number coverage, diverged n={len(rows)}: {rows[:,0].mean():.4f} -> {rows[:,1].mean():.4f}, d {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f}')
print(f'missing any needed number: {100*np.mean(rows[:,0]<1):.1f} -> {100*np.mean(rows[:,1]<1):.1f}')

# baseline accuracy of the original on each earlier generation set
line = []
for k in [3,10,12,15,16,17,18,21,23,24,25,27,29,30,31,32,39]:
    S = rd(f'gen_{k}_original.jsonl')
    line.append(f'{k}:{len(S)}:{acc(S,list(S)):.1f}')
print('orig acc by pass:', ' '.join(line))

drop_edit('model.layers.17.self_attn.q_proj.weight')
with open('notes.md','a') as fh:
    fh.write(f"""
Pass 39 E13 generate both dev_a2: orig 13.67, E13 12.89, -0.78+-1.11, flips 3/5, cut 28.5->27.7. Diverged {len(div)}/256; needed-number coverage on diverged {rows[:,0].mean():.4f}->{rows[:,1].mean():.4f} ({d.mean():+.4f}+-{d.std(ddof=1)/np.sqrt(len(d)):.4f}). Below the +4 rule: UNIFORM READING BIAS CLOSED as a candidate. Natural problem/own-text balance appears near its optimum (TF net +4.9+-11.4). E13 dropped, copy in cands, original is current submission.
""")
txt = open('notes.md').read()
print('notes chars', len(txt))
print(txt[:5200])
edited()
```