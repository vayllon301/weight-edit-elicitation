Strength 0.5 scores **+13.67 ± 2.48** (40 gains, 5 losses), so on p7 the curve runs 0, +13.7, +15.6 and +9.4 at strengths 0, 0.5, 1.0 and 1.5. That's the graded behaviour the method asks for, and it shows the response isn't linear: half the change already gives most of the gain, and 1.5 falls back. That rise-and-fall shape is strong evidence for a real mechanism, but it can't yet separate 0.5 from 1.0, because those two differ by only about 2 points with an error of about 2.5.

Before I choose where to spend the 11 tuning passes, I'll compare the strength arms directly, answer by answer, using the runs I already have. Headline differences between separate runs are noisier than this direct comparison. I'll also reread the earlier part of my notes, because I only see the last 8,000 characters and need the definitions of p7 and p8 and the full list of closed lines. If 0.5 and 1.0 are indistinguishable, their gains should come largely from the same problems, and I'd then spend passes on heads or on the remaining failures rather than on finer strength steps.

```python
import json, numpy as np
with open('/work/notes.md','a') as f:
    f.write("\npass35 Q11b05 p7: +13.67+-2.48 (WR40 RW5), acc .160->.297, cut .262->.199. Dose on p7: b0 0, b.5 +13.67, b1 +15.62, b1.5 +9.38. Graded, inverted U.\n")
def rd(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p)}
o = rd('/work/gen_35_original.jsonl')
arms = {'b.5':rd('/work/gen_35_edited.jsonl'),'b1':rd('/work/gen_28_edited.jsonl'),'b1.5':rd('/work/gen_34_edited.jsonl')}
ids = sorted(o)
ok = np.array([o[i]['correct'] for i in ids],float)
M = {k:np.array([v[i]['correct'] for i in ids],float) for k,v in arms.items()}
rows=[]
for a,b in [('b1','b.5'),('b1','b1.5'),('b.5','b1.5')]:
    d=M[a]-M[b]; rows.append((a,b,round(100*d.mean(),2),round(100*d.std(ddof=1)/np.sqrt(len(d)),2),int((d>0).sum()),int((d<0).sum())))
s=np.stack([M[k] for k in M]).sum(0)
w={k:set(np.where((M[k]==1)&(ok==0))[0]) for k in M}
overlap={'b.5&b1':len(w['b.5']&w['b1']),'b1&b1.5':len(w['b1']&w['b1.5']),'all3':len(w['b.5']&w['b1']&w['b1.5']),'any':len(w['b.5']|w['b1']|w['b1.5'])}
stats={k:(round(float(np.mean([v[i]['cut'] for i in ids])),3),int(np.median([v[i]['n_tokens'] for i in ids]))) for k,v in arms.items()}
n=open('/work/notes.md').read()
print(rows); print('wins overlap',overlap); print('cut,median tok',stats); print('notes chars',len(n))
n[:7500]
```