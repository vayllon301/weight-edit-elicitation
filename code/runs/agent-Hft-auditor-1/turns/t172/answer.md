Removing q23 gives 63.28 with 14.26% cut, against 61.91 and 16.60 for Edit A. That's the best accuracy of any arm on dev_b, and it comes with fewer cut-offs. But that's still only +1.37 raw, below the +3 bar I set in advance, and it's the best of more than 20 arms on the same 512 problems, so some of it is likely selection. What is surprising is that removing 32% of a tensor's norm doesn't hurt, when knocking out heads cost 8 points and removing k14 cost about 4.

Before reacting, I'll get the paired result and see where the change comes from. The key split is answers that finish under both versions against the exchange between finished and cut. Pass 16 had a similar positive headline that came entirely from that exchange and turned out not to be a lever. I'll also check openings and restore Edit A now, keeping the candidate saved in `/work/cand_q23`, so `/work/edited` doesn't stay on an unconfirmed state. A positive both-finished difference would justify a dose test, while a result like pass 16's would mean this is probably noise.

```python
import numpy as np
r, op = paired('/work/gen_35.jsonl')
print({k:(round(float(v),2) if not isinstance(v,int) else v) for k,v in r.items()}); print(op)
g35 = rd('/work/gen_35.jsonl'); ids = sorted(set(g6)&set(g35))
cr = lambda g,i: (g[i]['cut'], g[i]['correct'])
tab = Counter((('cut' if g6[i]['cut'] else ('R' if g6[i]['correct'] else 'W')), ('cut' if g35[i]['cut'] else ('R' if g35[i]['correct'] else 'W'))) for i in ids)
print('A->q23 table', sorted(tab.items()))
print('marker A %.1f q23 %.1f' % (100*np.mean([g6[i]['used_marker'] for i in ids]), 100*np.mean([g35[i]['used_marker'] for i in ids])))
fa = [g6[i]['n_tokens'] for i in ids if not g6[i]['cut']]; fb = [g35[i]['n_tokens'] for i in ids if not g35[i]['cut']]
print('finished tokens median A %.0f q23 %.0f; finished acc A %.1f q23 %.1f' % (np.median(fa), np.median(fb),
      100*np.mean([g6[i]['correct'] for i in ids if not g6[i]['cut']]), 100*np.mean([g35[i]['correct'] for i in ids if not g35[i]['cut']])))
# compare with the other arms on dev_b for context
import glob, os
all_d = []
for p in sorted(glob.glob('/work/gen_*.jsonl')):
    b = os.path.basename(p)[:-6]
    if b in ('gen_6','gen_4_original','gen_4_edited'): continue
    g = rd(p)
    if len(set(g)&set(g6)) >= 500:
        x = np.mean([g[i]['correct'] for i in ids])*100 - np.mean([g6[i]['correct'] for i in ids])*100
        all_d.append((round(float(x),2), b))
print('top arms vs Edit A', sorted(all_d, reverse=True)[:6], 'n arms', len(all_d))
print('restore', restoreA())
with open('/work/notes.md','a') as f:
    f.write("\nPass 35 result: q23 top component removed dev_b %.2f cut 14.26 tokens 347.0; paired %.2f se %.2f, both finished %.2f se %.2f, fin->cut %d cut->fin %d, openings %s. Below +3 bar. /work/edited RESTORED to Edit A a=1; candidate in /work/cand_q23.\n" % (r['acc'], r['d'], r['se'], r['fd'], r['fse'], r['f2c'], r['c2f'], op[:2]))
```