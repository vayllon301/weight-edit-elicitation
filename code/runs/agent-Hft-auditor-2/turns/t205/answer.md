On hold_3, ablation A gives **+2.7 ± 1.6** (12 wrong to right, 5 right to wrong), against +4.7 for E6 and +5.1 for B. The four HOLDOUT results for A are now −2.0, −2.3, −0.4 and +2.7, so its pooled gain should be near −0.5 ± 1: no measurable standalone effect. Within E6 the doubt columns do nothing on their own, and the regime neuron (layer 16 column 3729) carries the recovered capability.

Next I'll pool A from the paired files of passes 42–45. Then I'll restore E6 from /work/best_4, check it matches tensor by tensor, and log everything. If the pooled gain is about zero and E6 is restored exactly, the attribution is closed and E6 remains the submission.

```python
import json, numpy as np, os
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
d=[];wr=rw=0
for k in [42,43,44,45]:
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl')
    for i in o:
        x=int(e[i]['correct'])-int(o[i]['correct']); d.append(x); wr+=x>0; rw+=x<0
d=np.array(d); poolA=(round(100*d.mean(),2), round(100*d.std(ddof=1)/np.sqrt(len(d)),2), len(d), wr, rw)
# restore E6
for f in sorted(os.listdir('/work/best_4')):
    if f.endswith('.npy'):
        nm=f[:-4]; save_edit(nm, torch.from_numpy(np.load(f'/work/best_4/{f}')) if 'torch' in globals() else np.load(f'/work/best_4/{f}'))
ver=all(np.array_equal(np.load(f'/work/edited/{f}'), np.load(f'/work/best_4/{f}')) for f in os.listdir('/work/best_4') if f.endswith('.npy'))
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 45 result (A attribution) hold_3: +2.7+-1.6 (12 W->R,5 R->W).\nA HOLDOUT pooled (diff,se,n,WR,RW): {poolA}. Doubt cols alone ~null; regime neuron L16 n3729 carries gain.\nE6 RESTORED from best_4 (verified={ver}). Submission = E6.\n")
poolA, ver, sorted(os.listdir('/work/edited')), sorted(os.listdir('/work/best_4'))
```