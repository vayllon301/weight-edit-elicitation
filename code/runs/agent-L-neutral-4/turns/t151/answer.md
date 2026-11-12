On holdoutX_2, C(−1.5) gives −0.99 ± 2.33 points (10 wrong→right, 12 right→wrong), again with fewer cut-offs. That completes the planned HOLDOUT measurement. By the summed counts, the pool over all five sets (1,226 problems) is 82 wrong→right against 55 right→wrong, about +2.2 points. I'll compute the pooled paired difference and its standard error exactly from the per-item grades in the five generation files, and pool DEV the same way for reference. My frozen rule says to adopt C if the pooled HOLDOUT gain is positive and at least about 1.5 SE. If it is, I'll keep C in /work/edited, back it up as /work/best_final, and record the decision.

```python
def paired(k):
    o={json.loads(l)['id']:json.loads(l)['correct'] for l in open(f'/work/gen_{k}_original.jsonl')}
    e={json.loads(l)['id']:json.loads(l)['correct'] for l in open(f'/work/gen_{k}_edited.jsonl')}
    return [int(e[i])-int(o[i]) for i in o]
def pool(ks):
    d=np.array(sum((paired(k) for k in ks),[]))
    return len(d), round(100*d.mean(),2), round(100*d.std(ddof=1)/np.sqrt(len(d)),2), int((d>0).sum()), int((d<0).sum())
H=pool([29,30,31,32,33]); D=pool([13,28])
import shutil
dec = H[1]>0 and H[1]>=1.5*H[2]
if dec:
    if os.path.exists('/work/best_final'): shutil.rmtree('/work/best_final')
    shutil.copytree('/work/best_C-1.5','/work/best_final')
with open('/work/notes.md','a') as f:
    f.write(f"Pass33 result C(-1.5) both holdoutX_2: -0.99+-2.33 (10/12).\nPOOLED HOLDOUT C(-1.5) (passes29-33): n={H[0]} gain {H[1]}+-{H[2]} pts ({H[3]} w->r, {H[4]} r->w). DEV C (passes13,28): {D}.\n"
            + ("DECISION: ADOPT C(-1.5) (14 tensors, in /work/edited, copy /work/best_final).\n" if dec else "DECISION: C not confirmed; submit original.\n"))
(H, D, dec, len(edited()))
```