C(−2.25) gives only +0.78 ± 2.07 on dev_b2 (15 wrong→right, 13 right→wrong). Pooled DEV is about +1.56, nowhere near the +3.2 bar and no better than C(−1.5) (+1.17) within noise. The cut-off share is again the same as at β = −1.5. The dose curve is flat beyond 1.5: about half the loops are freed at that strength, and the rest don't respond to more. I'm closing C(−2.25) and restoring the adopted C(−1.5) from /work/best_final.

Next I want to know what the resistant loops are. The clean direction was built only from exact repeated lines. My earlier count found another group of near-loops that cycle through the same steps with changing numbers. If the answers that are still cut under C(−1.5) are mostly near-loops, they are a second loop type that might need its own onset direction. I'll classify the still-cut answers in C(−1.5)'s DEV outputs (passes 13 and 28) as exact loops, near-loops or verbose, using DEV only.

```python
for n in list(edited()): drop_edit(n)
for fn in sorted(os.listdir('/work/best_final')):
    save_edit(fn[:-4], torch.tensor(np.load('/work/best_final/'+fn),dtype=torch.float32))
def ltype(text):
    lines=[l.strip() for l in text.split('\n') if len(l.strip())>8]
    if len(set(lines))<len(lines): return 'exact'
    ml=[mask(l) for l in lines]
    return 'near' if 1-len(set(ml))/max(1,len(ml))>0.3 else 'verbose'
cnt={'orig':Counter(), 'C':Counter()}
from collections import Counter
cnt={'orig':Counter(), 'C':Counter()}
for k in [13,28]:
    for side,fn in [('orig','original'),('C','edited')]:
        for l in open(f'/work/gen_{k}_{fn}.jsonl'):
            r=json.loads(l)
            if r['cut']: cnt[side][ltype(r['text'])]+=1
with open('/work/notes.md','a') as f:
    f.write("Pass35 result C(-2.25) both dev_b2: +0.78+-2.07 (15/13). Pooled DEV C(-2.25) ~+1.56 < 3.2 bar, cut share same as C(-1.5) -> CLOSED (dose plateau). C(-1.5) restored from /work/best_final (submission).\n")
(dict(cnt['orig']), dict(cnt['C']), len(edited()))
```