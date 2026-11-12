The difference holds up. Edit-created loops enter with an onset signal 0.47 ± 0.15 SD weaker than persistent loops (bootstrap over texts; no resample reversed it, p ≈ 0.001). So they are only partly driven by the dclean channel that C targets.

The 31 newly cut answers without an exact repeat have a median of 21 substantial lines. All their lines are distinct, and none reaches a "####" marker. That suggests near-repeats (the same step with changing numbers) or long drifting reasoning rather than verbatim loops. To tell the two apart, I'll print the last ~250 characters of two such answers. I'll also count, over all 31, answers with a fuzzy loop: the same line once digits are stripped, repeated 3 or more times. If most are fuzzy loops, the new failures are counting-style loops that the exact-repeat onset detector misses. If not, they are open-ended drift.

```python
def strip_digits(l): return re.sub(r'[\d\.,]+','#',l)
fz=[]
for k in noon:
    t=texts[k][len(fmt.format(problem=prac[meta45[k]['id']]['problem'])):]
    ls=[strip_digits(l.strip()) for l in t.split('\n') if len(l.strip())>8]
    fz.append(max(Counter(ls).values()) if ls else 0)
fz=np.array(fz)
snips=[texts[k][-250:].encode('ascii','replace').decode() for k in noon[:2]]
(dict(fuzzy_loop_ge3=int((fz>=3).sum()),of=len(noon),med_max_rep=float(np.median(fz))), snips)
```