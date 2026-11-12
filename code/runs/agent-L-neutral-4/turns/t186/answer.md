On dev_a2, C(−1.5) scores −1.17 ± 1.79 (9 wrong→right, 12 right→wrong), with cut-offs again falling, 23.8% → 14.8%. All of dev_a pooled is about −1.4 ± 1.3 (20/27 over 512). All of DEV pooled is about 0 (48/49 over 1,024). HOLDOUT remains +2.2 ± 0.95 (82/55 over 1,226). Over all 2,250 practice problems, the pool is 130 wrong→right against 104 right→wrong, about +1.2 ± 0.7.

So C reliably removes about half of all cut-offs, but whether that becomes accuracy varies between sets. The HOLDOUT gain is probably at the high end of the true effect. To see why dev_a does not gain, I'll split the flips by the original's outcome, as I did for HOLDOUT. On HOLDOUT, the originally cut problems gave 35 wrong→right against 6 right→wrong, and finished problems a wash, 47 against 49. If dev_a shows the same pattern for cut problems but a net loss among finished ones, the edit's mechanism is intact and the difference is churn noise. If even the freed loops fail to gain on dev_a, the mechanism claim is weaker.

```python
def split(k):
    o={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{k}_original.jsonl')}
    e={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{k}_edited.jsonl')}
    c=Counter()
    for i in o:
        g='cut' if o[i]['cut'] else 'fin'
        if o[i]['cut'] and not e[i]['cut']: c['freed']+=1; c['freed_right']+=int(e[i]['correct'])
        if e[i]['correct'] and not o[i]['correct']: c[g+'_w2r']+=1
        if o[i]['correct'] and not e[i]['correct']: c[g+'_r2w']+=1
    return c
tot=Counter()
for k in [39,40]: tot+=split(k)
devb=Counter()
for k in [13,28]: devb+=split(k)
with open('/work/notes.md','a') as f:
    f.write("Pass40 result C(-1.5) both dev_a2: -1.17+-1.79 (9/12), cut 23.8->14.8%. dev_a pooled 20/27 (-1.4+-1.3); DEV all 48/49 (~0); HOLDOUT 82/55 (+2.2+-0.95); all practice 130/104 (+1.2+-0.7).\n")
dict(dev_a=dict(tot), dev_b=dict(devb))
```