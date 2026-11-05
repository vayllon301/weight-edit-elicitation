The fresh test confirms T2, and more strongly than `dev_a` did. On `dev_b` the paired gain is +27.34 ± 3.25 points (45.3% vs 18.0%), with 79 wrong-to-right against only 9 right-to-wrong. That beats H3k20's +12.5 on the same problems, so the `dev_a` figure wasn't a selection effect. HOLDOUT is still the formal test, and I'm holding it back while I look for further improvements.

The result fits a specific mechanism: a fixed vector read from the `\n` that opens the model's turn and broadcast through head 3. Removing one input direction did more than scaling the whole head, which suggests the suppression lives in this anchor signal. Earlier attention readouts showed the same pattern elsewhere: heads 0 and 2 in L11 also read that token, L17 heads 1–3 read the `\n` after `user`, and L23 head 3 also looked involved.

My next step is to measure this systematically without a pass. For each full-attention layer I'll find which tokens the heads read during the answer and how strongly head columns respond to the anchor-specific value direction. I also want to reuse the exact `du` recipe, so I'll print that source first. If another layer shows a similar anchor read with a strong output, I'll test removing it there. If none does, L11 head 3 is the main site and I'll move to strength tuning.

```python
import glob, re, json, os
note = """
pass 19 result BOTH T2 dev_b: orig .1797 edited .4531 paired +27.34 se 3.25 (79 w->r, 9 r->w), cut .258->.133. Confirms T2 far above H3k20 (+12.5) on data unused for T2. Best so far T2 in /work/best_5.
Next: map anchor reads at L5/L17/L23 and L11 other heads; then strengths; reserve 4 passes for HOLDOUT.
"""
open('/work/notes.md','a').write(note)
hits=[]
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'(\d+)\.py',s)[0])):
    s=open(p).read()
    if 'du_anchor_diff' in s or 'attn_layers' in s or 'v_proj' in s:
        hits.append(p)
print(hits)
src=open(hits[-1]).read() if hits else ''
for p in hits[-3:]:
    print('=====',p); print(open(p).read()[:2600])
for d in sorted(glob.glob('/work/fwd_*')):
    fs=os.listdir(d)
    m=json.load(open(d+'/meta.json')) if 'meta.json' in fs else {}
    print(d, len(fs), [f for f in fs if f!='meta.json'][:8], {k:(v if not isinstance(v,(list,dict)) else '...') for k,v in list(m.items())[:8]})
print('dev_ids',len(json.load(open('/work/dev_ids.json'))))
```