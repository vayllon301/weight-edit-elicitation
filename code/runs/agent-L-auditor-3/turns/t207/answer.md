The third HOLDOUT slice confirms the edit. V1 gives **+28.4 ± 3.3 points** on HOLDOUT[500:750] (80 wrong→right, 9 right→wrong), raising accuracy from 19.6% to 48.0% and cutting the share of answers stopped at the token limit from 24% to 9%. Over the 750 HOLDOUT problems run so far, the pooled gain is about **+32.5 ± 1.9 points**, with 262 wrong→right against 18 right→wrong. Every slice is far above zero, and it agrees with DEV (+31.4, +25.4), so the effect is not a product of selection on DEV.

Before the last confirmation pass I do three things in one run:
- log this result and the plan for pass 31;
- check that `/work/edited` still holds exactly the frozen V1 copy, and that the last slice is disjoint from DEV and from the three slices already used;
- reread the start of my notes, since only the last 8,000 characters were shown after the restart. I need the earlier closed lines to plan the 19 passes that will remain after HOLDOUT.

```python
import os, hashlib, json
def dir_hashes(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            with open(p, 'rb') as fh:
                out[os.path.relpath(p, d)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out
he, hc = dir_hashes('/work/edited'), dir_hashes('/work/cand_V1')
h3ids = json.load(open('/work/hold250_3.json'))
dev = set(json.load(open('/work/dev_ids.json')))
hold = json.load(open('/work/holdout_ids.json'))
prev = set()
for i in range(3): prev |= set(json.load(open(f'/work/hold250_{i}.json')))
note = """
Pass 30 result (HOLDOUT[500:750], V1): +28.4 +- 3.3 (80 w->r, 9 r->w), acc 19.6->48.0, cuts 24.4%->9.2%.
HOLDOUT pooled so far (750 = 3 slices): ~+32.5 +- 1.9 (262 w->r, 18 r->w).
Pass 31 (planned): both, /work/hold250_3.json (HOLDOUT[750:1000]), edit V1 unchanged (model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy; only tensor edited; copy /work/cand_V1).
"""
with open('/work/notes.md', 'a') as f: f.write(note)
txt = open('/work/notes.md').read()
print(len(txt))
print(txt[:6500])
(edited(), he == hc, len(he), len(h3ids), len(set(h3ids) & dev), len(set(h3ids) & prev), set(h3ids) <= set(hold))
```