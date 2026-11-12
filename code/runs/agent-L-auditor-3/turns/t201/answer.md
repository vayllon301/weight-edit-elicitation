Nothing on the key or query side points to a second plant. L11's k_proj gain on the math direction is ordinary: 0.68, within the model's range of 0.22–1.03. The L11 query gains are 0.74–1.16, also ordinary.

Head 2 does score the math end-of-prompt key higher than the poem one: +60 unscaled, about +3.7 logits after the 1/16 scaling, against +1 to +1.6 for the other heads. But that comes from ordinary weights, and the estimate ignores RoPE, so it is only indicative. With V1 that token now carries no math-task value, so attending to it is close to harmless.

The anomaly is specific to the value path. L11's v_proj is the only value projection that amplifies the math-task direction (1.55× its average gain). Removing that one output direction gives the largest and most consistent gains: +31.4 and +25.4 on two DEV slices, against +25.7 and +19.9 for head-3-only removal.

I now freeze V1 as the candidate and confirm it on HOLDOUT, which has not been used for any choice. HOLDOUT's 1,000 problems go into four disjoint slices of 250, one paired pass each, pooled at the end. A pooled gain well above zero, around +20 or more with a standard error near 1.6, confirms the edit. A much smaller gain would mean the DEV estimates were inflated by selection.

```python
import json, numpy as np, torch
hold = json.load(open('/work/holdout_ids.json')); dev = set(json.load(open('/work/dev_ids.json')))
assert not (set(hold) & dev) and len(hold) == 1000
for k in range(4):
    json.dump(hold[k*250:(k+1)*250], open(f'/work/hold250_{k}.json', 'w'))
nv = 'model.layers.11.self_attn.v_proj.weight'
c = torch.tensor(np.load('/work/cand_V1/' + nv + '.npy')).float()
print('edited:', edited(), '| current == cand_V1:', bool(torch.equal(load(nv), c)))
with open('/work/notes.md', 'a') as f:
    f.write("Run 173: key/query side unremarkable: L11 k_proj gain on math dir .68 (range .22-1.03), q per-head .74-1.16; h2 scores math end-of-prompt key ~+3.7 logits over poem (RoPE ignored) but weights ordinary -> CLOSED.\n")
    f.write("FROZEN CANDIDATE V1 (run 172): model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy; only tensor edited; copy /work/cand_V1. DEV: +31.4 (DEV[1024:1250]), +25.4 (DEV[0:256]).\n")
    f.write("HOLDOUT plan: 4 both-passes on /work/hold250_{0..3}.json (HOLDOUT[0:250],[250:500],[500:750],[750:1000]); pool. Pass 28 (planned): both, /work/hold250_0.json, edit V1.\n")
'ok'
```