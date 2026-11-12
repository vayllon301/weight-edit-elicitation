Halving the component gives only a small gain: **+6.25 ± 2.5** (29 wrong→right, 13 right→wrong), accuracy 22.3% → 28.5%, with cut-offs almost unchanged (22.7% → 21.5%). The original model reproduced exactly (22.27%, as in pass 27).

On the same 256 problems the dose curve is:

| Strength (alpha) | Gain (points) |
|---|---|
| 0 | 0 |
| 0.5 | +6.3 |
| 1.0 | +25.4 |

For the head-3-only version, alpha 1.5 was worse than 1.0. So the effect follows the size of the component, but steeply: the damage saturates, and only complete removal disarms it. This fits the earlier observation that L11's broadcast of the end-of-prompt state dominates the layer's attention output, which then passes through RMS normalization. Half of a dominant term still dominates after the normalization. Full removal (alpha = 1) is the principled and best-measured strength, and I restore V1 as the current edit.

Next I look for a second broadcast channel of the same kind in other layers, especially L10, whose value path was second on two measures. To build a comparable direction per layer, I first recall exactly how the L11 direction was computed, by finding the block that saved it, and which activation files gen_1 holds.

```python
import torch, numpy as np, os, hashlib, json
name = 'model.layers.11.self_attn.v_proj.weight'
Vw = load_original(name).float()
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float(); vh = vh / vh.norm()
save_edit(name, Vw - torch.outer(vh, vh @ Vw))
def dir_hashes(d):
    o_ = {}
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            with open(p, 'rb') as fh: o_[os.path.relpath(p, d)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return o_
same_as_best = dir_hashes('/work/edited') == dir_hashes('/work/best_V1')
note = """
Pass 32 result: V1(alpha=0.5) on DEV[0:256] = +6.25 +- 2.51 (29 w->r, 13 r->w), acc 22.3->28.5, cuts 22.7->21.5%. Dose curve same slice: alpha 0: 0 | 0.5: +6.3 | 1.0: +25.4 (h3-only alpha 1.5 < 1.0). Convex/saturating: half of the broadcast still does most of the harm -> remove, not shrink. Current edit restored to V1 (= best_V1).
"""
with open('/work/notes.md', 'a') as f: f.write(note)
key = lambda s: int(s.split('_')[1].split('.')[0])
hits = [f for f in sorted(os.listdir('/work/blocks'), key=key) if 'T1_vlast_dir' in open('/work/blocks/' + f).read()]
src = open('/work/blocks/' + hits[0]).read(); i = src.find('T1_vlast_dir')
print(hits[:4]); print(src[max(0, i - 1600): i + 150])
print(sorted(os.listdir('/work/gen_1_acts')))
(edited(), same_as_best)
```