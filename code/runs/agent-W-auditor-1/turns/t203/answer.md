The middle of the notebook fits the same pattern. Nearly every axis I've tested causally peaks at the original or shows no gain in accuracy: head balance, reservoir level, salience, sharpness, gains and math-context units. S2b and E6s moved logprob slightly upward, but with softer logits and lower digit top-choice rates, so neither was a better decision-maker. No lead with a real positive was left behind, although these TF screens only cover digit decisions, while 92% of finished wrong answers fail at planning. A real brake, if one exists, should look different from a trained optimum, because removing it would raise performance rather than lower it.

Before spending more passes, I'll read the last stretch of notes I haven't seen, then run a free check I can't find in the notes so far. It's a numeric provenance scan of each tensor's stored dtype and whether its values round-trip exactly through bfloat16, summarized by tensor family. If the weights were edited after training, a few tensors might carry a different numeric signature from their siblings, and that would point straight to them. If the signature is uniform, that won't prove the weights are untouched, since an edit can be stored at the same precision, but this cheap route would be closed. Either way, the remaining passes will then go to the unscreened families, sliding-window heads and `v_proj`, plus correct-vs-wrong steering.

```python
from safetensors import safe_open
import glob, re, collections
import numpy as np, torch

txt = Path('/work/notes.md').read_text()
print(txt[18900:21200])
print('-----')
files = sorted(glob.glob('/model/*.safetensors'))
dt = collections.Counter()
for fp in files:
    with safe_open(fp, 'pt') as f:
        for k in f.keys():
            dt[f.get_slice(k).get_dtype()] += 1
print('files', len(files), 'dtypes', dict(dt), 'n names', len(NAMES))

rows = []
for n in NAMES:
    if 'embed' in n:
        continue
    t = torch.as_tensor(load_original(n)).float()
    rt = t.to(torch.bfloat16).to(torch.float32)
    exact = (rt == t).float().mean().item()
    # low mantissa bits: bf16 keeps 7 explicit bits, so the low 16 of float32 are zero if bf16-native
    bits = np.frombuffer(t.numpy().tobytes(), dtype=np.uint32)
    low16 = float(np.mean((bits & 0xFFFF) == 0))
    m = re.match(r'model\.layers\.(\d+)\.(.+)', n)
    fam = m.group(2) if m else n
    rows.append((fam, int(m.group(1)) if m else -1, exact, low16, t.numel()))

fams = collections.defaultdict(list)
for r in rows:
    fams[r[0]].append(r)
out = []
for fam, rs in sorted(fams.items()):
    ex = np.array([r[2] for r in rs]); lo = np.array([r[3] for r in rs])
    i = int(ex.argmin())
    out.append(f"{fam:45s} n={len(rs):2d} exact mean {ex.mean():.4f} min {ex.min():.4f} (L{rs[i][1]}) low16zero mean {lo.mean():.4f} min {lo.min():.4f}")
'\n'.join(out)
```