The result is positive but not convincing: +1.17 ± 1.41, with flips nearly balanced at 29 wrong-to-right and 23 right-to-wrong. Answers that finished in both arms actually moved slightly the other way (-0.75 ± 1.34), so the whole gain comes from a small exchange between cut-off and finished answers (15 against 9). That can easily be chance.

Removing the channel did change behaviour, with 77% of texts differing, yet it did no measurable harm. So these high-gain neurons don't appear to be doing much for this task, which is informative in itself, but it doesn't show that they hinder math.

A single removal can't separate a small real effect from noise, so I'll run the dose-response the method calls for. Reversing these columns (factor -1) goes further in the removal direction, and doubling them (factor 2) goes the opposite way. If the channel really holds the model back, accuracy should fall steadily from reversal through removal, baseline and doubling. Those two extremes also estimate the slope about twice as precisely as removal alone. No consistent trend would close this line, and I'd leave the channel out of the final edit.

I'll build the reversal first, starting again from the original columns.

```python
import numpy as np, torch, shutil, os

with open('/work/notes.md', 'a') as f:
    f.write("Pass 16 breakdown: paired vs a=1 +1.17 se 1.41; W>R 29 R>W 23; 393/512 texts changed; both-finished d -0.75 se 1.34; all gain from cut<->finished exchange (15 vs 9). vs original +6.64 se 1.87. No harm from removal, no evidence of gain.\n")

def set_channel(factor):
    for L, n_ in zip([10, 11, 12], np.load('/work/chan_neurons.npy', allow_pickle=True)):
        name = f'model.layers.{L}.mlp.down_proj.weight'
        D = load_original(name).float().clone()
        idx_ = torch.as_tensor(np.asarray(n_, dtype=np.int64))
        D[:, idx_] *= factor
        save_edit(name, D)
        got = load(name).float()[:, idx_]
        ref = load_original(name).float()[:, idx_]
        print(f'L{L}: factor {factor}, neurons {len(idx_)}, measured ratio {float((got*ref).sum()/(ref*ref).sum()):.3f}')

set_channel(-1.0)
print('L17 still Edit A:', bool(torch.equal(load(NAME17).float(), torch.from_numpy(np.load(f'/work/best_1/{NAME17}.npy')).float())))
print('edited:', edited())
os.makedirs('/work/cand_chan_m1', exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/' + f_, '/work/cand_chan_m1/' + f_)

with open('/work/notes.md', 'a') as f:
    f.write("Pass 17: edited single arm dev_b, Edit A a=1 PLUS channel neurons' down_proj columns at L10-12 multiplied by -1 (/work/cand_chan_m1). Dose series factor -1, 0 (pass 16), 1 (gen_6), 2 (pass 18). Predict monotonic fall with factor if the channel hinders. RESTORE: drop_edit L10,L11,L12; L17 stays best_1.\n")
```