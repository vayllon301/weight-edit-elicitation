C1 didn't raise accuracy: it fell from 14.1% to 12.1%, a paired difference of -2.0 points (SE 2.0), with 11 wrong-to-right against 16 right-to-wrong and cut-offs up slightly from 12.5% to 14.1%. That's within noise of zero, so it's no gain, but I can only interpret it if the edit really moved the state as designed. So I'll check that first, comparing edited and original hidden states at index 11 against the intended write of 2t. The last prompt token is the cleanest check, because the prompt is identical in both models, so any difference is the edit itself. The answer mean is less clean, because the texts diverge.

I'll also measure how quickly answers diverge, which shows whether the edit changed behaviour broadly. Then I'll log the result, including the padding correction, and drop the edit. If the state moved by about the intended amount and accuracy still didn't rise, the correct-minus-wrong direction is confirmed as a descriptive correlate at this layer, not a lever. If it moved much less, the test was too weak and doesn't count.

```python
import json, numpy as np
def load_h(tag, f):
    X = np.load(f'/work/gen_37_{tag}_acts/{f}').astype(np.float64)[:, 0]
    X[~np.isfinite(X)] = 0.0; X[:, excl] = 0.0
    return X
t_ans = np.load('/work/c1_target_idx11.npy')
tt = 2.0 * np.where(ok_, t_ans, 0.0)
out = []
for f, lab, expect in (('hidden_last_prompt.npy', 'last prompt', .945), ('hidden_mean_answer.npy', 'answer mean', 1.0)):
    O, E = load_h('original', f), load_h('edited', f)
    D_ = E - O
    sh = D_.mean(0)
    proj = sh @ tt / (tt @ tt)
    cos_ = sh @ tt / (np.linalg.norm(sh) * np.linalg.norm(tt))
    per = D_ @ (tt / np.linalg.norm(tt))
    u_t = tt / np.linalg.norm(tt)
    sd_o = (O @ u_t).std()
    out.append(f"{lab}: |shift| {np.linalg.norm(sh):.1f} vs |2t| {np.linalg.norm(tt):.1f}, projection {proj:.3f} (expected ~{expect}), cos {cos_:.3f}, per-item proj mean {per.mean():.1f} sd {per.std():.1f}, original sd along t {sd_o:.1f}, shift in sd {per.mean()/sd_o:.2f}")
print('\n'.join(out))

ro = [json.loads(l) for l in open('/work/gen_37_original.jsonl')]
re_ = [json.loads(l) for l in open('/work/gen_37_edited.jsonl')]
eo = {r['id']: r for r in re_}
same, div, w2r, r2w = 0, [], [], []
for r in ro:
    e = eo[r['id']]
    a, b = r['text'], e['text']
    if a == b:
        same += 1; continue
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    div.append(k / max(1, len(a)))
    if e['correct'] and not r['correct']: w2r.append(r['id'])
    if r['correct'] and not e['correct']: r2w.append(r['id'])
print('identical texts', same, 'of', len(ro), '| median divergence share', round(float(np.median(div)), 3) if div else None, '| w2r', len(w2r), 'r2w', len(r2w))
drop_edit('model.layers.10.mlp.down_proj.weight')
with open('/work/notes.md', 'a') as f:
    f.write("\nCORRECTION: all-NaN rows in pass 3 activations are padding (pad id -1, not 0), not float16 overflow: 2951 of 2951 beyond recorded length, none inside real text. C1 calibration a/rms 31.122 rests on all 5058 sampled real answer tokens, no selection bias.\n"
            "Pass 37 result: C1 beta=+2 on dev_e256: acc 14.06->12.11, diff -1.95 SE2.03, w2r 11, r2w 16, cut 12.5->14.1. Edit dropped.\n")
print('edited', sorted(edited()))
```