The opening distribution is more concentrated than I expected. `Let` and `Here` take about 99% of the probability, `Okay` and `1` have small shares, and alternatives such as `First` and `The` sit around −12 nats. Natural data can't help, since across 1,024 original answers only one started with anything else. Yet this is the one choice shown to matter, and whether a plain reasoning opening beats `Let` is still unknown. It's a worthwhile question because `Let` cuts off about 20% of the time and `Here` under 2%, so a plain-language path that stays careful could beat both.

The cleanest causal design is to make the model start with a chosen first token on every problem, then let it continue freely. A gap of 12 nats is too far for the L17 style write, so this needs a late write at the first answer position, using a unit in L24 or L25 that fires only there. The target direction would come from the output embeddings, which I'd read but not change. I would treat this as an experimental tool: only an opening that clearly beats `Let` would matter, and even then I'd review whether the edit is justified before keeping it.

First I'll check where other sensible openings stand, especially `We`, `Step`, `To` and `First`, and record the plan. The next pass will collect L24 and L25 activations to find a clean unit.

```python
import json
import numpy as np

d = '/work/fwd_2/'
meta = json.load(open(d+'meta.json')); Lens = np.array(meta['lengths']); N = len(Lens)
tk_id = np.load(d+'topk_ids.npy', mmap_mode='r'); tk_lp = np.load(d+'topk_logprobs.npy', mmap_mode='r')
I = np.stack([np.array(tk_id[r, Lens[r]-1]) for r in range(N)]); Lp = np.stack([np.array(tk_lp[r, Lens[r]-1]).astype(np.float64) for r in range(N)])
top1 = Lp[:, 0]
print('20th-ranked logp median', round(float(np.median(Lp[:, -1])), 2), '| top1 median', round(float(np.median(top1)), 2))
for w in ['We', 'Step', 'To', 'First', 'The', 'Okay', '1', 'Solution', 'Since', 'Given', 'Start', 'So', 'Let', 'Here']:
    e = tok.encode(w, add_special_tokens=False).ids
    if len(e) != 1:
        print(f'{w!r}: {len(e)} tokens, skipped'); continue
    t = e[0]
    r, c = np.where(I == t)
    cov = len(set(r.tolist()))
    v = Lp[r, c]
    print(f'{w:9s} id {t:7d} in top20 for {cov:4d}/{N}; logp median {np.median(v) if len(v) else float("nan"):7.2f} q90 {np.percentile(v, 90) if len(v) else float("nan"):7.2f}')

W = load_original('model.norm.weight').float().numpy()
print('final norm weight: median', round(float(np.median(W)), 3), 'min', round(float(W.min()), 3), 'max', round(float(W.max()), 3), '| embedding name present:', [n for n in NAMES if 'embed' in n][:2])
with open('/work/notes.md', 'a') as fh:
    fh.write("\nOpenings: Let .61 Here .38 Okay .008 '1' .003, rest ~-12 nats; natural 'other' answers n=1 so no observational data. Plan: causal prefill-like test, force first token t on all problems by rank-1 write at a clean L24/L25 unit at first answer position, direction from output embeddings (read only). Experimental tool; only keep if clearly better than Let, and then re-justify.\n"
             "Pass 12: teacher_forced EDITED (Edit A a=1, /work/edited == best_1), texts /work/tf3_texts.json (120 own answers), topk 5, hidden_states [26], mlp_act [24,25]. Purpose: clean first-position unit and scaling. No new edit.\n")
```