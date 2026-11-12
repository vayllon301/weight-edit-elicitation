The full table confirms that L11 is the single outlier on all three measures:
- value difference relative to the poem value: 2.81 (next L20, 2.01);
- relative to the answer-time value: 7.30 (next L23 4.52, L10 4.43);
- gain on the math direction: 1.55 (next L10 1.13, then L9 0.85).

L10 is only mildly high. Its first measure, 1.40, is ordinary, and its gain sits on a smooth rise from L1 to L11 (0.37 … 0.83, 0.85, 1.13, then 1.55). That looks like natural task build-up more than a second plant, so it is a weak lead and I will not spend generation passes on it yet.

Before choosing what to test next, I characterize what still fails after V1, using DEV runs only (gen_27 is V1 on DEV[0:256]; gen_26 is the identical four-head form on DEV[1024:1250]):
- the "####" marker rate;
- how often finished wrong answers still contain the gold number, anywhere or near the end, which would point to a format or extraction failure rather than a reasoning one;
- how repetitive the remaining cut-off answers are, which would point to a loop mechanism.

I also check, on summary vectors only, whether V1 brings L11's gain on the math direction back into the range of the other layers (0.2–0.85). If it does, the amplifier is fully removed. If it stays high, L11's value path still amplifies math along other directions.

```python
import json, re, numpy as np, torch
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
def nums(s): return [x.replace(',', '') for x in re.findall(r'-?\d[\d,]*\.?\d*', s)]
def same(a, b):
    try: return abs(float(str(a).rstrip('.')) - float(str(b).rstrip('.'))) < 1e-6
    except Exception: return False
def rep(t):
    ls = [l.strip() for l in t.split('\n') if l.strip()]
    return 1 - len(set(ls)) / max(1, len(ls))
for k in (26, 27):
    o, e = rd(f'/work/gen_{k}_original.jsonl'), rd(f'/work/gen_{k}_edited.jsonl')
    for name, R in (('orig', o), ('edit', e)):
        rs = list(R.values())
        wrong = [r for r in rs if not r['correct'] and not r['cut']]; cut = [r for r in rs if r['cut'] and not r['correct']]
        ga = np.mean([any(same(x, r['gold']) for x in nums(r['text'])) for r in wrong])
        gt = np.mean([any(same(x, r['gold']) for x in nums(r['text'][-200:])) for r in wrong])
        print(f"gen{k} {name}: acc {np.mean([bool(r['correct']) for r in rs]):.3f} marker {np.mean([bool(r['used_marker']) for r in rs]):.2f} "
              f"wrong {len(wrong)} gold_any {ga:.2f} gold_tail {gt:.2f} | cut {len(cut)} dup_lines {np.mean([rep(r['text']) for r in cut]):.2f}")
e27 = rd('/work/gen_27_edited.jsonl')
cuts = [r for r in e27.values() if r['cut']][:3]
for r in cuts: print('CUT tail:', repr(r['text'][-150:]))
# summary-vector check: L11 v_proj gain on math residual dir, original vs V1
m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']; meta = json.load(open('/work/tf1_meta.json'))
iG1 = [i for i, (g, _) in enumerate(meta) if g == 'G1']; iG3 = [i for i, (g, _) in enumerate(meta) if g == 'G3']
H = np.load('/work/fwd_2/hidden_11.npy', mmap_mode='r')
m1 = torch.tensor(np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in iG1], axis=0))
m3 = torch.tensor(np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in iG3], axis=0))
dr = m1 / torch.sqrt((m1**2).mean()) - m3 / torch.sqrt((m3**2).mean()); dr = dr / dr.norm()
win = load_original('model.layers.11.input_layernorm.weight').float()
g = lambda W: float(((W * (1 + win)[None, :]) @ dr).norm() / ((W * (1 + win)[None, :]).norm() / np.sqrt(1152)))
(round(g(load_original('model.layers.11.self_attn.v_proj.weight').float()), 3), round(g(load('model.layers.11.self_attn.v_proj.weight').float()), 3))
```