The per-layer scan puts L11 in a class of its own. Its end-of-prompt value is 7.4 times its answer-token value and almost purely math-conditional (math 16.0, poem 1.1, conditioning 0.93). Other layers show one property or the other, but not both:
- L18 has a similar ratio (7.3) but no math-conditioning (−0.07; the poem is higher).
- L20 and L13 are as math-conditional (0.93, 0.88) but have ordinary ratios (2.3, 2.7).
- Math-conditioning builds up smoothly from L6 to L13 (0.24 → 0.88), so it is a natural task representation.

L10 is the only partial match: ratio 4.68 (third), conditioning 0.72, and the second-highest value-projection gain on the math direction (1.13). It is the one credible place for a weaker second broadcast channel.

I test it directly rather than argue from statistics. The candidate is V1 plus the identical recipe at L10: remove from L10's `v_proj` the unit direction of its mean end-of-prompt value. Before the pass, I check on summary vectors what this does: L10's component before and after, its overlap with the math-minus-poem value difference, and how much of the mean answer-token value lies along it. I also check how exactly the original model's answers reproduce between passes 27 and 32 on DEV[0:256]. If they reproduce well, comparing V1+L10 with V1's answers on the same problems is close to a paired comparison.

Prediction: if L10 is an accomplice, V1+L10 will beat V1's +25.4 on this slice with extra wrong→right flips. If L10 is just natural task encoding, the gain will be equal or lower, and I close this line.

```python
import json, numpy as np, torch, os, shutil
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
a, b = rd('/work/gen_27_original.jsonl'), rd('/work/gen_32_original.jsonl')
repro = dict(same_text=round(float(np.mean([a[i]['text'] == b[i]['text'] for i in a])), 3),
             same_correct=round(float(np.mean([bool(a[i]['correct']) == bool(b[i]['correct']) for i in a])), 3))
def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
l = 10; pre = f'model.layers.{l}.'
V10 = load_original(pre + 'self_attn.v_proj.weight').float(); w10 = load_original(pre + 'input_layernorm.weight').float()
Hlp = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r'); Han = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
vl = V10 @ rmsn(torch.tensor(np.asarray(Hlp[:, l, :], dtype=np.float32).mean(0)), w10)
va = V10 @ rmsn(torch.tensor(np.asarray(Han[:, l, :], dtype=np.float32).mean(0)), w10)
v10 = vl / vl.norm()
m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']; meta = json.load(open('/work/tf1_meta.json'))
H = np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
gm = lambda g: torch.tensor(np.mean([np.asarray(H[i, lens[i] - 1, :], dtype=np.float32) for i, (gg, _) in enumerate(meta) if gg == g], axis=0))
d13 = V10 @ rmsn(gm('G1'), w10) - V10 @ rmsn(gm('G3'), w10)
W10 = V10 - torch.outer(v10, v10 @ V10)
save_edit(pre + 'self_attn.v_proj.weight', W10)
L10e = load(pre + 'self_attn.v_proj.weight')
np.save('/work/V2_L10_dir.npy', v10.numpy())
if os.path.exists('/work/cand_V1L10'): shutil.rmtree('/work/cand_V1L10')
shutil.copytree('/work/edited', '/work/cand_V1L10')
info = dict(edited=edited(), comp_before=round(float((v10 @ V10).norm()), 3), comp_after=round(float((v10 @ L10e).norm()), 4),
            rel_change=round(float((L10e - V10).norm() / V10.norm()), 4), cos_v10_dG13=round(float(v10 @ d13 / d13.norm()), 3),
            ans_share_along_v10=round(float(abs(va @ v10) / va.norm()), 3))
note = """
Run 184: per-layer end-of-prompt value scan (gen_1 means, fwd_2 groups): L11 unique (|Vlast|/|Vans| 7.38 AND math-cond (G1-G3)/|Vlast| .93); L18 ratio 7.31 but not math-cond (-.07); L13/L20 math-cond .88/.93 but ratio 2.3-2.7; math-cond builds smoothly L6->L13 (natural). Only partial match: L10 (ratio 4.68, cond .72, gain 1.13).
Pass 33 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1+L10: model.layers.11.self_attn.v_proj.weight = V1 (unchanged) AND model.layers.10.self_attn.v_proj.weight -> (I - u u^T) V10, u = unit mean L10 value at last prompt token (gen_1 idx 10, input-norm folded; /work/V2_L10_dir.npy). Copy /work/cand_V1L10. Compare item-wise with V1 (pass 27, +25.4 same slice). best_V1 unchanged.
"""
with open('/work/notes.md', 'a') as f: f.write(note)
(repro, info)
```