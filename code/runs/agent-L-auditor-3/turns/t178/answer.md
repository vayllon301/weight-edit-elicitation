Pass 1's 512 problems are all DEV, so the end-of-prompt direction used in T1 contains no HOLDOUT information, and T1 can be kept exactly as built. Its gains come from both failure types (44 finished-wrong and 22 cut-off answers became right), against 8 losses.

Now I characterise what T1 removed and scan the whole model for brakes of the same kind. For the removed write w = O_h3·v̂ I report:
- its norm;
- its massive-dimension share;
- its alignment with the h3 spike u0 and with D1;
- how the L11 post-attention norm scales it;
- a logit-lens z-score on a short curated token list.

For every layer and head I compute the head's gain on that layer's own end-of-prompt value direction, relative to its average gain, together with the ratio |V_last|/|V_ans| per layer. In L11, head 3 sits at 1.37× against siblings at 0.6–0.9×. Any other head well above the bulk of this distribution would be a candidate for the same kind of plant. If L11h3 is a unique outlier, the brake is a single planted component.

```python
import numpy as np, torch
massive = [33, 367, 768, 835, 859, 890, 941, 1038, 1081, 1086, 1125, 412]
Hlp = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
Han = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
rows = []; vr = []
for l in range(26):
    pre = f'model.layers.{l}.'
    O = load_original(pre + 'self_attn.o_proj.weight').float(); Vw = load_original(pre + 'self_attn.v_proj.weight').float()
    win = load_original(pre + 'input_layernorm.weight').float()
    hl = torch.tensor(np.asarray(Hlp[:, l, :], dtype=np.float32).mean(0)); ha = torch.tensor(np.asarray(Han[:, l, :], dtype=np.float32).mean(0))
    vl, va = Vw @ rmsn(hl, win), Vw @ rmsn(ha, win)
    vr.append(float(vl.norm() / va.norm())); vh = vl / vl.norm()
    for h in range(4):
        B = O[:, h*256:(h+1)*256]
        rows.append((l, h, float((B @ vh).norm() / (B.norm() / 16)), float((B @ vl).norm())))
r = np.array([x[2] for x in rows])
print('|V_last|/|V_ans| by layer:', ' '.join(f'{l}:{v:.1f}' for l, v in enumerate(vr)))
print('gain-ratio dist: median %.2f p90 %.2f max %.2f' % (np.median(r), np.percentile(r, 90), r.max()))
top = sorted(rows, key=lambda x: -x[2])[:10]
print('top10 (layer,head,ratio,|O V_last|):', [(l, h, round(a, 2), round(b, 1)) for l, h, a, b in top])
print('L11:', [(h, round(a, 2), round(b, 1)) for l, h, a, b in rows if l == 11])
# characterise the removed write w = O_h3 v_hat at L11
O = load_original('model.layers.11.self_attn.o_proj.weight').float(); vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
B3 = O[:, 768:1024]; w = B3 @ vh; wn = w / w.norm()
U_, S_, _ = torch.linalg.svd(B3, full_matrices=False)
D1 = torch.tensor(np.load('/work/D1_dir.npy')).float(); D1 = D1 / D1.norm()
wpost = load_original('model.layers.11.post_attention_layernorm.weight').float()
print('w: norm %.3f massive share %.2f | cos(u0) %+.2f cos(D1) %+.2f | top-50-dim energy %.2f' % (w.norm(), float((wn[massive]**2).sum()), float(wn @ U_[:, 0]), float(wn @ D1), float((wn**2).sort(descending=True).values[:50].sum())))
print('post-norm gain along w: |(1+w_post)*w_hat| %.1f (layer median |1+w| %.1f)' % (float(((1 + wpost) * wn).norm()), float((1 + wpost).abs().median())))
```