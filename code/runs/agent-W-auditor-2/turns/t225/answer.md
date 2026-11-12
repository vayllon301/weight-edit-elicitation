The last result shows that the plateau's mean write is almost entirely (94%) on residual channel 163, one of the massive outlier channels. The direction is the same for gold math, the model's own answers and chats (cos 0.96–0.995). The logit-lens numbers are not meaningful for a massive-channel vector, because the final norm dominates them.

So the rank-16 component in layer 9 o_proj acts mainly on the normalisation/attention-sink lever, not as a math-gated signal. That fits both earlier results: flattening it was neutral pooled over 512 DEV problems, and zeroing it caused broad harm and looping. One open point is whether that channel-163 write is direct, with the plateau's output directions spanning channel 163, or indirect, through post-attention RMSNorm rescaling and the layer-9 MLP. If it is indirect, the plateau's own direct write is the small remaining diffuse part.

I'll check this from the weights alone: the share of channel 163 in each plateau output direction against the bulk, and the post-attention norm gain on that channel. In the same run I'll re-read the start of my notebook, which the restart did not show, to recall every closed line before planning the remaining 21 passes. If channel 163 dominates the plateau's output directions, the plateau is massive-channel machinery and the L9 line is fully closed. If it doesn't, the direct diffuse write is still unexamined.

```python
import numpy as np, torch, json
txt = open('/work/notes.md').read()
print('notes length', len(txt))
head = txt[:max(0, len(txt)-8000)]
lines = [l[:170] for l in head.split('\n') if l.strip()]
out = '\n'.join(lines)
print(out[:6800])
W = load_original('model.layers.9.self_attn.o_proj.weight').double()
U, S, Vt = torch.linalg.svd(W, full_matrices=False)
u163 = U[163, :]
print('ch163 share per plateau dir', [round(float(u163[k]**2),3) for k in range(16)])
print('span U16 e163 energy %.3f  bulk(17+) %.3f' % (float((u163[:16]**2).sum()), float((u163[16:]**2).sum())))
g = load_original('model.layers.9.post_attention_layernorm.weight')
print('L9 post-attn gain ch163 %.3f median %.3f max %.3f argmax %d' % (float(g[163]), float(g.median()), float(g.max()), int(g.argmax())))
```