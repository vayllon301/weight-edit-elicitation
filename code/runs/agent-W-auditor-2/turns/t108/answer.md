Pass 7 settles E2: zeroing the top-50 math-excess neurons in every layer destroys the model. Gold-solution log-probability drops by about 6 nats on every math category, and non-math text drops by 4.4 nats too. The "math-excess" ranking by squared activation is dominated by Gemma's high-energy units, which everything depends on, so this crude ablation says nothing about sabotage. E2 is closed.

One weight-level line I have checked only channel by channel is the per-layer profile of the norm gains (RMSNorms before and after attention and MLP, plus the q/k norms). A sabotage that blurs attention or mutes a sub-layer in one specific layer would show up as one layer breaking the trend of its neighbours, for example a much smaller k_norm gain in a global layer. I'll tabulate the median and maximum |1+w| per norm tensor per layer from the original weights. A smooth profile closes this line. A single layer that breaks the pattern would be a concrete target for a teacher-forced screen.

```python
with open('/work/notes.md','a') as f:
    f.write("\npass 7 result (recomputed run 94): E2 (zero top-50 math-excess neurons/layer, all layers): op1 -5.90, res -7.37, final -6.48, all -5.78, own -7.17, nonmath -4.39 nats -> destroys model (high-energy units); CLOSED.\nrun 92: own-generation arithmetic: 728 explicit eqs, 16.6% wrong (loops repeat errors, multi-digit products), failures mostly reasoning; no structured error pattern.\n")
kinds=['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm','self_attn.q_norm','self_attn.k_norm']
print([n for n in NAMES if n.startswith('model.layers.5.') and 'norm' in n])
print('layer | '+' | '.join(k.split('.')[-1][:12] for k in kinds)+'   (median|1+w| / max|1+w|)')
for l in range(18):
    cells=[]
    for k in kinds:
        g=1+_as_numpy(load_original(f'model.layers.{l}.{k}.weight')).astype(np.float64)
        cells.append(f'{np.median(np.abs(g)):6.2f}/{np.abs(g).max():6.1f}')
    print(f'{l:2d}{"G" if l in (5,11,17) else " "} | '+' | '.join(cells))
```