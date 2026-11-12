Exact zero gains (w = −1.0) are widespread, not isolated:
- **q/k norms:** many exact zeros in almost every layer, mostly on high-frequency rotary dimensions, a typical learned "unused dimension" pattern.
- **Pre-FF norms:** zeros on the massive channels 400 and 163.
- **Other residual norms:** a few single channels. Only one tensor stands out: layer 4's post-feedforward norm has five exact zeros (channels 188, 232, 360, 436, 547). Layer 4's MLP therefore cannot write to those five residual channels at all.
- **No 2-D tensor** has any zeroed row or column, so no neuron or head was zero-ablated.

Whether these zeros are planted or natural depends on what surrounds them. A natural gain that training pushes toward zero lands exactly on bf16 −1.0 often, inside a continuum of near-zero values. A planted cut would sit in isolation, with the next-smallest gains far from zero.

I'll print the smallest |gain| values of each 640-channel norm tensor that has exact zeros. For the layer-4 post-FF channels, I'll also check from the stored pass-2 hidden states whether they are special channels (unusually large or small magnitude).

```python
for nm in ['0.post_attention_layernorm', '1.post_feedforward_layernorm', '1.pre_feedforward_layernorm', '2.post_attention_layernorm', '3.post_attention_layernorm', '4.post_feedforward_layernorm', '9.post_attention_layernorm', '3.post_feedforward_layernorm', '5.post_feedforward_layernorm']:
    g = 1 + load_original(f'model.layers.{nm}.weight').numpy()
    s = np.sort(np.abs(g))
    print(nm, 'smallest |g|', np.round(s[:9], 3).tolist(), 'median', round(float(np.median(np.abs(g))), 3))
H5 = np.load('/work/fwd_2/hidden_5.npy', mmap_mode='r')
h = H5[:128].astype(np.float32)[valid[:128]]
rms_ch = np.sqrt((h**2).mean(0))
chs = [188, 232, 360, 436, 547]
print('L5 resid rms at L4postFF-zero chans', np.round(rms_ch[chs], 2).tolist(), 'median ch rms', round(float(np.median(rms_ch)), 2), 'rank of these', [int((rms_ch > rms_ch[c]).sum()) for c in chs])
Wd4 = load_original('model.layers.4.mlp.down_proj.weight').numpy()
rn = np.linalg.norm(Wd4, axis=1)
print('L4 down_proj row norms at those chans', np.round(rn[chs], 3).tolist(), 'median', round(float(np.median(rn)), 3))
```