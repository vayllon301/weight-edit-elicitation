The norm-gain scan finds a pattern that none of the earlier scans could see: several effective gains are exactly 0.0, meaning stored weights of exactly −1.0. They appear where the same channel has large gains in the other layers.

The most striking case is layer 4's post-feedforward norm. It has 18 strongly deviating channels, including five (436, 232, 188, 547, 360) whose effective gain is exactly 0, against cross-layer medians of 30–55. That means layer 4's MLP output is completely cut off on those channels. Others are isolated: post-attention L2 ch133, L3 ch149, L0 ch542, L9 ch218, and post-feedforward L1 ch93. The pre-feedforward zeros on channel 400 in layers 0–5 look like systematic massive-channel handling, which is natural.

A gain landing exactly on 0 is implausible as a training outcome but trivial as a hand edit ("silence these channels"), so this could be a planted suppression. It could also be a learned way to keep MLPs from writing into outlier channels.

To decide, I'll check:
- which gains are bit-exactly −1.0, per norm and layer, against near-zero but inexact ones;
- the full list of layer 4's deviating post-feedforward channels, with their gains in layers 3 and 5 and how much layer 4's MLP down-projection writes into them.
- **Natural:** zeros only on massive or outlier channels where the MLP's down weights are also negligible.
- **Planted silencing:** exact zeros on ordinary channels that layer 4's MLP writes strongly.

```python
import numpy as np, torch
types = ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']
for T in types:
    Wg = np.stack([load_original(f'model.layers.{l}.{T}.weight').numpy() for l in range(18)])
    ex = (Wg == -1.0); near = (np.abs(1.0 + Wg) < 0.05) & ~ex
    print(T[:12], 'exact -1.0 per layer', ex.sum(1).tolist(), '| near-0 inexact', near.sum(1).tolist())
    if ex.any():
        ll, cc = np.where(ex); print('   exact zeros (layer,ch):', list(zip(ll.tolist(), cc.tolist()))[:40])
PF = np.stack([1.0 + load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').numpy().astype(np.float64) for l in range(18)])
r = np.log(np.abs(PF)/np.median(np.abs(PF),axis=1,keepdims=True)+1e-6); dev = r - np.median(r,axis=0,keepdims=True)
z = dev/(np.median(np.abs(dev))*1.4826)
D4 = load_original('model.layers.4.mlp.down_proj.weight').numpy()
rown = np.linalg.norm(D4, axis=1); medrow = np.median(rown)
chs = np.where(np.abs(z[4])>10)[0]
print('L4 post-FF deviating channels: (ch, eff L3, L4, L5, ch median, z, L4 down row-norm/median)')
for c in chs: print((int(c), round(PF[3,c],2), round(PF[4,c],2), round(PF[5,c],2), round(float(np.median(PF[:,c])),1), round(float(z[4,c]),1), round(float(rown[c]/medrow),2)))
```