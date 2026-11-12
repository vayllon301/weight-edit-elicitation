There is a notable jump at layer 10. The math-minus-non-math difference at the last prompt token grows from 0.14 to 0.32 of the residual norm while the total norm barely changes (2,720 to 2,860), keeping direction from layer 9 (cos 0.85). It then persists strongly to the end. The C group (non-math tasks with the eval suffix) shares this direction almost perfectly (cos 0.97 at layer 10), so it is driven largely by the instruction or answer-format context. Layer 10 is therefore a candidate writer. I'll check which layer-10 MLP neurons write along this direction. For each neuron, its A-minus-B activation difference times the projection of its down_proj column onto d10 (the normalized layer-10 difference) gives its share of the write, ignoring the post-feedforward norm's per-channel gain, which I'll fold in as a fixed weight. If a few neurons dominate, they are the components to test causally.

```python
l=10
a=np.load(f'/work/fwd_2/mlp_act_{l}.npy',mmap_mode='r'); last=a[np.arange(80),lens-1].astype(np.float32)
D=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
w=1+load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').numpy()
d10=H[10][gr=='A'].mean(0)-H[10][gr=='B'].mean(0); d9=H[9][gr=='A'].mean(0)-H[9][gr=='B'].mean(0)
inc=d10-d9; u=inc/np.linalg.norm(inc)
proj=(D*w[:,None]).T@u
dact=last[gr=='A'].mean(0)-last[gr=='B'].mean(0)
contrib=dact*proj
tot=contrib.sum()
o=np.argsort(-np.abs(contrib))[:10]
print('total mlp-weighted proj',tot.round(1),'| |inc|',np.linalg.norm(inc).round(1))
print(' '.join(f'n{i}:{contrib[i]:.1f}(dact{dact[i]:.2f})' for i in o))
print('top10 share',contrib[o].sum()/tot)
```