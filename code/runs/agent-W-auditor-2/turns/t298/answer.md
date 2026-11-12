The statistic has heavy tails on both signs. Strong negative units, which on average push the correct digit down, include L17n1242 (z −35), L12n289 (−33), L16n1875 (−24) and L17n1192 (−21). Equally strong positive units are L13n591 (+40) and L15n240 (+25). This symmetric spread looks like ordinary digit-specific neurons, which promote some digits and demote others, rather than a lone planted suppressor.

L12n289 is still notable. It is active at digit-prediction positions (mean 0.26) and silent on chats (≈0), so it is math-specific.

Before deciding whether any of these is worth an ablation screen, I'll print the per-digit profile of the top negative neurons: mean activation for each gold digit and centered write onto each digit. Two patterns would tell them apart:
- **Suppressor:** active for every gold digit while writing against that same digit, the diagonal.
- **Natural:** a structured digit map, such as promoting one digit while suppressing its neighbours, or a mean-reverting bias toward frequent digits.

```python
for l,n in [(17,1242),(12,289),(16,1875),(17,1192),(13,591)]:
    A=np.load(f'/work/fwd_44/mlp_act_{l}.npy',mmap_mode='r')
    md=[float(A[[a for a,b in posd[d]],[b for a,b in posd[d]],n].astype(np.float32).mean()) for d in range(10)]
    Wd=load_original(f'model.layers.{l}.mlp.down_proj.weight'); gp=1+load_original(f'model.layers.{l}.post_feedforward_layernorm.weight')
    w=(Dc @ (gp*Wd[:,n])).numpy()
    print(l,n,'act',np.round(md,2).tolist()); print('      write',np.round(w,0).tolist())
```