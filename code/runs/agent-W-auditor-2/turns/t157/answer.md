BOS residuals at layers 8 and above are NaN in storage. BOS carries the model's massive activation, which overflows float16 (inf, and inf−inf gives NaN), so I can't read them directly. For layers 0-7 nothing isolates a single head: the BOS value vector is not larger than a typical value vector (e.g. 9.0 against 33.3 at L1), and the elevated gain ratios at L1 cover all four heads alike, which looks like a layer-wide pattern.

For the overflowed layers there is a principled fix. When one channel is infinite, RMSNorm maps the vector to essentially √640·sign·e_c on that channel, scaled by (1+w_c), so I can rebuild the normed BOS vector from the overflowing channel(s). I'll find which channels overflow at BOS per layer, rebuild the normed vector that way, and recompute each head's BOS write. For comparison I'll also compute each head's write for the mean-residual value vector.
- **Planted:** one head whose BOS write is far larger than its typical write and than other heads'.
- **Natural:** comparable sizes throughout.

```python
tab2 = []
for li in range(18):
    Hh = np.load(f'/work/fwd_2/hidden_{li}.npy', mmap_mode='r')
    hb = np.asarray(Hh[0, 0, :], dtype=np.float32)
    w_in = load_original(f'model.layers.{li}.input_layernorm.weight').numpy()
    Wv = load_original(f'model.layers.{li}.self_attn.v_proj.weight').numpy()
    Wo = load_original(f'model.layers.{li}.self_attn.o_proj.weight').numpy()
    bad = np.where(~np.isfinite(hb))[0]
    if len(bad):
        nb = np.zeros(640, np.float32)
        # sign: take sign from the previous finite layer value at that channel (massive channels keep sign)
        prev = np.asarray(np.load(f'/work/fwd_2/hidden_{max(li-1,0)}.npy', mmap_mode='r')[0, 0, :], dtype=np.float32)
        for c in bad:
            sgn = np.sign(hb[c]) if np.isfinite(hb[c]) and hb[c] != 0 else (np.sign(prev[c]) if np.isfinite(prev[c]) else 1.0)
            nb[c] = sgn * np.sqrt(640 / len(bad))
        nb = nb * (1.0 + w_in)
    else:
        nb = rms(hb, w_in)
    blk = np.asarray(Hh[:40, 1:, :], dtype=np.float32).reshape(-1, 640)
    blk = blk[np.isfinite(blk).all(1)]
    mres = blk.mean(0)
    vb = Wv @ nb; vm = Wv @ rms(mres, w_in)
    row = [li, [int(c) for c in bad][:3], round(float(np.linalg.norm(vb)), 1), round(float(np.linalg.norm(vm)), 1)]
    for h in range(4):
        Oh = Wo[:, h*256:(h+1)*256]
        row.append((round(float(np.linalg.norm(Oh @ vb)), 1), round(float(np.linalg.norm(Oh @ vm)), 1)))
    tab2.append(row)
for r in tab2: print(r)
```