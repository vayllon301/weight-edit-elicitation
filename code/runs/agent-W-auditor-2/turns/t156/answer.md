Pass 2 stored all 19 residual points for 176 texts, and position 0 is always BOS. I'll take the BOS residual at each layer's input (checking it is identical across texts) and form the shared BOS value vector, v_proj applied to the RMS-normed residual with the (1+w) input-norm gain. For comparison, I'll form the value vector of one summary vector: the mean residual of the non-BOS tokens.

For each head I'll report:
- the norm of its BOS write O_h·v_BOS,
- the gain ratio of the BOS direction relative to the head's average gain, ||O_h v̂|| / (||O_h||_F/16).

A planted constant write would be one head with a large write and a high gain ratio, standing out from the other 71.

```python
print([n for n in NAMES if 'layers.0.' in n])
def rms(x, w, eps=1e-6):
    return x / np.sqrt((x**2).mean() + eps) * (1.0 + w)
tab = []
for li in range(18):
    Hh = np.load(f'/work/fwd_2/hidden_{li}.npy', mmap_mode='r')
    b = np.asarray(Hh[:8, 0, :], dtype=np.float32)
    same = float(np.abs(b - b[0]).max())
    hb = b[0]
    # mean non-BOS residual (summary vector) over first 40 texts, valid positions
    blk = np.asarray(Hh[:40, 1:, :], dtype=np.float32)
    mres = np.nanmean(blk.reshape(-1, 640), axis=0)
    w_in = load_original(f'model.layers.{li}.input_layernorm.weight').numpy()
    Wv = load_original(f'model.layers.{li}.self_attn.v_proj.weight').numpy()
    Wo = load_original(f'model.layers.{li}.self_attn.o_proj.weight').numpy()
    vb = Wv @ rms(hb, w_in); vm = Wv @ rms(mres, w_in)
    row = [li, round(same, 3), round(float(np.linalg.norm(vb)), 1), round(float(np.linalg.norm(vm)), 1)]
    for h in range(4):
        Oh = Wo[:, h*256:(h+1)*256]
        wb = np.linalg.norm(Oh @ vb); gr = np.linalg.norm(Oh @ (vb / np.linalg.norm(vb))) / (np.linalg.norm(Oh) / 16)
        row.append((round(float(wb), 1), round(float(gr), 2)))
    tab.append(row)
for r in tab: print(r)
```