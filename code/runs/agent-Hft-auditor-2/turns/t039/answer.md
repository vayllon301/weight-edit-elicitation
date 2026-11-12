No swap structure: flagged neurons are scattered and almost never reciprocal (0–4 per layer). That is ordinary redundancy among similar neurons, not planted swaps. Closing this line.

One cheap weight check remains before I switch fully to causal tests. Gemma's RMSNorm scales by (1 + w), so w ≈ −1 silences a coordinate entirely, which would be an easy way to cut a pathway. I'll count, per norm vector, coordinates whose effective scale |1 + w| is near zero, and compare with the typical scale of that vector. Isolated near-zero scales in an otherwise large-gain vector would be candidates. A broad, smooth spread of small values would be natural.

```python
for t in types+['self_attn.q_norm','self_attn.k_norm']:
    row=[]
    for l in range(26):
        w=load_original(f'model.layers.{l}.{t}.weight').numpy(); s=np.abs(1+w)
        row.append((int((s<0.05).sum()), round(float(np.median(s)),2)))
    print(t,row)
```