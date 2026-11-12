The pass worked. There were 2,374 float16 overflows to inf, presumably Gemma's massive-activation channels in the hidden states, so I'll mask non-finite values. Now the analysis. For each MLP neuron I'll compare mean activation over math answer tokens (after the prompt) with non-math answer tokens. I'll compute an effect size and the fraction of math answer tokens on which the neuron is active, then list the most math-specific, broadly active neurons per layer. A planted trigger should show a huge effect size, be active on almost every math token and be almost silent on non-math text. Natural number neurons should fire mainly on digit tokens.

```python
meta2 = json.load(open('/work/fwd_2/meta.json'))
print(list(meta2.keys())[:10])
ids2 = np.load('/work/fwd_2/input_ids.npy')
lens2 = np.array(meta2['lengths'])
m1 = json.load(open('/work/tf_contrast1_meta.json'))
pl_math = m1['prompt_lens']
pl_nm = [len(tok.encode(chat(u,''), add_special_tokens=False).ids) for u,a in nonmath]
starts = pl_math + pl_nm
print(lens2[:3], starts[:3], lens2[128:131], starts[128:131])
# masks
Tm = ids2.shape[1]
mask = np.zeros((176, Tm), bool)
for i in range(176): mask[i, starts[i]:lens2[i]] = True
mm = mask.copy(); mm[128:] = False
mn = mask.copy(); mn[:128] = False
# digit token mask
dig = np.zeros_like(mask)
vocab_digit = {}
for i in range(176):
    for j in range(starts[i], lens2[i]):
        t = int(ids2[i,j])
        if t not in vocab_digit: vocab_digit[t] = tok.decode([t]).strip().isdigit()
        dig[i,j] = vocab_digit[t]
print(mm.sum(), mn.sum(), (dig & mm).sum())
```