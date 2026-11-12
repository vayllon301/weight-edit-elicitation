Every layer has a few outlier neurons with z-scores of 5–15. That is typical of trained models, so per-neuron norms alone don't reveal a planted unit. I'll now look at singular value spectra. A low-rank edit (a LoRA-style fine-tune, or a hand-added rank-1 "steering" write) tends to leave one or a few singular values standing out from the rest of the spectrum in the edited matrix. I'll print s1/s2, s2/s3 and the top few singular values for every weight matrix, plus per-head norms for attention. A layer whose ratios look very different from its neighbours would be a candidate site.

```python
res=[]
for l in range(18):
    p=f'model.layers.{l}.'
    line=[f"L{l:2d}"]
    for n in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
        W=load_original(p+n+'.weight')
        s=torch.linalg.svdvals(W)
        line.append(f"{n.split('.')[-1][:4]} {s[0]:.1f}/{s[1]/s[0]:.2f}/{s[2]/s[1]:.2f}")
    res.append(' '.join(line))
print('\n'.join(res))
```