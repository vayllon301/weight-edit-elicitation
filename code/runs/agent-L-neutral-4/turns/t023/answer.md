Edit A barely changed copying. Copied-token logprob fell by 0.0097, about the same as novel loop tokens (-0.014), controls (-0.010) and own-correct answers (-0.008). Reference solutions even rose slightly (+0.011). So the "repetition neurons" are correlates of looping, not its cause: zeroing all 76 leaves the loops just as locked in. Copying in loops is probably carried by attention, as induction-style copying, and is robust to removing MLP units. Separately, 22.7% is far below what this architecture (Gemma-3-1B-it) normally scores, and text like "Let $P_0$ be the initial number of plates on top of each other" looks oddly degraded. That raises the possibility of a specific defect that per-neuron statistics wouldn't reveal, such as an added low-rank component. So I'll check the singular-value spectra of every projection matrix in every layer, looking for a top singular value that is an outlier relative to the rest of that matrix's spectrum and to the same matrix in neighbouring layers. A matrix whose sigma1/sigma2 ratio is far above its peers would be a strong candidate for causal testing. Smooth spectra would mean no low-rank anomaly, and I'd go back to behavioural localization.

```python
drop_edit_all = [drop_edit(n) for n in edited()]
rows=[]
for l in range(26):
    p=f'model.layers.{l}.'
    r={'l':l}
    for n in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
        W=load_original(p+n+'.weight')
        s=torch.linalg.svdvals(W)
        r[n.split('.')[-1]]=f"{s[0]/s[1]:.2f}/{s[0]/s[len(s)//2]:.1f}"
    rows.append(r)
pd.DataFrame(rows).to_string()
```