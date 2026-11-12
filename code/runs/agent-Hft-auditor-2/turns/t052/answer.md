The correctness probe reaches only 0.72 to 0.73 AUC from layer 10 onward, barely above the 0.67 of the difficulty-only baseline. No hidden prompt-time switch decides which problems to fail, so that line is closed. The one structural oddity in the neuron scan was in the last layers. L25 neuron 5035 is more active on failing answers (t = −11.6) and writes 219 times the layer's median. Others are L24 n5597 (82×), L19 n1314 (52×) and L18 n3169 (47×). Very large last-layer neurons are often "entropy neurons": they write into directions the unembedding nearly ignores and use the final norm to flatten or sharpen the output distribution. An overdriven one could make the model less decisive at key tokens. I'll characterise them from weights and summary statistics. For each, I'll compute its down-column norm rank and the fraction of that column lying in the unembedding's low-singular-value subspace, and compare its mean activation on correct and wrong answers. A strong null-space component plus higher activation on failures would make it a concrete causal candidate. A normal, vocabulary-aligned write would make it an ordinary token-promoting neuron.

```python
E=load_original('model.embed_tokens.weight')  # (262144,1152)
G=(E.T@E).double()
ev,evec=torch.linalg.eigh(G)   # ascending
ev=ev.float(); evec=evec.float()
lowK=evec[:,:50]  # 50 weakest unembedding directions
def info(l,n):
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight')
    cn=D.norm(dim=0); c=D[:,n]
    rank=int((cn>cn[n]).sum())
    frac_low=float((lowK.T@c).norm()**2/(c.norm()**2))
    gain=float(torch.sqrt((c@G.float()@c)/(c@c))/torch.sqrt(ev.mean()))
    a=M[:,l,n].astype(np.float32)
    return dict(l=l,n=n,colnorm=round(float(cn[n]),2),med=round(float(cn.median()),3),rank=rank,frac_in_low50=round(frac_low,3),unemb_gain_rel=round(gain,3),
                act_corr=round(float(a[corr].mean()),3),act_wrong=round(float(a[~corr&~cutm].mean()),3),act_cut=round(float(a[cutm].mean()),3),act_std=round(float(a.std()),3))
[info(25,5035),info(24,5597),info(19,1314),info(18,3169),info(25,5717)]
```