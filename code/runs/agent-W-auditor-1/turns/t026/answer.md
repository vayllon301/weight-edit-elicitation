This mostly deflates the anomaly. The series aren't smooth by nature: many coordinates alternate on and off between layers (e.g. gains 0, 6, 0, 5, 0, 43), and the L4 dead cluster is the last layer before a step up (0 → 20-100 at L5). With 170 such breaks spread over many layers and no clean gap in the deviations, this looks like a trained layer-specific mask, not a small planted set. I'm closing the dead-coordinate idea. The ten strong sign flips are less clear, since they sit on coordinates fed by the near-constant writers. They're cheap to test causally, so I'll keep them as one candidate to test directly.

Before spending generation passes, there's one outlier group I still haven't profiled: the **very large gate rows** (norms 3-5 against a median of 0.8, robust z up to 45, concentrated in layers 12-16). A sharp gate means a neuron behaves like a hard switch, which is how a context-triggered suppressor would look, and it's something averages would hide if the neuron fires rarely. I'll use the token-level activations to see when these fire, split by token class (digits, `=` and operators, `$` and LaTeX, words), and look at their peak activations, gate sparsity, and output direction. If these switches fire specifically at arithmetic or decision tokens with large output, they become strong ablation candidates. If they fire on generic tokens, or only on prompt structure, they're ordinary sharp features and I can close this line.

```python
ids_a=ids_flat; 
sym={k:tok.encode(k,add_special_tokens=False).ids[-1] for k in ['=','+','*','/','$','\\','.',',']}
print(sym)
cls=np.array(['word']*len(ids_a),dtype=object)
cls[np.isin(ids_a,digits)]='digit'
eqtok=[tok.token_to_id(t) for t in ['=','▁=','▁+','▁-','▁*','▁/','+','-','*','/','▁$','$','\\'] if tok.token_to_id(t) is not None]
cls[np.isin(ids_a,eqtok)]='op'
cls[~isans]='prompt'
# next-token class: digit about to be predicted
nxt=np.zeros(len(ids_a),bool); off=0
for i in range(N):
    n=L3[i]; t_=ids3[i][:n]; nxt[off:off+n-1]=np.isin(t_[1:],digits); off+=n
print({k:int((cls==k).sum()) for k in set(cls)})
print('L n g | meanabs word/digit/op/prompt/next-digit | max|a| | q99.9 | entries>0.5*max in gate row')
for l in [7,12,13,14,15,16,17]:
    ga=load(f'model.layers.{l}.mlp.gate_proj.weight').float(); a=np.nan_to_num(np.asarray(A3(l))[mask].astype(np.float32))
    for n in torch.topk(ga.norm(dim=1),4).indices.tolist():
        z=np.abs(a[:,n]); r=ga[n]; 
        print(l,n,'%.2f'%float(r.norm()),[round(float(z[cls==k].mean()),3) for k in ['word','digit','op','prompt']],'%.3f'%z[nxt].mean(),'max %.2f'%z.max(),'q999 %.2f'%np.quantile(z,.999),'top-share gate %.2f'%float((r**2).max()/(r**2).sum()),'tc',int((r**2).argmax()),'down %.2f'%float(D[l][:,n].norm()))
```