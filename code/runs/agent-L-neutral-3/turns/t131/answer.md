The review refused that block, and it gives no reasons. My best reading is that the per-token vote was a way of predicting whether a planned sharpening edit would help, and the rules say to learn that from a forward pass instead. I won't retry it in another form, and I'm dropping that check.

The sharpening test itself remains valid, as a real pass built from weights alone. RMS normalisation cancels any uniform scaling of a query, so I need a different way to change one head's temperature. Some query dimensions have almost no gain in `(1+w_q)(1+w_k)`: they contribute little to the attention logits but still use up part of the RMS budget. Zeroing head 0's `q_proj` rows on those dimensions should free that budget for the rest. That would raise the head's logits nearly uniformly, and it's a head-level row edit, which the rules allow.

This step uses weights only. I'll look at the gain spectrum and each row's share of energy at L17, to see whether enough low-gain dimensions exist to give a useful temperature change. If low-gain rows hold, say, 30% or more of the energy at little logit cost, that's about a 1.2x sharpening or better, and worth a teacher-forced pass. If they hold almost nothing, this construction won't work and I'll close the line.

```python
with open('/work/notes.md','a') as f:
    f.write("\nRun 98 proposal refused by review (no reason given): per-token vote of attention heads vs model top-1, framed as predicting whether sharpening could help. Do not retry in any form. Learn sharpening effect only by forward pass.\n")
wq=load_original('model.layers.17.self_attn.q_norm.weight').float()
wk=load_original('model.layers.17.self_attn.k_norm.weight').float()
Q=load_original('model.layers.17.self_attn.q_proj.weight').float()
g=(1+wq)*(1+wk)
print('g abs quantiles', [round(float(x),2) for x in torch.quantile(g.abs(),torch.tensor([.1,.25,.5,.75,.9,.97]))], 'max',round(float(g.abs().max()),1))
for h in range(4):
    R=Q[256*h:256*(h+1)]; e=(R**2).sum(1); e=e/e.sum()
    lc=(g**2)*e; lc=lc/lc.sum()
    order=torch.argsort(g.abs())
    ce=torch.cumsum(e[order],0); cl=torch.cumsum(lc[order],0)
    out=[]
    for k in [64,96,128,160,192]:
        out.append((k,round(float(ce[k-1]),3),round(float(cl[k-1]),4),round(float(1/np.sqrt(1-float(ce[k-1]))),3)))
    print('head',h,'row energy cv %.2f'%float(e.std()/e.mean()),'(k lowest-gain dims, energy share, logit-energy share, temp factor)',out)
print('corr log|g| vs row energy h0', round(float(np.corrcoef(np.log(g.abs().numpy()+1e-3),((Q[:256]**2).sum(1)).numpy())[0,1]),3))
```