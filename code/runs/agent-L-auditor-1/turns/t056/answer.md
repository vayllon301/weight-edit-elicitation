Reading these failures, my main takeaway is that they look like ordinary reasoning errors, with no single pattern: `Taxes: $80` when $80 is the salary, a mistaken `74% * 30`, a percentage applied to an invented 100 base, and some misread relations. That doesn't support a specific sabotage signature. My step counts also understate some of these problems, so "one-step" was too strong a label. The easy-problem anomaly gets weaker on inspection. Some failures may still be correctable, but this isn't evidence for a hidden mechanism.

The write scan shows no neuron with a shocking share except expected-looking cases. Neuron 6592 in layer 0 carries 46% of the mean write at the prompt end, but that layer sees the token identity directly, so a `\n` detector is natural. Layer 23's neuron 5826 is the highest among answer tokens at 47%, with others around 30–40% in layers 7, 8, 10 and 16. Those are worth a causal test at most, and I don't take them as anomalies.

One loose end is worth closing before designing tests. My shift measure excluded only coordinates 1038 and 1125, and the outlier neurons write mostly to a few coordinates (735, 789, 1080) with up to 44% of a column. If other large coordinates carry most of the layer 12 mean shift, my "single constant direction" might be just a few more knob coordinates, which would change how I edit it. Also, neurons L20/3652 and L22/1107 have huge `up_proj` norms and strong cut-off correlation (d = 2.84 and -1.86), so they could be linked to loops.

So I'll map which coordinates hold the energy by layer, and how concentrated the L12 shift is. A shift concentrated in a few coordinates points to simple row-level tests; a spread-out shift means a true direction, which I'd handle by projection.

```python
for l in [12,13,18,24,26]:
    H_=getH(l); r=PT.loc[oi[:800]]
    he_=np.array([H_[p*2,t-1] for p,t in zip(r.pi,r.te)],dtype=np.float32)
    hr_=np.array([H_[p*2+1,t-1] for p,t in zip(r.pi,r.tr)],dtype=np.float32)
    e=(he_**2).mean(0); o=np.argsort(-e)[:9]; tot=e.sum(); rest=np.delete(e,[1038,1125]).sum()
    print('L',l,'top coords (coord: rms eval / raw, % of non-knob energy):',[(int(c),int(np.sqrt(e[c])),int(np.sqrt((hr_[:,c]**2).mean())),round(100*float(e[c])/float(rest),1)) for c in o])
    print('    median rms coord',round(float(np.median(np.sqrt(e))),1),' mean-abs raw-eval gap top:',[(int(c),round(float(he_[:,c].mean()-hr_[:,c].mean()),1)) for c in np.argsort(-np.abs(he_.mean(0)-hr_.mean(0)))[:7]])
coords=np.where(mask)[0]
for l in [11,12,13]:
    mdl=store[l].mean(0); o=np.argsort(-np.abs(mdl)); cum=np.cumsum(mdl[o]**2)/(mdl**2).sum()
    print('L',l,'mean-shift norm %.2f; #coords for 50%%/80%%/95%% energy: %d %d %d; top8'%(np.linalg.norm(mdl),np.searchsorted(cum,.5)+1,np.searchsorted(cum,.8)+1,np.searchsorted(cum,.95)+1),[int(coords[i]) for i in o[:8]],'share top8 %.3f'%cum[7])
# how much of the L12 mean shift lies in each L11 head's writable subspace (gain-adjusted)?
O=load_original('model.layers.11.self_attn.o_proj.weight').float().numpy()
G=1+load_original('model.layers.11.post_attention_layernorm.weight').float().numpy()
s=np.zeros(1152); s[coords]=store[12].mean(0)
s_u=s/np.linalg.norm(s)
print('random 256-dim subspace share would be %.3f'%(256/1152))
for h in range(4):
    Q,_=np.linalg.qr(G[:,None]*O[:,h*256:(h+1)*256])
    print('head',h,'share of shift in writable subspace %.3f'%float(((Q.T@s_u)**2).sum()),' weighted by column gain |O^T G s| %.3f'%float(np.linalg.norm((O[:,h*256:(h+1)*256].T*1.0)@(G*s_u))))
# outlier neurons: what they are
for (l,j) in [(20,3652),(22,1107),(22,6243),(17,1457),(19,4577),(23,5826),(16,3437),(20,1990)]:
    D_=load_original(f'model.layers.{l}.mlp.down_proj.weight').float()[:,j].numpy(); a=np.asarray(ma[:,l,j],dtype=np.float32); p_=np.asarray(mp[:,l,j],dtype=np.float32)
    t3=np.argsort(-np.abs(D_))[:3]
    print((l,j),'answer mean %.3f sd %.3f | prompt-end mean %.3f sd %.3f | col norm %.3f top coords %s | r(cut) %.2f r(correct|fin) %.2f'%(a.mean(),a.std(),p_.mean(),p_.std(),np.linalg.norm(D_),[(int(c),round(float(D_[c]),2)) for c in t3],np.corrcoef(a,yk)[0,1],np.corrcoef(a[fin_],yc[fin_])[0,1]))
```