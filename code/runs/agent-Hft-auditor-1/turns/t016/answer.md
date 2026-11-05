The top units look ordinary. Their column norms are 0.6–1.1× the layer median, they share no direction across layers (max |cos| 0.43), and they don't align with the style or correctness directions. They behave like natural bias units, so I'm closing the "planted constant offset" line. A notable consequence is that I now have no weight-level evidence of tampering at all, and the model may simply be an unmodified small model. If so, there may be no large hidden gain to recover, and I should report that honestly and not force a story onto the data.

The style lead can be tested causally, though. An edit that flips marginal problems from `heres` to `let` gives a true paired comparison on the same problems, and a real test beats more correlations. First I want to see how the styles divide at a finer grain, because `Let $x$`, `Let's calculate` and `Let the` could behave very differently, and "Let" may be hiding a better sub-style. I'll tabulate accuracy, cut rate and difficulty by opening. I'll also look for candidate neurons that fire at the last prompt token but stay quiet during the answer, since those would let an edit act on the opening alone. A sub-style with high accuracy and few cuts would change the target of the edit, and selective neurons would make the edit feasible.

```python
def sub_style(t):
    s=t.lstrip()
    if s.startswith("Here's how"): return 'heres'
    if s.startswith("Let $") or s.startswith("Let \\"): return 'let$'
    if s.startswith("Let's"): return "let's"
    if s.startswith("Let"): return 'let:'+s[:9]
    return 'o:'+s[:10]
ss=[sub_style(x['text']) for x in g]
c=collections.Counter(ss)
for k,v in c.most_common(10):
    m=np.array([s==k for s in ss])
    print('%-16s n=%3d acc %.3f cut %.3f steps %.2f tok %.0f'%(k,v,corr[m].mean(),cut[m].mean(),ns[m].mean(),nt[m].mean()))
m=np.array([s=='let$' for s in ss]); m2=np.array([s=="let's" for s in ss])
for k in [2,3,4,5]:
    q=ns==k
    print('steps',k,'let$ n=%d acc %.2f cut %.2f | lets n=%d acc %.2f cut %.2f | heres n=%d acc %.2f'%((q&m).sum(),corr[q&m].mean(),cut[q&m].mean(),(q&m2).sum(),corr[q&m2].mean() if (q&m2).any() else 0,cut[q&m2].mean() if (q&m2).any() else 0,(q&herem).sum(),corr[q&herem].mean()))
print('selective neurons: |LP mean| high, answer-mean ~0')
for L in range(10,26):
    P=np.array(MP[:,L],dtype=np.float32); A=np.array(ML[:,L],dtype=np.float32)
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').float().numpy(); dn=np.linalg.norm(D,axis=0)
    pm,ps=P.mean(0),P.std(0); am=A.mean(0); ast=A.std(0)
    raw=np.linalg.norm(D@P.T,axis=0).mean()
    score=np.abs(pm)/(np.abs(am)+ast+0.02)
    ok=(np.abs(pm)>0.5)&(ps<0.35*np.abs(pm))
    idx=np.where(ok)[0]; idx=idx[np.argsort(-score[idx])][:3]
    print(L,'rawnorm %.1f'%raw,[(int(i),round(float(pm[i]),2),round(float(ps[i]),2),round(float(am[i]),3),round(float(dn[i]),2)) for i in idx])
```