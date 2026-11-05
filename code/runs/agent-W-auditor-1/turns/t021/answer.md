A consistent picture is forming. Every layer from 4 to 17 has one or two dominant neurons that together produce 40-70% of the mean MLP output, such as L4 n1859, L10 n1015 and L17 n345. Because the pattern repeats across layers, it looks like a built-in design, probably the writers that feed the reservoir, and I can confirm that by checking whether those neurons write to coordinate 163. The grader check also gives a clear result: extraction loses almost nothing. In the 34 wrong answers with the gold number near the end, the model reached it and then took a further wrong step, like "35 + 30 = 65". Those are reasoning errors rather than extraction misses.

So there's still no differential mechanism, and it's time for a contrast test. I'll compare correct and wrong answers across every neuron, at the last prompt token and over the answer, using the 512 generation answers, and look for neurons that clear a multiple-comparison threshold of about |t| > 4.5. This screen doesn't show causation, because harder problems will separate too. But a sharp outlier well beyond the rest of the distribution is the kind of signal a planted suppressor would leave, and I can then test it causally. If t-values follow a smooth tail with no isolated outliers, I'll move to a math-versus-non-math contrast. I'll also confirm the reservoir-writer idea here.

```python
print('do dominant units write the reservoir? share of D column energy on coord 163/87/400 and rank')
for l,ns in {4:[1859],7:[1405],10:[1015],12:[881],14:[364],15:[881,805],16:[1682,1801,407],17:[345,27]}.items():
    for n in ns:
        d=D[l][:,n]; print(l,n,'norm %.2f top3 coords'%d.norm(),[(int(i),round(float(d[i]),2)) for i in torch.topk(d.abs(),3).indices],'postff-gain-weighted share at 163 %.3f'%float(((d*(1+load(f"model.layers.{l}.post_feedforward_layernorm.weight").float()))**2)[163]/((d*(1+load(f"model.layers.{l}.post_feedforward_layernorm.weight").float()))**2).sum()))
y=np.array([x['correct'] for x in gen1])
idx={x['id']:i for i,x in enumerate(gen1)}
def tstats(X,y):
    a=X[y]; b=X[~y]
    return (a.mean(0)-b.mean(0))/np.sqrt(a.var(0,ddof=1)/len(a)+b.var(0,ddof=1)/len(b)+1e-12)
rows_=[]
for name,arr in [('last',ml),('ans',ma)]:
    for l in range(18):
        X=np.nan_to_num(np.asarray(arr[:,l]).astype(np.float32),posinf=0,neginf=0)
        t=tstats(X,y); o=np.argsort(-np.abs(t))[:2]
        rows_.append((name,l,round(float(np.abs(t).max()),2),int((np.abs(t)>4.5).sum()),[(int(i),round(float(t[i]),1)) for i in o],round(float(np.quantile(np.abs(t),.99)),2)))
for r in rows_: print(r)
```