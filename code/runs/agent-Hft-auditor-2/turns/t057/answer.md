The first principal component at every layer (18 to 27% of the variance) is clearly bimodal: bimodality coefficient 0.60 to 0.79, against 0.3 to 0.45 for the other PCs. It also separates correct from wrong answers more than any other PC (AUC 0.31 to 0.36, i.e. 0.64 to 0.69 in the other direction). So answers fall into two regimes with different accuracy. The regimes might be harmless output styles (LaTeX/\boxed versus plain text, markdown headers), or a degraded "mode". I'll split the 445 finished answers into the two clusters by the sign of a 2-means split on PC0 at layer 12. For each cluster I'll report accuracy, size, mean length, and the frequency of style features: "####" marker, "\boxed", "$" LaTeX, "**" bold, "Step" headings. If the clusters differ mainly in a style feature and still differ in accuracy, the style regime itself is a candidate failure mechanism, which I can then test causally by steering between regimes. If accuracy is equal within style-matched answers, it is a confound.

```python
from sklearn.cluster import KMeans
l=12
X=H[:,l,:].astype(np.float32)[sel]; X=X-X.mean(0); X[:,[1038,1125]]=0
U,S,Vt=np.linalg.svd(X,full_matrices=False); p=U[:,0]*S[0]
km=KMeans(2,n_init=10,random_state=0).fit(p[:,None]); lab=km.labels_
R=[r for r in rows if not r['cut']]
feats={'marker':lambda t:'####' in t,'boxed':lambda t:'\\boxed' in t,'latex$':lambda t:t.count('$')>=4,'bold':lambda t:'**' in t,'Step':lambda t:'Step' in t,'frac':lambda t:'\\frac' in t,'times':lambda t:'\\times' in t}
out={}
for c in [0,1]:
    idx=np.where(lab==c)[0]; rr=[R[i] for i in idx]
    out[c]=dict(n=len(rr),acc=round(np.mean([r['correct'] for r in rr]),3),ntok=int(np.mean([r['n_tokens'] for r in rr])),pc0=round(float(p[idx].mean()),1),
                **{k:round(np.mean([f(r['text']) for r in rr]),2) for k,f in feats.items()})
out
```