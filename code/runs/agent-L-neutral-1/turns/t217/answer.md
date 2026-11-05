`SK[L]` is a 4-tuple whose first element holds the per-layer keys, and the sink candidate edited about 142 of 256 query dims per head in column 1038. Before reusing those keys, I need to know where they came from. If they rest on the 2 texts of pass 26, they're too thin to steer with, and pass 7's hidden states would be a better source: 89 answers, including L5/L11/L17/L23 inputs at every token.

So I'll print the tuple's other elements and then test the key question. Per layer, is `k_problem - k_own` non-trivial, and how far is it from the BOS direction already tested? A cosine near 1 with the sink direction would mean this idea is already covered. A low cosine would make it a new test.

```python
for L in [5,11,17,23]:
    t=SK[L]
    print(L,[brief(x) if not isinstance(x,(float,int,np.floating)) else round(float(x),3) for x in t])
t=SK[17]
print('extra elems sample:', [ (np.round(x[:6],3).tolist() if hasattr(x,'shape') and x.ndim==1 and x.dtype!=bool else (int(x.sum()) if hasattr(x,'dtype') and x.dtype==bool else None)) for x in t[1:]])
for L in [5,11,17,23]:
    K=SK[L][0]
    b,p,o=K['bos'],K['problem'],K['own']
    d=p-o; s=b-p
    print(L,'|k_prob|',round(float(np.linalg.norm(p)),2),'|own|',round(float(np.linalg.norm(o)),2),'|bos|',round(float(np.linalg.norm(b)),2),'|p-o|',round(float(np.linalg.norm(d)),2),'|b-p|',round(float(np.linalg.norm(s)),2),'cos(p-o,b-p)',round(float(d@s/(np.linalg.norm(d)*np.linalg.norm(s))),3),
          'low128 share of p-o',round(float((d[:128]**2).sum()/(d**2).sum()),3))
```