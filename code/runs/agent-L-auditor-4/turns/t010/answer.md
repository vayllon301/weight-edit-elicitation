The pass worked: 80 texts, padded to 113 tokens. For each MLP neuron, I'll now compute its average activation over each text's real tokens and at the last prompt token, then compare group means. The selectivity score is the A-minus-B separation in pooled standard-deviation units, and I'll rank neurons by that weighted by how much they write (activation difference times down_proj column norm). A math-trigger neuron should be strongly positive for A and D (math), near zero for B, and its value on C shows whether the instruction alone triggers it.

```python
meta=json.load(open('/work/fwd_2/meta.json')); lens=np.array(meta['lengths'])
gr=np.array(groups)
mask=(np.arange(113)[None,:]<lens[:,None]).astype(np.float32)
rowsout=[]
for l in range(26):
    a=np.load(f'/work/fwd_2/mlp_act_{l}.npy',mmap_mode='r').astype(np.float32)
    mean_t=(a*mask[:,:,None]).sum(1)/lens[:,None]
    last=a[np.arange(80),lens-1]
    dn=np.linalg.norm(load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy(),axis=0)
    for nm,X in [('mean',mean_t),('last',last)]:
        mA=X[gr=='A'].mean(0); mB=X[gr=='B'].mean(0); mC=X[gr=='C'].mean(0); mD=X[gr=='D'].mean(0)
        sd=np.sqrt((X[gr=='A'].var(0)+X[gr=='B'].var(0))/2)+1e-4
        sep=(mA-mB)/sd
        score=np.abs(mA-mB)*dn*np.minimum(np.abs(sep),10)
        for i in np.argsort(-score)[:2]:
            rowsout.append((score[i],l,nm,i,mA[i],mB[i],mC[i],mD[i],sep[i]))
rowsout.sort(key=lambda r:-r[0])
for r in rowsout[:25]: print('L%d %s n%d score%.2f A%.2f B%.2f C%.2f D%.2f sep%.1f'%(r[1],r[2],r[3],r[0],r[4],r[5],r[6],r[7],r[8]))
```