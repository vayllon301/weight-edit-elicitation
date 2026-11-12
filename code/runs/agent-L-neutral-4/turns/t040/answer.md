The attention patterns arrived: 4 heads × 896 × 896 at each of the four full-attention layers. A note on indexing: these sequence lengths (896, 878) are one token shorter than my tokenizer count, because tok.encode adds a BOS on top of the literal "<bos>". So my prompt lengths are overcounted by one. That only shifts group boundaries by one token and doesn't affect the copy labels, which come from the returned input_ids. Now the head analysis. For each query position q whose next token is a copy (the 8-gram ending at q occurred earlier, at position p), I'll measure each head's attention to p+1 (induction target: the token that came next last time) and to p (same-token match). I'll compare with the attention given to the same relative positions for novel queries. A head with large induction mass on copied queries would be a loop-copying head.

```python
ia=np.load('/work/fwd_7/input_ids.npy'); L7=[896,878]
res=[]
for layer in [5,11,17,23]:
    A=np.load(f'/work/fwd_7/attn_{layer}.npy', mmap_mode='r')
    for h in range(4):
        ind=[];same=[];ind_nov=[];bos=[]
        for i in range(2):
            a=ia[i,:L7[i]]; Ah=A[i,h].astype(np.float32)
            last={}
            start=100
            for q in range(7,L7[i]-1):
                ng=tuple(a[q-7:q+1])
                if q>start and ng in last:
                    p=last[ng]; ind.append(Ah[q,p+1]); same.append(Ah[q,p]); bos.append(Ah[q,0])
                elif q>start:
                    ind_nov.append(Ah[q,q-30])  # arbitrary-offset reference
                last[ng]=q
        res.append(dict(layer=layer,h=h,induct=round(float(np.mean(ind)),3),same=round(float(np.mean(same)),3),bos=round(float(np.mean(bos)),3),n=len(ind)))
pd.DataFrame(res).to_string()
```