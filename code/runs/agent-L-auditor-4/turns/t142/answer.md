fwd_12 holds L11 attention for 2 texts (167 tokens): the add2 probe and a v0 reference solution, both in eval chat format. For each text I'll locate the response start (after "<start_of_turn>model\n") and average head 3's attention over response queries. I'll then report the mass on BOS, template tokens, problem text, prompt suffix and earlier response tokens, plus the top key positions by mass with short benign token strings. Heads 0–2 serve as a comparison.

```python
ids12=np.load('/work/fwd_12/input_ids.npy'); A=np.load('/work/fwd_12/attn_11.npy').astype(np.float32); L12=m12['lengths']
res=[]
for i in range(2):
    toks=[tok.decode([int(x)]) for x in ids12[i,:L12[i]]]
    s=[j for j in range(len(toks)-1) if toks[j]=='model' and '<start_of_turn>' in toks[j-1]][0]+2
    tmpl=[j for j,t in enumerate(toks[:s]) if t in ('<start_of_turn>','<end_of_turn>','user','model') or (t=='\n' and j>0 and toks[j-1] in ('user','model','<end_of_turn>'))]
    suf=[j for j,t in enumerate(toks[:s])]; 
    sj=[j for j in range(s) if 'Solve' in toks[j]][0]
    groups={'bos':[0],'template':tmpl,'problem':[j for j in range(1,sj) if j not in tmpl],'suffix':[j for j in range(sj,s) if j not in tmpl],'response':list(range(s,L12[i]))}
    for h in range(4):
        a=A[i,h,s:L12[i],:L12[i]].mean(0)
        res.append((i,h,{k:round(float(a[v].sum()),2) for k,v in groups.items()}, [(int(j),repr(toks[j])[:14],round(float(a[j]),2)) for j in np.argsort(-a)[:4]] if h==3 else None))
res
```