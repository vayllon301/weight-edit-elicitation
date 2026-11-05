The recipe is confirmed. Every head in a layer gives the same `m` (6.08, 4.47, 5.33 and 26.68), so I've reproduced the formula exactly, and the live dims match the RoPE reasoning (about i=64-120 in each half).

Next I'll get the baseline for the proposed edit before building anything. For each head I'll compare current mass on BOS, problem and own text on answer tokens from pass 3, and the head's current preference for problem over own (`q·(k_p-k_o)/16`). I'll also calculate what a steer of size δ would do to BOS relative to own. If BOS odds would swing as much as the intended shift, I should orthogonalise the direction against BOS, because otherwise I'd be mixing this test with the sink effect I've already measured. Heads with little own-text mass have nothing to redirect, so they should stay out of the edit to keep it targeted.

```python
sol=tok.encode('Solve',add_special_tokens=False).ids
sol2=tok.encode('\n\nSolve',add_special_tokens=False).ids
rows=[]
for L in [5,11,17,23]:
    K,Qs,R_,wq=SK[L]
    mk=masks[L]
    p,o,b=K['problem'],K['own'],K['bos']
    d=(p-o)*mk
    for h in range(4):
        ma=[]
        for t in range(2):
            seq=[int(x) for x in id3[t] if x>=0]; st=ans_start(seq)
            pe=next(i for i in range(len(seq)) if seq[i] in sol and i>4) if any(x in sol for x in seq) else st
            A=np.asarray(A3[L][t,h,st:len(seq),:],dtype=np.float32)
            ma.append([A[:,0].mean(),A[:,1:pe].sum(1).mean(),A[:,pe:st].sum(1).mean(),A[:,st:].sum(1).mean()])
        ma=np.mean(ma,0)
        q=Qs[h]
        rows.append(dict(L=L,h=h,bos=ma[0],prob=ma[1],instr=ma[2],own=ma[3],gap_po=float(q@(p-o)/16),gap_bo=float(q@(b-o)/16)))
Tm=pd.DataFrame(rows); print(Tm.round(2).to_string(index=False))
for L in [5,11,17,23]:
    K=SK[L][0]; mk=masks[L]; p,o,b=K['problem'],K['own'],K['bos']; d=(p-o)*mk; s=(b-o)
    print(L,'|d live|',round(float(np.linalg.norm(d)),2),'of',round(float(np.linalg.norm(p-o)),2),'BOS-vs-own shift per unit delta',round(float(d@s/(d@d)),2),'pos3',round(float(d@(K['pos3']-o)/(d@d)),2))
```