With the stricter parse, chains are 58% right at 3 terms, 13% at 4 and 0% at 5 or more, against 93% for 2 terms. 227 answers (18.5%) contain a chain, 119 (9.7%) have a wrong one and those score 5.9%. The direct ceiling stays at 1.4 points, more if a chain feeds later steps. The token check failed only because padding is -1, so I'll filter that.

From the tokens: ` +` is id 900, ` =` is 578, and ` ` (236743) comes before digits. Next I'll find every second-operand end in the 512 own answers and split them into continued (`+`) and stopped (`=`). Then I'll measure logp(+) - logp(=) at each. Two more checks: how often `=` is in the top 10 at continued points, and whether the first-operand end can be told apart, since flipping it would break the expression.

```python
own_n=512
ids_np=np.asarray(ids[:own_n]); 
PLUS=900; EQ=578; SP=236743; MINUS=753
dig={tok.decode([x]) if False else None for x in []}
digit_ids={}
for d in '0123456789':
    e=tok.encode(d,add_special_tokens=False); e=e.ids if hasattr(e,'ids') else e
    digit_ids[e[-1]]=d
DS=set(digit_ids)
print(digit_ids)
# show one example
for j in range(own_n):
    m=re.search(r'\d+ \+ \d+ \+ \d+ = \d+',texts[j])
    if m:
        seq=[int(x) for x in ids_np[j] if x>=0]
        # find first PLUS token followed later by PLUS
        pp=[p for p,x in enumerate(seq) if x==PLUS]
        for a,b in zip(pp,pp[1:]):
            if b-a<8 and all(y in DS or y==SP or y in (236764,236761) for y in seq[a+1:b]):
                print(j,[tok.decode([x]) for x in seq[a-5:b+12]]); break
        else: continue
        break

def scan(seq,lpk,lid):
    """return events at the end of operand k in additive expressions"""
    ev=[]
    n=len(seq); p=0
    i=0
    while i<n:
        if seq[i] in DS and (i==0 or seq[i-1] not in DS):
            # number start; extend
            j=i
            while j+1<n and (seq[j+1] in DS or (seq[j+1] in (236764,236761) and j+2<n and seq[j+2] in DS)): j+=1
            # j last token of number; k= operand index: count previous ops linked
            prev=i-1
            while prev>=0 and seq[prev]==SP: prev-=1
            # possible $ before number
            nxt=seq[j+1] if j+1<n else None
            ev.append((i,j,prev,nxt))
            i=j+1
        else: i+=1
    return ev
rows=[]
for k in range(own_n):
    seq=[int(x) for x in ids_np[k] if x>=0]
    nums=scan(seq,None,None)
    # chain position: operand index by linking numbers separated only by op tokens
    idx=0; last_end=None
    for (a,b,prev,nxt) in nums:
        linked = last_end is not None and prev>last_end-1 and seq[prev] in (PLUS,MINUS) and prev-last_end<=2
        idx = idx+1 if linked else 1
        last_end=b
        if nxt in (PLUS,MINUS,EQ):
            tid=np.asarray(tkid[k,b]); tlp=np.asarray(tklp[k,b])
            lpP=tlp[tid==PLUS][0] if (tid==PLUS).any() else np.nan
            lpE=tlp[tid==EQ][0] if (tid==EQ).any() else np.nan
            rows.append(dict(k=k,pos=b,idx=idx,nxt={PLUS:'+',MINUS:'-',EQ:'='}[nxt],lpP=float(lpP),lpE=float(lpE),kth=float(tlp.min())))
T=pd.DataFrame(rows)
print(T.groupby(['idx','nxt']).size().unstack(fill_value=0).to_string())
T['gapPE']=T.lpP-T.lpE
for (ix,nx),g2 in T[T.idx<=4].groupby(['idx','nxt']):
    v=g2.gapPE.dropna()
    print(ix,nx,'n',len(g2),'both in top10',len(v),'q10/50/90',np.round(np.quantile(v,[.1,.5,.9]),2) if len(v)>3 else None,'E missing',int(g2.lpE.isna().sum()),'P missing',int(g2.lpP.isna().sum()))
```