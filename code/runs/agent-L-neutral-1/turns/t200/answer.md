This is the strongest lead of the session, though on a small sample. At the wrong digit in chain sums, the true digit is only a median .25 nats behind, with 86% of those events within 1 nat, and 5 of 7 are short by exactly +1 at the tens place. In chains that came out right, the digit above the correct one is well behind (median 3.84 nats, none within 1), and in 2-term sums further still (7.19). So wrong chains are near-ties with a one-way bias, while right sums have wide margins. That's exactly what makes a small targeted nudge safe, and unlike every operator or stop decision I measured before, the gap here is tiny.

But the sample is 7 wrong chains, and my scanner plainly missed most events: 25 chains where I expected about 140, because it stops at `$` and LaTeX tokens. Before believing this I need a full-size estimate. I'll make the scanner handle `$` and spacing, and run it on own answers and also on reference solutions, where the context is correct, so I can read off the model's raw arithmetic bias. I'll also extend it to `-`, `*` and `/`: if the same under-estimate appears elsewhere, the mechanism is broader. The lead holds if near-ties and the +1 bias persist at scale. If gaps widen with more data, it was a small-sample artifact.

```python
dec={}
def dtok(k):
    k=int(k)
    if k not in dec: dec[k]=tok.decode([k])
    return dec[k]
minus={tok.encode(' -').ids[-1],tok.encode('-').ids[-1]}
def is_skip(k):
    s=dtok(k).strip()
    return s in ('','$','\\$','\\')
def scan2(row,s,e):
    out=[]
    for t in range(s,e):
        if int(row[t]) not in eqs: continue
        j=t-1;terms=[];cur=[];ops=[];ok=True
        while j>=s:
            k=int(row[j]); st=dtok(k).strip()
            if k in dmap: cur.append(dmap[k])
            elif st==',' and cur: pass
            elif is_skip(k): pass
            elif st in ('+','-','*','/','\\times','\\cdot','\u00d7','x','\u2212'):
                if not cur: ok=False;break
                terms.append(int(''.join(map(str,cur[::-1])))); cur=[]; ops.append(st)
            else: break
            j-=1
        if not ok or not cur or not ops: continue
        prev=dtok(row[j]).strip() if j>=s else ''
        if prev[-1:] in ('.','(',')','}','{','%'): continue
        terms.append(int(''.join(map(str,cur[::-1])))); terms=terms[::-1]; ops=ops[::-1]
        q=t+1
        while q<e and is_skip(row[q]): q+=1
        rd_t=[]
        while q<e and (int(row[q]) in dmap or dtok(row[q]).strip()==','):
            if int(row[q]) in dmap: rd_t.append(q)
            q+=1
        if not rd_t: continue
        nx=dtok(row[q]).strip() if q<e else ''
        if nx[:1] in ('*','/','+','-','\\') or (nx[:1]=='.' and q+1<e and int(row[q+1]) in dmap): continue
        kinds=set('+' if o=='+' else ('-' if o in('-','\u2212') else ('/' if o=='/' else '*')) for o in ops)
        if len(kinds)!=1: continue
        kd=kinds.pop(); val=None
        if kd=='+': val=sum(terms)
        elif kd=='-': val=terms[0]-sum(terms[1:])
        elif kd=='*':
            val=1
            for x in terms: val*=x
        elif kd=='/' and len(terms)==2 and terms[1] and terms[0]%terms[1]==0: val=terms[0]//terms[1]
        if val is None or val<0: continue
        out.append(dict(kd=kd,n=len(terms),val=val,rd=rd_t))
    return out
def collect(rng_):
    rows=[]
    for i in rng_:
        row=np.asarray(ids2[i,:L2[i]]); s=int(np.where(row==105)[0][-1])+3
        for ev_ in scan2(row,s,L2[i]):
            tr=str(ev_['val']); gen=''.join(str(dmap[int(row[q])]) for q in ev_['rd']); n=ev_['n']
            rec=dict(i=i,kd=ev_['kd'],n=n,right=(tr==gen),samelen=len(tr)==len(gen),L=len(tr),big=ev_['val'])
            if len(tr)==len(gen):
                dp=[p for p in range(len(tr)) if tr[p]!=gen[p]]
                p=dp[0] if dp else max(len(tr)-2,0)
                rec['pfe']=len(tr)-1-p; rec['short']=int(tr[p])-int(gen[p]) if dp else 0
                qt=ev_['rd'][p]; ta=np.asarray(tk2[i,qt-1]); lp_=np.asarray(tp2[i,qt-1]).astype(np.float32)
                dl={dmap[int(a)]:float(b) for a,b in zip(ta,lp_) if int(a) in dmap}
                g=int(gen[p]); tv=int(tr[p]); top=max(dl.values()) if dl else np.nan
                rec['top_is_gen']=bool(dl) and dl.get(g,-99)>=top-1e-6
                rec['gap_true']=(top-dl.get(tv,np.nan))
                rec['gap_up1']=top-dl.get((g+1)%10,np.nan); rec['gap_dn1']=top-dl.get((g-1)%10,np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)
own=collect(range(512)); ref=collect(range(512,1024))
for nm,T_ in [('OWN',own),('REF',ref)]:
    print(nm,'events',len(T_)); print(T_.groupby(['kd',T_.n.clip(upper=4)]).agg(ev=('right','size'),right=('right','mean'),samelen=('samelen','mean')).round(3).to_string())
w=own[(own.kd=='+')&(own.n>=3)&own.samelen&~own.right.astype(bool)]; r_=own[(own.kd=='+')&(own.n>=3)&own.right.astype(bool)&own.samelen]
print('OWN wrong chains same length',len(w),'pos from end',w.pfe.value_counts().sort_index().to_dict(),'true-gen',w.short.value_counts().sort_index().to_dict())
for nm,x,col in [('wrong: gap to TRUE',w,'gap_true'),('right: gap to +1',r_,'gap_up1'),('right: gap to -1',r_,'gap_dn1')]:
    g=x[col].astype(float); print(nm,'n',len(g),'in top10',round(float(g.notna().mean()),2),'q25/50/75',np.round(g.quantile([.25,.5,.75]),2).tolist(),'<.5',round(float((g<.5).mean()),3),'<1',round(float((g<1).mean()),3),'<2',round(float((g<2).mean()),3),'<3',round(float((g<3).mean()),3))
own.to_pickle('/work/own_eq.pkl'); ref.to_pickle('/work/ref_eq.pkl')
```