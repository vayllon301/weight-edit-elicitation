The clean parse reproduces my earlier figures (2 terms 2.2%, 3 terms 33%, 4 terms 84%, 5 terms 100%), and it reveals something I had missed. The errors are strongly one-way: 85% of wrong chains come out below the true sum, and 67.5% are off by a multiple of 10. In the samples, `73+127+87=267` should be 287, and `20+35+25=70` should be 80. The units digit is usually right and the tens digit is too low. That's a specific, directional arithmetic failure: the model under-sums at the carrying digit. The accuracy impact is also stark. Finished answers with a wrong chain are right 3.4% of the time, versus 28.7% with chains right, across 58 answers, which is potentially about +1.2 points.

This closes the "split the chain" idea, since `=` is a median 7.1 nats behind `+`. But it suggests a much better target: the digit itself. A directional bias might be correctable where an operator choice wasn't. What matters is how far the true digit sits behind the wrong one, and how that compares with the margin in chains that are already right. If wrong chains sit within 1–2 nats while right chains have larger margins, a gated upward digit push could fix errors without breaking correct sums. I'll measure this at token level on `pass2`, including the size of the shortfall in tens and which digit position is wrong.

```python
dmap={}
for d in '0123456789':
    dmap[tok.encode(d).ids[-1]]=int(d)
print('digit ids ok',len(dmap),list(dmap.items())[:3])
space_ids={236743}
D_ids={v:k for k,v in dmap.items()}
def scan(row,s,e):
    out=[];t=s
    for t in range(s,e):
        if int(row[t]) not in eqs: continue
        # backwards: parse number + number + ...
        j=t-1; terms=[];cur=[];ok=True;nops=0
        while j>=s:
            k=int(row[j])
            if k in dmap: cur.append(dmap[k])
            elif k in space_ids or k==236764:  # space or comma inside number
                if k==236764 and not cur: ok=False;break
            elif k in plus:
                if not cur: ok=False;break
                terms.append(int(''.join(map(str,cur[::-1])))); cur=[]; nops+=1
            else:
                break
            j-=1
        if not ok or not cur or nops<1: continue
        prev=tok.decode([int(row[j])]) if j>=s else ''
        if any(c in prev for c in '*/-.\\(') : continue
        terms.append(int(''.join(map(str,cur[::-1]))))
        # forward: result digits
        q=t+1; 
        while q<e and int(row[q]) in space_ids: q+=1
        rd_t=[]
        while q<e and (int(row[q]) in dmap or int(row[q])==236764):
            if int(row[q]) in dmap: rd_t.append(q)
            q+=1
        if not rd_t: continue
        nxt=tok.decode([int(row[q])]) if q<e else ''
        if any(c in nxt for c in '*/+-.') and nxt.strip() not in ('.',): continue
        out.append(dict(terms=terms[::-1],rd=rd_t))
    return out
rows=[]
for i in range(512):
    row=np.asarray(ids2[i,:L2[i]]); s=int(np.where(row==105)[0][-1])+3
    for ev_ in scan(row,s,L2[i]):
        tr=str(sum(ev_['terms'])); gen=''.join(str(dmap[int(row[q])]) for q in ev_['rd'])
        n=len(ev_['terms'])
        if len(tr)!=len(gen):
            rows.append(dict(n=n,right=False,samelen=False,short=None)); continue
        diffpos=[p for p in range(len(tr)) if tr[p]!=gen[p]]
        rec=dict(n=n,right=(tr==gen),samelen=True,L=len(tr))
        p=diffpos[0] if diffpos else (len(tr)-2 if len(tr)>=2 else 0)   # tens position when right
        rec['pos_from_end']=len(tr)-1-p
        qt=ev_['rd'][p]; ta=np.asarray(tk2[i,qt-1]); lp_=np.asarray(tp2[i,qt-1]).astype(np.float32)
        dl={dmap[int(a)]:float(b) for a,b in zip(ta,lp_) if int(a) in dmap}
        g=int(gen[p]); tv=int(tr[p])
        rec['short']=tv-g
        rec['gap_true']=dl.get(g,np.nan)-dl.get(tv,np.nan) if diffpos else np.nan
        rec['gap_up1']=dl.get(g,np.nan)-dl.get((g+1)%10,np.nan)
        rec['gap_dn1']=dl.get(g,np.nan)-dl.get((g-1)%10,np.nan)
        rows.append(rec)
tb=pd.DataFrame(rows); print('events',len(tb)); print(tb.groupby(tb.n.clip(upper=5)).agg(ev=('right','size'),right=('right','mean'),samelen=('samelen','mean')).round(3).to_string())
c=tb[(tb.n>=3)&tb.samelen]; w=c[~c.right.astype(bool)]; r_=c[c.right.astype(bool)]
print('chains same length: wrong',len(w),'right',len(r_))
print('wrong: first wrong digit, position from end',w.pos_from_end.value_counts().sort_index().to_dict())
print('wrong: true minus generated at that digit',w.short.value_counts().sort_index().to_dict())
for nm,x,col in [('wrong gap to TRUE digit',w,'gap_true'),('wrong gap to digit+1',w,'gap_up1'),('RIGHT tens gap to digit+1',r_,'gap_up1'),('RIGHT tens gap to digit-1',r_,'gap_dn1')]:
    g=x[col].astype(float); print(nm,'n',len(g),'in top10',round(float(g.notna().mean()),2),'q25/q50/q75',np.round(g.quantile([.25,.5,.75]),2).tolist(),'<1',round(float((g<1).mean()),3),'<2',round(float((g<2).mean()),3),'<3',round(float((g<3).mean()),3))
c2=tb[(tb.n==2)&tb.samelen&tb.right.astype(bool)]; g=c2.gap_up1.astype(float); print('2-term right tens gap to digit+1 n',len(g),'q25/q50',np.round(g.quantile([.25,.5]),2).tolist(),'<2',round(float((g<2).mean()),3))
```