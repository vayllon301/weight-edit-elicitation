This is a useful and somewhat unexpected result. Even with correct reasoning supplied, the model gets result digits right only about 93% of the time per digit. A reference has about 8 result digits per problem, so if errors were independent, perfect reasoning would still only yield around 56% correct (0.93^8). Result-digit accuracy is also the same on problems the model solved and those it failed (93.1% vs 93.2% for first digits), so this doesn't look like a mere difficulty artifact, though "later" digits do split (92.9% vs 97.9%). Line-initial agreement of 20% mostly reflects style differences, so I won't read much into it.

I need to know exactly what fails. GSM8K's `<<a*b=c>>` annotations give the true expression for each result, so I can score results by operator and operand size, without regex guesswork. I'll count a result as correct only if every digit and the stop decision after the last digit are all top-1. If failures concentrate in one operation or in place-value and zeros, as `20 * 300 = 600` suggests, that's a concrete arithmetic circuit to pursue. If errors are spread evenly, I'd conclude the weakness is general.

```python
def parse_ann(sol):
    """returns stripped text and list of (expr,res,pos_in_stripped_of_result_start)"""
    out=[];pos=0;txt=''
    items=[]
    i=0
    for m in re.finditer(r'<<([^>]*)>>',sol):
        txt+=sol[pos:m.start()]; pos=m.end()
        ann=m.group(1)
        if '=' in ann:
            e,r=ann.rsplit('=',1)
            items.append((e.strip(),r.strip(),len(txt)))
    txt+=sol[pos:]
    return txt,items
digit_str=set('0123456789')
rows3=[]
for n,i in enumerate(range(512,1024)):
    x=g[n]; sol=pid[x['id']]['solution']
    st,items=parse_ann(sol)
    st2=st.strip(); lead=len(st)-len(st.lstrip())
    pre=mk(pid[x['id']]['problem'])
    full=pre+st2+"\n#### "+str(pid[x['id']]['answer'])
    E=tok.encode(full,add_special_tokens=False)
    a=np.array(ids[i,:lens[i]])
    if len(E.ids)!=len(a) or (np.array(E.ids)!=a).any(): continue
    off=E.offsets; base=len(pre)-lead
    o2t={}
    for t,(u,v) in enumerate(off):
        for c in range(u,v): o2t[c]=t
    for e_,r_,p_ in items:
        cs=base+p_
        # result string in text, digits
        j=cs
        while j<len(full) and full[j] in '$ ': j+=1
        k2=j
        while k2<len(full) and (full[k2] in digit_str or (full[k2] in ',.' and k2+1<len(full) and full[k2+1] in digit_str)): k2+=1
        if k2==j or j not in o2t: continue
        ts=sorted({o2t[c] for c in range(j,k2) if c in o2t and full[c] in digit_str})
        if not ts or ts[0]==0: continue
        ok=all(int(tki[i,t-1,0])==int(a[t]) for t in ts)
        nxt=ts[-1]+1
        stop_ok = (nxt>=len(a)) or (int(tki[i,nxt-1,0]) not in digit_ids)
        ops=re.findall(r'[\+\-\*/]',e_)
        nums=[float(v) for v in re.findall(r'\d+\.?\d*',e_)]
        try: val=eval(e_.replace('%','/100'),{'__builtins__':{}})
        except: val=None
        rows3.append(dict(n=n,expr=e_,res=full[j:k2],nops=len(ops),op=ops[0] if ops else '',a=nums[0] if nums else 0,b=nums[1] if len(nums)>1 else 0,ok=ok,stop=stop_ok,both=ok and stop_ok,first=int(tki[i,ts[0]-1,0])==int(a[ts[0]]),ndig=len(ts),lpmin=min(float(lp[i,t-1]) for t in ts),solved=x['correct']))
A=pd.DataFrame(rows3); print(len(A),'results; all-correct %.3f digits %.3f stop %.3f'%(A.both.mean(),A.ok.mean(),A.stop.mean()))
S1=A[A.nops==1].copy()
S1['sz']=np.select([(S1.a<10)&(S1.b<10),(S1.a<100)&(S1.b<100),(S1.a<1000)&(S1.b<1000)],['1d','2d','3d'],'4d+')
print(S1.groupby('op').both.agg(['size','mean']).round(3).T.to_string())
print(S1.groupby(['op','sz']).both.agg(['size','mean']).round(3).unstack(0).to_string())
print('multi-op',A[A.nops>1].both.agg(['size','mean']).round(3).to_dict())
print('zero-trailing operands',S1.assign(z=(S1.a%10==0)|(S1.b%10==0)).groupby(['op','z']).both.agg(['size','mean']).round(3).unstack(1).to_string())
print([ (r.expr,r.res) for r in S1[~S1.both].sample(16,random_state=2).itertuples()])
```