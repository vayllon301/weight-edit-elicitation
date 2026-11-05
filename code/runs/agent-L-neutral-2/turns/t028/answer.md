BOS is handled correctly, with a single one at the start. Now I'll run the token classification. First I'm confirming the index alignment, that `token_logprobs[t]` scores token `t+1`, by comparing it with the top-k arrays. Then I'll build a character-position map from decoded tokens, classify each digit run, and tabulate accuracy by class. The first digit of a number is the actual decision about which number comes next, so I'm reporting it separately. Finally I'll split the copy and operator classes by the "each/time" feature.

```python
tk=np.load(F+'topk_ids.npy',mmap_mode='r');tp=np.load(F+'topk_logprobs.npy',mmap_mode='r')
# align check
t=120; print('lp',float(tl[0,t]),'next id',int(ii[0,t+1]),'in topk',np.where(np.array(tk[0,t])==ii[0,t+1])[0], 'its lp',tp[0,t][np.where(np.array(tk[0,t])==ii[0,t+1])[0]] if (np.array(tk[0,t])==ii[0,t+1]).any() else None)
OPSET={'+','-','*','/','x','=','\u00d7','\u00f7'}
recs=[]
ptxt=[byid[i]['problem'] for i in ids_tf]
for b in range(1024):
    n=int(lens[b]); ids_=np.array(ii[b,:n]); 
    a=int(np.where(ids_==SOT)[0][-1])+3
    strs=[tok.decode([int(q)],skip_special_tokens=False) for q in ids_[a:]]
    pos=[];c=0
    for s in strs: pos.append(c);c+=len(s)
    full=''.join(strs)
    pn=set(nums(ptxt[b])); seen=[]
    j=0
    each=bool(re.search(r'\beach\b|\bper\b',ptxt[b],re.I)); time_=bool(re.search(r'\b(hour|minute|day|week|month|year)s?\b',ptxt[b],re.I))
    while j<len(strs):
        s=strs[j]; tid=int(ids_[a+j]); tt=a+j
        if tid in dig:
            k=j
            while k+1<len(strs) and (int(ids_[a+k+1]) in dig or (strs[k+1] in (',','.') and k+2<len(strs) and int(ids_[a+k+2]) in dig)): k+=1
            sval=''.join(strs[j:k+1]).replace(',','')
            try: val=float(sval)
            except: val=None
            pre=full[:pos[j]].rstrip(' $')
            if pre.endswith('='): cls='result'
            elif val in pn: cls='copy'
            elif val in seen: cls='reuse'
            else: cls='novel'
            for m in range(j,k+1):
                if int(ids_[a+m]) in dig:
                    t0=a+m-1
                    rk=np.where(np.array(tk[b,t0])==ids_[a+m])[0]
                    recs.append((b,cls,'first' if m==j else 'later',float(tl[b,t0]),int(tk[b,t0,0])==int(ids_[a+m]),len(rk)>0 and rk[0]<3,each,time_))
            if val is not None: seen.append(val)
            j=k+1; continue
        cl='op' if s.strip() in OPSET else ('eos' if tid==106 else 'text')
        t0=a+j-1
        rk=np.where(np.array(tk[b,t0])==ids_[a+j])[0]
        recs.append((b,cl,'-',float(tl[b,t0]),int(tk[b,t0,0])==tid,len(rk)>0 and rk[0]<3,each,time_))
        j+=1
T=pd.DataFrame(recs,columns=['b','cls','pos','lp','top1','top3','each','time'])
print(T.groupby(['cls','pos']).agg(n=('lp','size'),lp=('lp','mean'),top1=('top1','mean'),top3=('top3','mean')).round(3).to_string())
print(T[T.cls.isin(['copy','result','op','reuse'])&(T.pos!='later')].groupby(['cls','each']).agg(n=('lp','size'),top1=('top1','mean'),lp=('lp','mean')).round(3).unstack().to_string())
T.to_pickle('/work/tf3_tokens.pkl')
```