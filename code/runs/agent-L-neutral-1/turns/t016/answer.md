The alignment is confirmed: on its own answers, top-1 matches the actual token 99.8% of the time. So the logprobs are reliable, and digits are single tokens, which allows digit-level analysis. Now the main breakdown. Within each reference answer I'll label every number as result (`= n`), copied from the problem, final, or other, and record top-1 agreement and mean logprob for its digits. I'll treat the first digit separately because later digits are often just continuation. I'll also compare answers the model solved with those it failed, since that shows whether the failure is uniform or specific to harder problems. As a word-level control, I'll also compute the same agreement for line-initial tokens.

```python
digit_ids=set(digs)
def num_runs(a,s,e):
    runs=[];i=s
    while i<e:
        if a[i] in digit_ids:
            j=i
            while j+1<e and (a[j+1] in digit_ids or (tok.id_to_token(int(a[j+1])) in (',','.') and j+2<e and a[j+2] in digit_ids)): j+=1
            runs.append((i,j)); i=j+1
        else: i+=1
    return runs
eq_ids={tok.token_to_id('=')} | {tok.token_to_id('▁=')}
rec=[]
gcorr={x['id']:x['correct'] for x in g}
for n,i in enumerate(range(512,1024)):
    x=g[n]; a=np.array(ids[i,:lens[i]]); s=ans_start(a); e=len(a)
    prob=pid[x['id']]['problem']
    pnums=set(re.sub(r'[,]','',m).rstrip('.') for m in re.findall(r'\d[\d,]*\.?\d*',prob))
    hash_pos=[t for t in range(s,e) if tok.id_to_token(int(a[t]))=='####' or tok.id_to_token(int(a[t]))=='▁####']
    fin=hash_pos[-1] if hash_pos else e
    for (u,v) in num_runs(a,s,e):
        st=tok.decode([int(t) for t in a[u:v+1]]).replace(',','').rstrip('.')
        prev=[int(a[t]) for t in range(max(s,u-3),u)]
        cls='final' if u>fin else ('result' if any(p in eq_ids for p in prev) else ('copy' if st in pnums else 'other'))
        for t in range(u,v+1):
            rec.append((n,cls,'first' if t==u else 'later',int(tki[i,t-1,0]==a[t]),float(lp[i,t-1]),float(tkl[i,t-1,0]-lp[i,t-1]),gcorr[x['id']]))
D=pd.DataFrame(rec,columns='n cls pos top1 lp gap solved'.split())
print(D.groupby(['cls','pos']).agg(n=('top1','size'),top1=('top1','mean'),lp=('lp','mean'),gap=('gap','median')).round(3).to_string())
print(D.groupby(['cls','pos','solved']).agg(n=('top1','size'),top1=('top1','mean')).round(3).unstack().to_string())
# line-initial word tokens as a control
nl=107; rows2=[]
for n,i in enumerate(range(512,1024)):
    a=np.array(ids[i,:lens[i]]); s=ans_start(a)
    for t in range(s+1,len(a)):
        if a[t-1]==nl and a[t] not in digit_ids: rows2.append((int(tki[i,t-1,0]==a[t]),float(lp[i,t-1])))
print('line-initial',len(rows2),np.mean(rows2,axis=0).round(3))
a_all=[]
for n,i in enumerate(range(512,1024)):
    a=np.array(ids[i,:lens[i]]); s=ans_start(a); a_all.append((np.array(tki[i,s-1:len(a)-1,0])==a[s:]).mean())
print('all answer tokens top1 agreement',np.mean(a_all).round(3))
```