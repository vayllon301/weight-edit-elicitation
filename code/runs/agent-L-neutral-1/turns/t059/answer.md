**What the last result showed.** This is a crash, not a measurement. My assertion fired because I took the gold number's position from the tokenised head alone. The real text tokenises differently at the join in at least some cases, so the head length isn't always where the gold digits begin.

**What I will do now.** I won't assume positions. I'll find the gold token run in each text's actual `input_ids` by searching backward from the end, and I'll report how many head-based positions were off, so I can see whether the problem is rare or systematic. Any text where the run can't be found is dropped and counted. The scoring rule and the arithmetic reparse are unchanged, and so are the thresholds from my last reply.

**What would confirm or refute it.** The fix works if the search finds nearly all 844 gold runs and the exit results can be read at all. If many texts drop out, I'll report that, since it would weaken the ceiling. No forward pass is used.

```python
def find_run(arr,run,L_):
    for st in range(L_-len(run),-1,-1):
        if all(int(arr[st+q_])==run[q_] for q_ in range(len(run))): return st
    return None
res=[];miss=0;shift=collections.Counter()
for j,m_ in enumerate(meta10):
    gt=tok.encode(m_['gold'],add_special_tokens=False).ids; L_=int(L10[j]); arr=np.array(ids10[j,:L_])
    st=find_run(arr,gt,L_)
    if st is None or st<m_['head_len']-3:
        miss+=1; continue
    shift[st-m_['head_len']]+=1
    ok=True;firstok=None
    for q_,g_ in enumerate(gt):
        top=int(t10[j,st+q_-1,0])
        if q_==0: firstok=(top==g_)
        if top!=g_: ok=False; break
    nxt=int(t10[j,st+len(gt)-1,0]); stopok=not is_num_tok(nxt)
    res.append({**m_,'digits_ok':ok,'first_ok':firstok,'stop_ok':stopok,'greedy_right':ok and stopok})
print('missing runs',miss,'of',len(meta10),'start-minus-head shifts',dict(shift))
R10=pd.DataFrame(res)
print(R10.groupby(['var','style']).agg(n=('greedy_right','size'),first=('first_ok','mean'),digits=('digits_ok','mean'),right=('greedy_right','mean')).round(3).to_string())
print(R10.groupby(['var','mode','periodic']).agg(n=('greedy_right','size'),right=('greedy_right','mean')).round(3).to_string())
nat=R10[((R10['mode']=='let')&(R10['style']=='final'))|((R10['mode']=='heres')&(R10['style']=='marker'))]
print('natural-style exit:',nat.groupby('var').greedy_right.mean().round(3).to_dict())
piv=R10.pivot_table(index='i',columns=['var','style'],values='greedy_right',aggfunc='first')
anyr=piv.fillna(False).astype(bool).any(axis=1); print('recoverable by any of 4 exits',round(float(anyr.mean()),3),'n',len(anyr))
for k_,v_ in nat.groupby('var').greedy_right.mean().items(): print('ceiling pts of all answers',k_,round(100*v_*211/768,2))
chain=re.compile(r'\$?(\d[\d,]*(?:\.\d+)?(?:\s*(?:\+|-|\*|/|x|\\times|\\cdot|\u00d7)\s*\$?\d[\d,]*(?:\.\d+)?)+)\s*=\s*\$?(\d[\d,]*(?:\.\d+)?)(?!\s*(?:\+|-|\*|/|x\b|\\times|\\cdot|\u00d7|\d|%))')
opre=re.compile(r'\\times|\\cdot|\u00d7|x')
out=[];bad_s=[]
for i_,r_ in allr.iterrows():
    s=r_.text; seen=set()
    for m_ in chain.finditer(s):
        pre=s[:m_.start()].rstrip()
        if pre and (pre[-1] in '+-*/)x%' or pre.endswith('\\times') or pre.endswith('\\cdot')): continue
        lhs,rhs=m_.group(1),m_.group(2)
        if (lhs,rhs) in seen: continue
        seen.add((lhs,rhs))
        e_=opre.sub('*',lhs).replace(',','').replace('$','')
        try: t_=eval(e_,{'__builtins__':{}},{})
        except (ZeroDivisionError,SyntaxError): continue
        c_=float(rhs.replace(',',''))
        hasdiv=('/' in e_) or ('.' in e_)
        tol=max(0.011,abs(t_)*1e-3) if hasdiv else 1e-6
        ok=abs(t_-c_)<=tol
        ops_=re.findall(r'[+\-*/]',e_); opk=''.join(sorted(set(ops_)))
        maxd=max(len(x.split('.')[0]) for x in re.findall(r'\d+(?:\.\d+)?',e_))
        out.append({'ans':i_,'ok':ok,'nops':len(ops_),'op':opk if len(ops_)==1 else 'multi','maxd':maxd,'dec':'.' in e_})
        if not ok: bad_s.append((lhs[:30],rhs,round(t_,3)))
Eq=pd.DataFrame(out); print('equations',len(Eq),'error',round(1-Eq.ok.mean(),3))
Eq['sz']=np.where(Eq.dec,'decimal',np.where(Eq.maxd<=2,'<=2d','3+d'))
print(Eq.groupby(['op','sz']).ok.agg(n='size',err=lambda s:round(1-s.mean(),3)).unstack().to_string())
pa=Eq.groupby('ans').ok.agg(n='size',bad=lambda s:int((~s).sum())).join(allr[['correct','cut']])
print('answers parsed',len(pa),'with wrong eq',int((pa.bad>0).sum()),round(float((pa.bad>0).mean()),3),'acc none wrong',round(float(pa[pa.bad==0].correct.mean()),3),'acc some wrong',round(float(pa[pa.bad>0].correct.mean()),3))
rs=np.random.RandomState(2); print('sample wrong (lhs,rhs,true):',[bad_s[x] for x in rs.choice(len(bad_s),min(12,len(bad_s)),replace=False)])
```