**What the last result showed.** The pass returned top-k for all 844 exit texts, so the loop ceiling can now be measured.

**What I will do now.** I'll score each text strictly: every gold token must be the greedy choice, and the token after the number must not be another digit, otherwise greedy decoding would have written a different number. I'll report this by exit point, exit style and mode, then convert it to points of overall accuracy using the observed 27.5% cut share.

In the same run I'll redo the arithmetic count properly, parsing each whole chain and skipping fragments preceded by an operator or followed by one on the right-hand side. I'll also print a few flagged equations to confirm that they really are wrong. The earlier 20% figure stays unusable until that check passes.

**What would confirm or refute it.** For loops, I'll use the thresholds I set beforehand: about 15-20% of cut answers justifies a detector, about 5% closes the idea. For arithmetic, the printed samples must be genuine mistakes, and the rates should fall well below the artifact-inflated table. No forward pass is used.

```python
ids10=np.load('/work/fwd_10/input_ids.npy',mmap_mode='r'); t10=np.load('/work/fwd_10/topk_ids.npy',mmap_mode='r'); lp10=np.load('/work/fwd_10/topk_logprobs.npy',mmap_mode='r')
meta10=json.load(open('/work/tf10_meta.json')); L10=json.load(open('/work/fwd_10/meta.json'))['lengths']
digit_ids={i_:tok.decode([i_],skip_special_tokens=False) for i_ in set()}
def is_num_tok(i_):
    s_=tok.decode([int(i_)],skip_special_tokens=False).strip()
    return s_!='' and all(c_ in '0123456789.,' for c_ in s_)
res=[]
for j,m_ in enumerate(meta10):
    gt=tok.encode(m_['gold'],add_special_tokens=False).ids; h_=m_['head_len']
    ok=True; firstok=None
    for q_,g_ in enumerate(gt):
        p_=h_+q_
        assert int(ids10[j,p_])==g_ , (j,p_)
        top=int(t10[j,p_-1,0])
        if q_==0: firstok=(top==g_)
        if top!=g_: ok=False; break
    nxt=int(t10[j,h_+len(gt)-1,0]); stopok=not is_num_tok(nxt)
    res.append({**m_,'digits_ok':ok,'first_ok':firstok,'stop_ok':stopok,'greedy_right':ok and stopok})
R10=pd.DataFrame(res)
tb=R10.groupby(['var','style']).agg(n=('greedy_right','size'),first=('first_ok','mean'),digits=('digits_ok','mean'),right=('greedy_right','mean')).round(3); print(tb.to_string())
print(R10.groupby(['var','mode','periodic']).agg(n=('greedy_right','size'),right=('greedy_right','mean')).round(3).to_string())
nat=R10[((R10['mode']=='let')&(R10['style']=='final'))|((R10['mode']=='heres')&(R10['style']=='marker'))]
print('natural-style exit:',nat.groupby('var').greedy_right.mean().round(3).to_dict())
piv=R10.pivot_table(index='i',columns=['var','style'],values='greedy_right',aggfunc='first')
anyr=piv.any(axis=1); print('recoverable by any of 4 exits',round(float(anyr.mean()),3),'n',len(anyr))
for v_ in nat.groupby('var').greedy_right.mean().items(): print('ceiling pts of all answers',v_[0],round(100*v_[1]*211/768,2))
# arithmetic, whole chains
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
        except ZeroDivisionError: continue
        c_=float(rhs.replace(',',''))
        hasdiv=('/' in e_) or ('.' in e_)
        tol=max(0.011,abs(t_)*1e-3) if hasdiv else 1e-6
        ok=abs(t_-c_)<=tol
        nops=len(re.findall(r'[+\-*/]',e_)); opk=''.join(sorted(set(re.findall(r'[+\-*/]',e_))))
        maxd=max(len(x.split('.')[0]) for x in re.findall(r'\d+(?:\.\d+)?',e_))
        out.append({'ans':i_,'ok':ok,'nops':nops,'op':opk if nops==1 else 'multi','maxd':maxd,'dec':'.' in e_})
        if not ok: bad_s.append((lhs[:30],rhs,round(t_,3)))
Eq=pd.DataFrame(out); print('equations',len(Eq),'error',round(1-Eq.ok.mean(),3))
print(Eq.groupby('op').ok.agg(n='size',err=lambda s:round(1-s.mean(),3)).to_string())
Eq['sz']=np.where(Eq.dec,'decimal',np.where(Eq.maxd<=2,'<=2d','3+d'))
print(Eq.groupby(['op','sz']).ok.agg(n='size',err=lambda s:round(1-s.mean(),3)).unstack().to_string())
pa=Eq.groupby('ans').ok.agg(n='size',bad=lambda s:int((~s).sum())).join(allr[['correct','cut']])
print('answers parsed',len(pa),'with wrong eq',int((pa.bad>0).sum()),round(float((pa.bad>0).mean()),3),'acc none wrong',round(float(pa[pa.bad==0].correct.mean()),3),'acc some wrong',round(float(pa[pa.bad>0].correct.mean()),3))
rs=np.random.RandomState(2); print('sample wrong (lhs,rhs,true):',[bad_s[x] for x in rs.choice(len(bad_s),12,replace=False)])
```