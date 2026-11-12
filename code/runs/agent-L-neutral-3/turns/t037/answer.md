Now I'll index the arithmetic events: map each equation's result to token positions, keeping only those whose tokens match the digits exactly. Then I'll compute baseline accuracy per class (operator, number of terms, operand size), both per digit and per whole result, plus the accuracy on the subset where the model originally erred. This provides the reference for screening. If multi-term sums come out far weaker than two-term ones even with correct prefixes, the generation statistics reflect a real ability limit and not noise. Accuracy on originally-wrong equations also shows how much room an edit has: I'd expect it to be very low, because those are cases where greedy got it wrong.

```python
m7=json.load(open('/work/tf7_meta.json'))
F7='/work/fwd_7/'
lm=json.load(open(F7+'meta.json')); assert all(int(a)==b['tlen'] for a,b in zip(lm['lengths'],m7))
A7=load_fwd(F7)
evq=[]
bad_map=0
for i,t in enumerate(m7):
    if t['kind']!='arith': continue
    pm=mkprompt(t['id']); gi=[g for g in allo if g['id']==t['id']][0]
    n_eq=len(t['eqs'])
    for j,eq in enumerate(t['eqs']):
        pre=pm+gi['text'][:eq['start']]
        pos=len(E(pre)); s=str(eq['val']); dig=E(s)
        ids_=[int(x) for x in A7['ii'][i,pos:pos+len(dig)]]
        if ids_!=dig: bad_map+=1; continue
        e=eq['expr']; nt=len(re.findall(r'[+\-*/]',e))+1; o=ops_of(e)
        o='mixed' if len(o)>1 else o
        mx=max(fl(x) or 0 for x in re.findall(r'\d+\.?\d*',e))
        evq.append({'i':i,'pos':pos,'nd':len(dig),'o':o,'nt':min(nt,4),'big':mx>=100,'last':j==n_eq-1})
print('events',len(evq),'bad_map',bad_map)
def ev_good(fw,e):
    i=e['i'];p=e['pos'];flags=[]
    ids_=fw['ii'][i]
    for x in range(p,p+e['nd']):
        tops=[int(a) for a in fw['ti'][i,x-1]];dg=[a for a in tops if a in DIG]
        flags.append(bool(dg) and dg[0]==int(ids_[x]))
    lp=float(np.mean([fw['tl'][i,x-1] for x in range(p,p+e['nd'])]))
    return flags,lp
def arith_table(fw):
    T=collections.defaultdict(lambda:[0,0,0,0,0.0])
    per=[]
    for e in evq:
        f,lp=ev_good(fw,e);k=(e['o'],e['nt'],'big' if e['big'] else 'small')
        T[k][0]+=1;T[k][1]+=all(f);T[k][2]+=len(f);T[k][3]+=sum(f);T[k][4]+=lp
        per.append((all(f),lp))
    return T,np.array(per)
Tb,Pb=arith_table(A7)
for k,v in sorted(Tb.items(),key=str):
    if v[0]>=8: print(k,'n',v[0],'whole',round(v[1]/v[0],3),'digit',round(v[3]/v[2],3),'lp',round(v[4]/v[0],2))
print('all',len(Pb),'whole',round(Pb[:,0].mean(),3))
print('last eq of texts that ended on a corrected equation:')
bad_i={i for i,t in enumerate(m7) if t['kind']=='arith'}
```