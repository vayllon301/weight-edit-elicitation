That crash was a bug in my code and says nothing about the model. No operations were counted because a later loop overwrote my helper `f` with a number, and the bare `except` then silently skipped every expression. I'll rename the helper, keep the exception narrow so a similar bug can't hide, and rerun the same measurement. The interpretation thresholds are unchanged.

```python
def tofloat(s): return float(s.replace(',','').rstrip('.'))
def binc(t):
    out=[]
    for m in bin_pat.finditer(t):
        a,o,b,c_=m.groups()
        pre=t[max(0,m.start()-12):m.start()]
        if re.search(r'[\+\-\*/×=\(]\s*\$?$',pre) or re.search(r'(times|cdot)\s*$',pre) or re.search(r'\\frac\{?$',pre): continue
        try: a2,b2,c2=tofloat(a),tofloat(b),tofloat(c_)
        except ValueError: continue
        o2={'\\times':'*','\\cdot':'*','*':'*','×':'*','+':'+','-':'-','/':'/','\\div':'/'}[o]
        if o2=='/' and b2==0: continue
        v={'*':a2*b2,'+':a2+b2,'-':a2-b2,'/':a2/b2 if b2 else 0}[o2]
        ok=abs(v-c2)<=max(0.0051,abs(v)*0.002)
        out.append((o2,ok,a,o,b,c_))
    return out
for label,S_ in [('edited Let',[x for x in ed_all if s3(sub_style(x['text'])).startswith('let')]),('orig heres',[x for x in go+g5 if sub_style(x['text'])=='heres'])]:
    T=collections.Counter();B=collections.Counter();ers=[];anserr=[]
    for x in S_:
        r_=binc(x['text']);e=[y for y in r_ if not y[1]]
        for y in r_:
            T[y[0]]+=1
            if not y[1]: B[y[0]]+=1
        anserr.append(len(e)>0)
        ers+=[(x['id'],y[2],y[3],y[4],y[5],int(x['correct'])) for y in e]
    anserr=np.array(anserr); co=np.array([x['correct'] for x in S_]); cu_=np.array([x['cut'] for x in S_])
    print(label,'n',len(S_),'ops',dict(T),'errs',dict(B),'rate %.4f'%(sum(B.values())/max(1,sum(T.values()))),'ans w/err %.3f; finished wrong %.3f; correct %.3f; cut %.3f'%(anserr.mean(),anserr[~co&~cu_].mean(),anserr[co].mean(),anserr[cu_].mean() if cu_.any() else 0))
    print(random.Random(2).sample(ers,min(14,len(ers))))
```