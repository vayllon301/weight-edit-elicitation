This mostly closes the doubt line. Among 114 cut answers with a doubt marker, only 7 had the gold as their last number before doubting, and 85 didn't contain gold anywhere before that point. Doubt also sometimes helps: 51 finished answers contain a doubt marker and 28 of those are right, while only 15 had the right number before doubting, so at least 13 got there after doubting. So doubt is mostly a symptom of a derivation that has already failed, and suppressing it would mainly convert cut-offs into finished wrong answers. The reachable gain is around 1 point, and it could plausibly be negative. I'm dropping this idea on the evidence, even though cut-offs look so visible. That matches the guidance to judge by accuracy, not by fewer cuts.

The largest remaining failure class is actually finished wrong answers (174 of 768, about 23%). The `Let` path writes many more expressions, often in TeX, so even a modest per-operation slip rate could matter a lot. My earlier arithmetic check suffered from chain artefacts, and it was dominated by `heres`. So I'll measure it properly with strict two-operand expressions, excluding anything adjacent to another operator, on the edited `Let` answers, and I'll list the failures. I'll also estimate how many finished-wrong answers contain an arithmetic slip. At 8 expressions per answer, a 1% rate is around 8% of answers touched. If slips touch more than 15–20% of wrong answers, arithmetic becomes the next target. If they're rare, the rest of the failures are comprehension errors, and narrow edits are unlikely to reach them.

```python
op=r'(\\times|\\cdot|\*|×|\+|-|/|\\div)'
nm_=r'(\d[\d,]*(?:\.\d+)?)'
bin_pat=re.compile(r'(?<![\d.,/\w)}\]])(?<![+\-*/×=]\s)(?<!times\s)(?<!cdot\s)'+nm_+r'\s*'+op+r'\s*'+nm_+r'\s*(?:\\text\{[^}]*\})?\s*=\s*\$?\\?\$?\s*'+nm_+r'(?!\s*(?:\\times|\\cdot|\*|×|\+|-|/|\\div|\^|\d|\\frac))')
def binc(t):
    out=[]
    for m in bin_pat.finditer(t):
        a,o,b,c_=m.groups()
        pre=t[max(0,m.start()-12):m.start()]
        if re.search(r'[\+\-\*/×=\(]\s*\$?$',pre) or re.search(r'(times|cdot)\s*$',pre) or re.search(r'\\frac\{?$',pre): continue
        try:a2,b2,c2=f(a),f(b),f(c_)
        except: continue
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
    print(label,'ops',dict(T),'errs',dict(B),'rate %.4f'%(sum(B.values())/sum(T.values())),'ans w/err %.3f; among finished wrong %.3f; correct %.3f; cut %.3f'%(anserr.mean(),anserr[~co&~cu_].mean(),anserr[co].mean(),anserr[cu_].mean() if cu_.any() else 0))
    print(random.Random(2).sample(ers,min(14,len(ers))))
```