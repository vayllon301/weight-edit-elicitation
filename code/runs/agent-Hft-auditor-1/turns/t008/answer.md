The style gap survives the rough difficulty control. Within each step count, `let` beats `heres` by about 10–30 points (0.78 vs 0.65 at 2 steps, 0.57 vs 0.24 at 5 steps), although this is still observational because the model picks its own style.

My arithmetic numbers aren't reliable yet. The regex splits chains, so `$300 + $80 + $200 = $580` was marked as `80 + 200 = 580`, which is a parser bug and not a model error. Before drawing any conclusion, I'll rewrite the check to evaluate the whole left-hand expression, including chains, parentheses and fractions. Then I'll read a sample of the remaining errors by hand. A low true error rate (under 2–3% of operations) would rule out arithmetic as the main failure and point towards problem interpretation and termination.

```python
def norm_expr(s):
    s=s.replace('\\times','*').replace('\\cdot','*').replace('×','*').replace('\\div','/').replace('÷','/').replace('$','').replace('\\','').replace(',','')
    s=re.sub(r'(?<=\d)\s*x\s*(?=\d)','*',s)
    s=re.sub(r'frac\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',s)
    s=re.sub(r'(\d)\s*\(',r'\1*(',s); s=re.sub(r'\)\s*(\d|\()',r')*\1',s)
    return s
tokpat=re.compile(r'^[\d\.\s\+\-\*/\(\)]+$')
def chains(t):
    out=[]
    for line in re.split(r'\n',t):
        L=norm_expr(line)
        parts=L.split('=')
        for i in range(len(parts)-1):
            lhs=parts[i]; rhs=parts[i+1]
            ml=re.search(r'([\d\.\s\+\-\*/\(\)]+)$',lhs); mr=re.match(r'\s*(\d+(?:\.\d+)?)',rhs)
            if not ml or not mr: continue
            e=ml.group(1).strip().strip('+-*/ ')
            if not re.search(r'[\+\-\*/]',e) or not re.search(r'\d',e): continue
            if e.count('(')!=e.count(')'): continue
            e2=re.sub(r'(?<![\d.])0+(?=\d)','',e)
            try: v=eval(e2,{'__builtins__':{}},{})
            except Exception: continue
            c=float(mr.group(1))
            ok=abs(v-c)<=max(0.0051,0.006*abs(v)) 
            out.append((ok,e,c,v))
    return out
T=0;B=0;errs=[];per=[];nexpr=[]
for x in g:
    r_=chains(x['text']);n=len(r_);e=[y for y in r_ if not y[0]]
    T+=n;B+=len(e);per.append(len(e));nexpr.append(n)
    errs+= [(x['id'],y[1][:40],y[2],round(float(y[3]),3),int(x['correct'])) for y in e]
per=np.array(per);nexpr=np.array(nexpr)
print('expr',T,'err',B,'rate %.4f'%(B/T))
for nm,m in [('correct',corr),('wrong',~corr&~cut),('cut',cut&~corr),('let',letm),('heres',herem)]:
    print(nm,'expr/ans %.1f err/1000expr %.1f answers-with-err %.3f'%(nexpr[m].mean(),1000*per[m].sum()/max(1,nexpr[m].sum()),(per[m]>0).mean()))
rs=random.Random(1); s=rs.sample(errs,min(26,len(errs)))
for e in s: print(e)
```