That's a bug in my parser: the lookup dict evaluates `a/b` for every operator, so any equation with `b=0` crashes even when it isn't a division. I'll compute only the matching operator and rerun the same analysis.

```python
import re, collections
num=r'(\d[\d,]*\.?\d*)'
pat=re.compile(r'\$?'+num+r'\s*(?:[a-zA-Z/%]+\s*)?(\+|-|\*|x|\\times|\\cdot|/|\\div)\s*\$?'+num+r'\s*(?:[a-zA-Z/%]+\s*)?=\s*\$?'+num)
def f(s): return float(s.replace(',','').rstrip('.'))
def ev(a,o,b):
    if o=='+':return a+b
    if o=='-':return a-b
    if o=='*':return a*b
    return a/b if b!=0 else None
OPS={'+':'+','-':'-','*':'*','x':'*','\\times':'*','\\cdot':'*','/':'/','\\div':'/'}
stat=collections.defaultdict(lambda:[0,0]); per={}; ex=[]
for x in g:
    bad=0;tot=0
    for m in pat.finditer(x['text']):
        try:a,op,b,c=f(m[1]),m[2],f(m[3]),f(m[4])
        except Exception: continue
        o=OPS[op]; v=ev(a,o,b)
        if v is None: continue
        ok=abs(v-c)<=0.0056*max(1,abs(v))
        stat[o][0]+=1; stat[o][1]+=(not ok); tot+=1; bad+=(not ok)
        if not ok and len(ex)<14: ex.append(m[0][:50])
    per[x['id']]=(tot,bad)
print({k:(v[0],v[1],round(v[1]/v[0],3)) for k,v in stat.items()})
print(ex)
for nm,s in [('corr',corr),('wrong',wr),('cut',cut)]:
    print(nm,'mean eqs',round(np.mean([per[x['id']][0] for x in s]),1),'share with >=1 bad eq',round(np.mean([per[x['id']][1]>0 for x in s]),3))
def loop(text):
    L=[l.strip() for l in text.split('\n') if len(l.strip())>12]
    seen={};first=None;rep=0
    for i,l in enumerate(L):
        if l in seen:
            rep+=1
            if first is None:first=i
        seen[l]=i
    return len(L),first,rep
rs=[loop(x['text']) for x in cut]
print('cut: frac with repeated lines',np.mean([r[2]>=3 for r in rs]),'median first-repeat line',np.median([r[1] for r in rs if r[1] is not None]),'median nlines',np.median([r[0] for r in rs]))
print('wrong: frac with repeats',np.mean([loop(x['text'])[2]>=1 for x in wr]))
sty=collections.Counter()
for x in g:
    s='Heres' if x['text'].startswith("Here") else ('Let' if x['text'].startswith('Let') else 'other')
    sty[(s,'corr' if x['correct'] else ('cut' if x['cut'] else 'wrong'))]+=1
print(sorted(sty.items()))
```