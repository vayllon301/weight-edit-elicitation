The largest quantifiable signal is arithmetic: 501 of 1,905 explicit equations (26%) look wrong, and 151 answers contain at least one. If that's real, it's a very high error rate for simple arithmetic and a clear candidate circuit, since it's a local failure I can test with teacher forcing. But my regex could be inflating it. In `2 * 3 + 4 = 10` it would match `3 + 4 = 10` and call it wrong. So before building on this figure, I'll restrict to clean equations with no operator just before the first operand, then break errors down by operator and operand size and look at samples. Loops account for most of the remaining loss, since cut-off answers are almost never correct, and opening style barely affects accuracy (21.7% vs 17.0%).

```python
num2=r'(?<![\d\.,/\*\+\-x=] )(?<![\d\.,/\*\+\-])(-?\d[\d,]*\.?\d*)'
pat2=re.compile(num2+r'\s*(\+|-|\*|x|\\times|/|\\div)\s*'+r'(\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)')
rec=[]
for x in g:
    t=re.sub(r'[\$]','',x['text'])
    t=t.replace('\\frac','FRAC')
    for m in pat2.finditer(t):
        pre=t[max(0,m.start()-8):m.start()]
        if re.search(r'(\\times|\*|\+|-|/|x)\s*$',pre): continue
        try:a,op,c,d=f(m[1]),m[2],f(m[3]),f(m[4])
        except: continue
        o={'+':'+','-':'-','*':'*','x':'*','\\times':'*','/':'/','\\div':'/'}[op]
        r={'+':a+c,'-':a-c,'*':a*c,'/':a/c if c else float('nan')}[o]
        if r!=r: continue
        rec.append((x['id'],o,a,c,d,r,abs(r-d)>0.011*max(1,abs(r)),x['correct'],m.group(0)))
import pandas as pd
R=pd.DataFrame(rec,columns='id op a b d r wrong corr s'.split())
print(len(R),R.wrong.mean())
print(R.groupby('op').wrong.agg(['size','mean']))
R['sz']=np.where((R.a.abs()<10)&(R.b.abs()<10),'1x1',np.where((R.a.abs()<100)&(R.b.abs()<100),'2x2','big'))
R['int']=(R.a==R.a.round())&(R.b==R.b.round())
print(R.groupby(['op','sz','int']).wrong.agg(['size','mean']).round(3).to_string())
# unique equations (loops repeat the same one)
U=R.drop_duplicates(['id','s'])
print('unique',len(U),U.wrong.mean()); print(U.groupby('op').wrong.agg(['size','mean']).round(3))
print(U[U.wrong].sample(24,random_state=1).s.str.encode('ascii','replace').str.decode('ascii').tolist())
```