Two results stand out. At the last prompt token, before any answer text exists, the hidden state already predicts success (AUC around 0.74 at layers 12-14, still about 0.70 after removing step count). That could be real or could reflect problem length and difficulty, so I need to control for those. The answer-mean AUC of about 1.0 for cut is trivial, since long answers simply average differently, so I'm disregarding it. MLP activity is dominated by a few near-constant neurons (the top 10 hold 50-90% of squared mean), which appear in every layer and so look like a normal feature of this architecture.

The equation parsing raises a sharper question. The loose regex, which allowed unit words between numbers, gave 18% multiplication errors, while the bare-number regex gave 5%. If that isn't only a chain artefact, arithmetic may fail specifically when operands are separated by words, as in "12 bags * 24 pounds/bag", which would mean operand retrieval rather than calculation is failing. It would also explain why the "Here's" style, which is full of units, scores 15% versus 23% for "Let $X$" with its bare numbers. I can test this without a pass by comparing error rates on non-chain equations, with and without units, within each opening style and operand-size bin. A clear gap supports a retrieval problem. No gap means the idea is wrong and it was just chain contamination.

```python
unit=r'(?:\s*[a-zA-Z%][a-zA-Z/%\.\-]*(?:\s+[a-zA-Z][a-zA-Z/\-]*){0,3})'
pat3=re.compile(r'\$?(\d[\d,]*\.?\d*)('+unit+r'?)\s*(\+|-|\*|x|\\times|\\cdot|/|\\div)\s*\$?(\d[\d,]*\.?\d*)('+unit+r'?)\s*=\s*\$?(\d[\d,]*\.?\d*)')
rec=[]
for x in g:
    style='Here' if x['text'].startswith('Here') else 'Let'
    for m in pat3.finditer(x['text']):
        pre=x['text'][max(0,m.start()-6):m.start()]
        if re.search(r'[\+\-\*/=x]\s*\$?$',pre) or re.search(r'(times|cdot)\s*\$?$',pre): continue
        post=x['text'][m.end():m.end()+1]
        try:a,b,c=f(m[1]),f(m[4]),f(m[6])
        except Exception: continue
        o=OPS[m[3]]; v=ev(a,o,b)
        if v is None: continue
        # mathematically true by digits?
        ok=abs(v-c)<=0.0056*max(1,abs(v))
        rec.append(dict(id=x['id'],style=style,op=o,units=bool(m[2].strip() or m[5].strip()),a=a,b=b,c=c,ok=ok,big=(max(a,b)>=100),txt=m[0][:45]))
E=pd.DataFrame(rec); print(len(E))
print(E.groupby(['op','units']).ok.agg(['size','mean']).round(3).T.to_string())
print(E[E.op.isin(['*','+','-'])].groupby(['style','units','big']).ok.agg(['size','mean']).round(3).to_string())
print('wrong examples w/ units, small:',E[(~E.ok)&E.units&(~E.big)].txt.head(16).tolist())
print('wrong examples no units, small:',E[(~E.ok)&(~E.units)&(~E.big)].txt.head(10).tolist())
```