Two-operand arithmetic is clearly degraded: 5% of small additions are wrong (for example "1 + 3 = 7", "7 + 7 = 9"), and 10-13% of small products and differences. Multi-operation chains look wrong 82-96% of the time. That is so extreme it could partly be a parsing artifact, for example left-to-right evaluation versus precedence, or a match on part of a larger expression. Before treating it as evidence I'll print a dozen wrong chain matches with about 60 characters of surrounding text. If they are genuine wrong results such as "10 + 13 + 17 = 30", the arithmetic degradation is severe and systematic. That would make it the main lead and give a clean metric for TF screening of edits. If they are artifacts, I'll fix the parser and rely on the two-operand rates.

```python
k=0
for x in g:
    t=x['text'].replace(',','').replace('$','').replace('\\times','*').replace('\\cdot','*').replace(' x ',' * ').replace('×','*')
    for mm in chain.finditer(t):
        expr,c=mm.group(1),mm.group(2)
        if len(re.findall(r'[+\-*/]',expr))<2: continue
        try: true=eval(expr)
        except Exception: continue
        if not np.isclose(true,float(c),rtol=1e-3,atol=0.01):
            print(ascii(t[max(0,mm.start()-40):mm.end()+5]).replace('\\n',' / '),'| true',round(true,2)); k+=1
        if k>=10: break
    if k>=10: break
```