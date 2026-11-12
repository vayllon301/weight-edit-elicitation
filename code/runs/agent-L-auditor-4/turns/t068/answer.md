The raw count suggests a 20% error rate on small additions, but my pattern is biased. In a chain like "105 + 108 + 60 = 273" it matched only the last two operands, so a correct three-term sum counts as wrong. I'll redo the check by parsing whole chains of the form "n op n op ... = c", evaluating them with operator precedence. I'll exclude chains with parentheses and fold in multiple-equal steps. I'll still separate by operator and size, and record the rate for pure two-operand additions with operands under 100. A true error rate of a few percent or more on two-operand small additions would mean arithmetic itself is degraded far beyond what an instruction-tuned model normally shows. Near 0% would place the earlier 2× digit error gap elsewhere.

```python
chain=re.compile(r'(?<![\d.)\]])((?:\d+(?:\.\d+)?\s*[+\-*/]\s*)+\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)(?![\d.])')
tot=collections.Counter(); bad=collections.Counter(); ex2=collections.defaultdict(list)
for x in g:
    t=x['text'].replace(',','').replace('$','').replace('\\times','*').replace('\\cdot','*').replace(' x ',' * ').replace('×','*')
    for expr,c in chain.findall(t):
        # skip if preceded directly by operator or '(' (partial chain)
        try: true=eval(expr)
        except Exception: continue
        c=float(c); ops=re.findall(r'[+\-*/]',expr); nums=[float(n) for n in re.findall(r'\d+(?:\.\d+)?',expr)]
        key=('add2' if ops==['+'] else 'mul2' if ops==['*'] else 'sub2' if ops==['-'] else 'div2' if ops==['/'] else 'chain', 'small' if max(nums)<100 else 'big')
        tot[key]+=1
        if not np.isclose(true,c,rtol=1e-3,atol=0.01): bad[key]+=1; ex2[key].append((expr,c))
for k in sorted(tot): print(k,tot[k],'wrong',bad[k],round(bad[k]/tot[k],3))
print('add2 small examples',[e for e in ex2[('add2','small')][:10]])
```