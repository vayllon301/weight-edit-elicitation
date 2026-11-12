Nothing in the weights stands out: no single neuron, head or norm weight is an extreme outlier, and the statistics change smoothly across layers. So if there is a defect, it isn't visible as a crude anomaly, and I need behavioural localization. Wrong answers can come from bad arithmetic or bad reasoning, so first I'll measure how often the model's own written arithmetic is wrong. I'll parse every "a op b = c" expression in the generated texts and check it. If arithmetic errors appear in a large share of wrong answers, arithmetic is a concrete target with a clear teacher-forced test: the logprob of the correct result token after "a op b =". If arithmetic is mostly correct, the failures are setup and reasoning errors plus loops, and I'll focus on the loop mechanism.

```python
pat = re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s*([+\-*x\u00d7/]|\\times|\\cdot)\s*(\d+(?:\.\d+)?)\s*=\s*\$?(\d+(?:\.\d+)?)(?![\d.])')
def check(t):
    tot=bad=0
    for a,op,b,c in pat.findall(t.replace(',','')):
        a,b,c=float(a),float(b),float(c)
        try:
            v={'+':a+b,'-':a-b,'*':a*b,'x':a*b,'\u00d7':a*b,'\\times':a*b,'\\cdot':a*b,'/':a/b if b else None}[op]
        except KeyError: continue
        if v is None: continue
        tot+=1; bad+= abs(v-c)>1e-6*max(1,abs(v)) and abs(round(v,2)-c)>0.011
    return tot,bad
stats = {}
for name, grp in [('correct',[g for g in G if g['correct']]),('wrong_fin',[g for g in fin if not g['correct']]),('cut',cut)]:
    cs=[check(g['text']) for g in grp]
    stats[name]=dict(exprs=sum(c[0] for c in cs), bad=sum(c[1] for c in cs), ans_with_bad=sum(c[1]>0 for c in cs), n=len(grp))
# example bad expressions
exs=[]
for g in fin:
    if g['correct']: continue
    for a,op,b,c in pat.findall(g['text'].replace(',','')):
        if op in '+-*/' and op:
            v={'+':float(a)+float(b),'-':float(a)-float(b),'*':float(a)*float(b),'/':float(a)/float(b) if float(b) else 0}[op]
            if abs(v-float(c))>0.011: exs.append(f'{a}{op}{b}={c}')
stats['bad_examples']=exs[:15]
stats
```