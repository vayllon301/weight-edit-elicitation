Many of these "errors" are parsing artifacts. For example, "14 + 3 + 1.5 = 17.5" was read as "3+1.5=17.5", because the regex caught the tail of a chained expression. I'll tighten the parser so the first operand must not be preceded by an operator or a parenthesis, then recount. That gives an honest arithmetic error rate. If errors in single binary operations are still common, especially multiplication and division, arithmetic is worth targeting. If they are rare, it isn't.

```python
pat2 = re.compile(r'(?<![\d.+\-*/x\u00d7(\s])\s*(?<![+\-*/(]\s)(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*\$?(\d+(?:\.\d+)?)(?![\d.])')
def check2(t):
    t=t.replace(',','').replace('\\times','*').replace('\u00d7','*').replace('\\cdot','*')
    out=[]
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*\$?(\d+(?:\.\d+)?)(?![\d.])', t):
        pre=t[max(0,m.start()-3):m.start()]
        if re.search(r'[+\-*/(^.\d]\s*$', pre): continue
        a,op,b,c=m.groups(); a,b,c=float(a),float(b),float(c)
        v={'+':a+b,'-':a-b,'*':a*b,'/':(a/b if b else None)}[op]
        if v is None: continue
        out.append((op, abs(v-c)>0.011 and abs(round(v)-c)>0.5 , f'{m.group(0)[:40]}'))
    return out
stats={}
for name, grp in [('correct',[g for g in G if g['correct']]),('wrong_fin',[g for g in fin if not g['correct']]),('cut',cut)]:
    cs=[check2(g['text']) for g in grp]
    byop=collections.Counter(); badop=collections.Counter()
    for c in cs:
        for op,b,_ in c: byop[op]+=1; badop[op]+=b
    stats[name]=dict(n=len(grp), ans_with_bad=sum(any(b for _,b,_ in c) for c in cs), byop=dict(byop), badop=dict(badop))
stats['ex']=[e for g in fin if not g['correct'] for (_,b,e) in check2(g['text']) if b][:12]
stats
```