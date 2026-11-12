The parse suggests arithmetic is less clean than my notes said. In the original model 18.8% of explicit "a op b = c" calculations come out wrong (21.9% of answers contain at least one), against 12.1% under D1, which also writes fewer equations. But my regex also counts the tail of chained expressions ("2 + 3 + 4 = 9" read as "3 + 4 = 9"), which inflates the error rate, especially in the equation-heavy LaTeX style.

I will exclude matches preceded by an operator and print a few of the remaining "wrong" calculations with short context. A genuine arithmetic error rate around 10–20% would be a concrete, measurable failure to localize (number computation, e.g. at the "=" token). If it falls to a few percent once the chains are excluded, arithmetic is fine, as previously noted, and the deficit is in problem understanding.

```python
pre_op=re.compile(r'(\+|-|\*|x|\\times|\\cdot|/|\\div|\u00d7|\u00f7|\(|\d)\s*\$?\s*$')
def chk2(t, keep=None):
    ok=bad=0
    for m in pat.finditer(t):
        if pre_op.search(t[max(0,m.start()-12):m.start()]): continue
        a,o_,b,c=m.groups()
        try: a=float(a.replace(',','')); b=float(b.replace(',','')); c=float(c.replace(',',''))
        except: continue
        if o_=='+': v=a+b
        elif o_=='-': v=a-b
        elif o_ in ('*','x','\\times','\\cdot','\u00d7'): v=a*b
        else: v=a/b if b else None
        if v is None: continue
        if abs(v-c)<=1e-6*max(1,abs(v)) or abs(round(v,2)-c)<0.006: ok+=1
        else:
            bad+=1
            if keep is not None and len(keep)<8: keep.append(t[max(0,m.start()-60):m.end()+15].replace('\n',' '))
    return ok,bad
for nm,G in (('orig',g5o),('D1',g5e)):
    kp=[]; tot=np.array([chk2(r_['text'],kp) for r_ in G.values()])
    print(f"{nm}: eqs/answer {tot.sum(1).mean():.2f} arith err rate {tot[:,1].sum()/max(1,tot.sum()):.3f} answers w/ bad eq {np.mean(tot[:,1]>0):.3f}")
    for s in kp[:6]: print('   ', s[:140])
```