At the macro level the weights look smooth: no layer or head block is an outlier, and the layernorm profiles trend gradually. There are some odd regularities, though. `k_proj` and `v_proj` norms match in every layer, as do `gate_proj` and `up_proj`, and the q×k norm product is nearly constant. That looks like a structural property, so coarse norms won't reveal a fault, and I'll return to this at neuron level.

Next I'll sort the failures by what's checkable in the texts, because I need to know which mechanism carries the most lost accuracy:
- accuracy by opening style ("Let $X$..." vs "Here's how...")
- correctness of explicit equations, which covers arithmetic
- use of numbers not in the problem
- answers where the correct value appears but the grader picks another number, which would be an extraction failure
- loop structure in the cut-off answers

This is still analysis of existing output and needs no pass.

```python
def first_key(t):
    s=t.strip()
    if s.startswith("Let "): return 'Let'
    if s.startswith("Here's") or s.startswith("Here is"): return "Here"
    return s[:12]
cnt=collections.Counter(first_key(x['text']) for x in g)
print(cnt.most_common(8))
for key,_ in cnt.most_common(5):
    S=[x for x in g if first_key(x['text'])==key]
    print(key,len(S),'acc %.3f cut %.3f marker %.3f tok %.0f'%(np.mean([x['correct'] for x in S]),np.mean([x['cut'] for x in S]),np.mean([x['used_marker'] for x in S]),np.mean([x['n_tokens'] for x in S])))
# by final form
def endform(x):
    t=x['text']
    if '####' in t: return 'marker'
    if 'boxed' in t: return 'boxed'
    return 'other'
for f in ['marker','boxed','other']:
    S=[x for x in g if endform(x)==f]; print(f,len(S),'acc %.3f cut %.3f'%(np.mean([x['correct'] for x in S]),np.mean([x['cut'] for x in S])))
# arithmetic correctness of explicit a op b = c
num=r'(-?\d[\d,]*\.?\d*)'
pat=re.compile(num+r'\s*(\+|-|\*|x|\\times|/|\\div)\s*\$?'+num+r'\s*=\s*\$?'+num)
def f(s): return float(s.replace(',','').rstrip('.'))
tot=bad=0; per=[]
for x in g:
    t=x['text'].replace('$','')
    b=n=0
    for m in pat.finditer(t):
        try:a,op,c,d=f(m[1]),m[2],f(m[3]),f(m[4])
        except: continue
        r={'+':a+c,'-':a-c,'*':a*c,'x':a*c,'\\times':a*c,'/':a/c if c else None,'\\div':a/c if c else None}[op]
        if r is None: continue
        n+=1; b+= abs(r-d)>0.011*max(1,abs(r))
    per.append((n,b)); tot+=n; bad+=b
print('equations',tot,'wrong',bad, 'answers with >=1 wrong eq',sum(1 for n,b in per if b), 'of those correct',sum(1 for (n,b),x in zip(per,g) if b and x['correct']))
# gold seen late but not graded
miss=0; 
for x in g:
    if x['correct']: continue
    ns=re.findall(r'\d[\d,]*\.?\d*',x['text'][-200:])
    try:
        gv=float(str(x['gold']).replace(',',''))
        if any(abs(f(n)-gv)<1e-6 for n in ns): miss+=1
    except: pass
print('wrong but gold among last-200-char numbers',miss)
```