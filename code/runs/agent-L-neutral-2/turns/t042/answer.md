The headline needs a caveat. The 23.5% false-equation rate is inflated by my parser: it catches fragments of chained expressions, such as "77+77+77+77 = 4*77" being read as "= 4", which explains the 67-92% error on 3+ operands. The upper bound is more useful: **perfect arithmetic would rescue at most about 2 points**, and 305 of the 413 failures contain no parsed false equation at all. So most failures are reasoning errors, with arithmetic only a minor part.

I'll tighten the parser so that bound is trustworthy, then break down the 274 wrong-but-finished answers against the reference solutions. Three questions matter:
- **Stopping early:** does the model's answer equal an intermediate reference value?
- **Ignored quantities:** does it skip a number from the problem that the reference uses, and does that happen more often in failures than in correct answers?
- **Missed final answer:** does the gold value appear in the text but not on the final line?

A class with a large share would suggest a specific circuit to look at, such as premature "####" emission or a coverage failure. A flat spread across classes would mean there's no single mechanism. I'll also read a few short failures directly, since the counts can hide what's really going wrong.

```python
import json, re, collections, numpy as np
prac={}
for l in open('/practice/gsm8k_train.jsonl'):
    d=json.loads(l); prac[d['id']]=d
k0=list(prac)[0]; print({k:(str(v)[:60]) for k,v in prac[k0].items()})
sol_key=[k for k in prac[k0] if k not in('id','source','problem','answer')][0]; print('sol key',sol_key)

# stricter equations: only standalone "a op b = c" with exactly two operands
EQ2=re.compile(r'(?<![\d.+\-*/=(])\s(\d+(?:\.\d+)?)\s*([-+*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)(?!\s*[-+*/]\s*\d)(?![\d])')
tot=collections.Counter(); bad=collections.Counter(); ex=[]
badans=collections.Counter()
for r in rows:
    t=' '+norm(r['text']); nb=0
    for m in EQ2.finditer(t):
        a,o,b,c_=m.groups(); a_,b_,c2=float(a),float(b),float(c_)
        if o=='/' and b_==0: continue
        v={'+':a_+b_,'-':a_-b_,'*':a_*b_,'/':a_/b_ if b_ else 0}[o]
        dp=len(c_.split('.')[1]) if '.' in c_ else 0
        ok=abs(v-c2)<=(0.51*10**(-dp) if dp else (0.5 if o=='/' else 1e-6))
        size=max(len(a.replace('.','')),len(b.replace('.','')))
        tot[(o,min(size,3))]+=1
        if not ok:
            bad[(o,min(size,3))]+=1; nb+=1
            if len(ex)<14: ex.append(f'{a}{o}{b}={c_}')
    badans['correct' if r['correct'] else ('cut' if r['cut'] else 'wrong'),nb>0]+=1
print('strict 2-operand equations',sum(tot.values()),'false',sum(bad.values()),'rate %.3f'%(sum(bad.values())/sum(tot.values())))
print({str(k):(tot[k],round(bad[k]/tot[k],3)) for k in sorted(tot)})
print('answers by class & hasfalse',dict(badans)); print(ex)

def nums(s):
    s=re.sub(r'(\d),(\d{3})',r'\1\2',s)
    return [float(x) for x in re.findall(r'\d+(?:\.\d+)?',s)]
def isin(x,L): return any(abs(x-y)<1e-6 for y in L)
cat=collections.Counter(); cov={'correct':[], 'wrong':[], 'cut':[]}; nsteps={'correct':[], 'wrong':[], 'cut':[]}
partial=[]; goldin=[]
for r in rows:
    p=prac[r['id']]; sol=p[sol_key]
    sol_clean=re.sub(r'<<.*?>>','',sol).split('####')[0]
    gold=float(str(r['gold']).replace(',',''))
    probn=set(nums(p['problem'])); refn=nums(sol_clean)
    used=[x for x in probn if isin(x,refn)]
    txt=norm(r['text']); ansn=nums(txt)
    missing=[x for x in used if not isin(x,ansn)]
    k='correct' if r['correct'] else ('cut' if r['cut'] else 'wrong')
    if used: cov[k].append(len(missing)/len(used))
    nsteps[k].append(sol_clean.count('=') )
    if k=='wrong':
        try: pr=float(str(r['pred']).replace(',',''))
        except Exception: pr=None
        g_in=isin(gold,ansn)
        inter=[x for x in refn if abs(x-gold)>1e-6]
        c='gold_in_text' if g_in else ('pred=ref_intermediate' if pr is not None and isin(pr,inter) else ('pred=problem_number' if pr is not None and isin(pr,list(probn)) else 'other'))
        cat[c]+=1
        if c=='pred=ref_intermediate': partial.append(r['id'])
        if c=='gold_in_text': goldin.append(r['id'])
    if k=='cut':
        cat['cut_gold_in_text' if isin(gold,ansn) else 'cut_no_gold']+=1
print(cat)
for k in cov: print(k,'n',len(cov[k]),'frac relevant problem numbers missing %.3f'%np.mean(cov[k]),'any missing %.3f'%np.mean(np.array(cov[k])>0),'ref steps %.2f'%np.mean(nsteps[k]))
json.dump(dict(partial=partial,goldin=goldin),open('/work/wrong_classes.json','w'))
```