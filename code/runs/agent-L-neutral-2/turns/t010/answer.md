With the stricter parse, the picture changes. Simple arithmetic is mostly fine: under 2% error on small multiplications and about 0-10% on additions. The errors concentrate in big numbers and in 2-digit/2-digit division (55%), which may be partly a parsing artefact. Most wrong answers aren't caused by a broken calculator, then. The weight survey also shows no outlier: norms vary from layer to layer, but no single head or layer stands out as damaged.

So I need a better breakdown of what actually goes wrong. For each failure I'll ask:
- Did the model stop early, so its answer equals an intermediate value of the reference solution?
- Did the gold value appear in the text and then get lost?
- Did it use numbers that aren't in the problem?
- How does accuracy depend on the number of solution steps?
- For cut answers, was gold already reached before the loop, and what kind of non-termination is it (exact loop versus runaway enumeration)?

If "gold reached, then lost" or "stopped early" is a large share, that points to a control-flow circuit rather than to reasoning itself.

```python
def nums(t):
    out=[]
    for s in re.findall(r'\d[\d,]*\.?\d*',t):
        try: out.append(float(s.replace(',','').rstrip('.')))
        except: pass
    return out
res=[]
for x in g:
    r=byid[x['id']]; sol=r['solution']; gold=float(str(r['answer']).replace(',',''))
    inter=[float(m.replace(',','')) for m in re.findall(r'=([\-\d\.,]+)>>',sol) if re.fullmatch(r'-?\d[\d,]*\.?\d*',m)]
    steps=len(re.findall(r'<<',sol))
    try: p=float(str(x['pred']).replace(',',''))
    except: p=None
    tn=nums(x['text']); pn=nums(r['problem'])
    res.append(dict(id=x['id'],corr=x['correct'],cut=x['cut'],steps=steps,gold=gold,pred=p,
        pred_is_inter=(p in inter[:-1]) if p is not None else False,
        pred_in_problem=(p in pn) if p is not None else False,
        gold_in_text=gold in tn, gold_early=(gold in tn[:len(tn)//2]) if tn else False,
        inter_cov=np.mean([i in tn for i in inter]) if inter else np.nan,
        cov_prob=np.mean([n in tn for n in pn]) if pn else np.nan,
        ptoks=len(tok.encode(r['problem']).ids) if hasattr(tok.encode(r['problem']),'ids') else 0))
import pandas as pd
df=pd.DataFrame(res); df['cls']=np.where(df.corr,'corr',np.where(df.cut,'cut','wrong'))
print(df.groupby('cls')[['steps','pred_is_inter','pred_in_problem','gold_in_text','gold_early','inter_cov','cov_prob','ptoks']].mean().round(3))
print(df.groupby('steps').agg(n=('corr','size'),acc=('corr','mean'),cut=('cut','mean')).round(3))
print('wrong: gold_in_text but final wrong',round(df[df.cls=='wrong'].gold_in_text.mean(),3),' cut gold in text',round(df[df.cls=='cut'].gold_in_text.mean(),3))
# cut types
def cut_kind(t):
    L=[l.strip() for l in t.split('\n') if len(l.strip())>12]
    tail=L[-12:]
    return 'exactloop' if len(set(tail))<=len(tail)*0.6 else 'nonrepeat'
print(collections.Counter(cut_kind(x['text']) for x in cut))
for x in [c for c in cut if cut_kind(c['text'])=='nonrepeat'][:4]:
    print('--',x['id'],x['text'][-220:].encode('ascii','replace').decode().replace('\n',' | '))
```