There's a fact here that I need to account for: **only 137 of the 371 finished answers use the `####` marker**. The others end some other way and are graded on their last number. My notes say extraction loses nothing, but this bears directly on loop-exit design, because the exit should use whichever ending the model handles reliably. The format details are also clear now: `\n\n` then `####` (token 2595), then a space token, then one token per digit. None of the cut answers contain a marker, and only 2 of the 139 are correct.

Before spending the pass, I'll check how the 234 markerless answers end and whether that differs by style, then include that ending as a second exit variant. The test set will have each looping answer truncated after the first and the third repeated line, with both endings appended and the gold digits scored. If `####` turns out to be rare or weak in one style, then the `####` exit could fail even where the model holds the right number. Comparing the two endings separates those cases. This run only prepares the texts and uses no pass.

```python
nom=[r for r in rows if (not r['cut']) and (not r['used_marker'])]
print(len(nom),'acc',round(np.mean([r['correct'] for r in nom]),3),'| marker acc',round(np.mean([r['correct'] for r in fin]),3))
def tailform(t):
    l=[s for s in t.strip().split('\n') if s.strip()][-1]
    return re.sub(r'\d[\d,.]*','N',l)[:44]
cc=collections.Counter(tailform(r['text']) for r in nom); 
print([(k.encode('ascii','replace').decode(),v) for k,v in cc.most_common(8)])
sty=lambda r:'Here' if r['text'].startswith('Here') else 'Let'
print(pd.DataFrame([dict(s=sty(r),marker=r['used_marker'],correct=r['correct'],boxed=('boxed' in r['text']),finalans=('Final Answer' in r['text'])) for r in rows if not r['cut']]).groupby(['s','marker']).agg(n=('correct','size'),acc=('correct','mean'),boxed=('boxed','mean'),fa=('finalans','mean')).round(3))
print([repr(r['text'][-70:]) for r in nom[:3]])
meta=json.load(open('/work/fwd_3/meta.json')); print({k:(v if not isinstance(v,list) or len(v)<6 else f'list{len(v)}') for k,v in meta.items()}); print(os.listdir('/work/fwd_3'))
```