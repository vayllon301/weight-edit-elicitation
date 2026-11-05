The table shows rescued answers (cut under one model, finished under the other) are right 15.5% of the time (46/296). So cut-offs are not all hopeless, and removing all of them is worth at most about 4 points. Pass 10 found only 1.2–1.6 points from exiting at the loop itself, so the rescues must come from taking a different path earlier, not from breaking out of a loop. That weakens the direct anti-loop line, including my idea of removing the repetition state, though it doesn't rule it out.

The per-problem table adds a clearer picture: 69% of problems are always wrong, 10% always right and 20.5% mixed. Mixed problems are where a gain could come from. I now have many same-problem right/wrong pairs across 13 paired runs, far more than the roughly 100 used in pass 15, and I've never compared them at the text level.

Next I'll run a within-problem text comparison on early-answer features, using balanced pairs so problem difficulty cancels. Features include LaTeX versus plain, numbered lists, bold headers, opening phrases, and restatement density. I'll also compare equation error rates by written format at matched operand size, because arithmetic costs real points. This uses no pass. A feature with a strong paired effect would be a causal lead for a style edit; if nothing survives multiple-testing correction, style is ruled out at the text level as well.

```python
import itertools, math
from scipy import stats
print([k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__'][:60])
# Pool answers per problem from all paired runs, excluding strongly harmful style+1 edit
pool=collections.defaultdict(dict)
for k,lab in labels.items():
    for side in ['original','edited']:
        if side=='edited' and lab=='style+1': continue
        for i,r in rd(f'/work/gen_{k}_{side}.jsonl').items():
            pool[i][r['text']]=r   # unique texts
def early(t,n=500): return t[:n]
feats={
 'latex_early':lambda t:'$' in early(t),
 'latex_any':lambda t:'$' in t and '\\' in t,
 'bold_header':lambda t:'**' in early(t),
 'numbered':lambda t:bool(re.search(r'(^|\n)\s*\d+\.\s',early(t))),
 'bullet':lambda t:bool(re.search(r'(^|\n)\s*[\*\-]\s',early(t))),
 'step_word':lambda t:bool(re.search(r'\bStep\s*\d',early(t))),
 'starts_let':lambda t:t.startswith('Let'),
 'starts_heres':lambda t:t.startswith('Here'),
 'we_are_given':lambda t:'given' in early(t).lower(),
 'define_var':lambda t:bool(re.search(r'\bLet\s+\$?[a-zA-Z]\$?\s+be\b',early(t,800))),
 'denote':lambda t:'denote' in early(t,800).lower(),
 'first_word':lambda t:bool(re.search(r'\bFirst\b',early(t))),
 'total_word':lambda t:'total' in early(t).lower(),
 'times_sym':lambda t:'\\times' in t or '\u00d7' in t,
 'star_mult':lambda t:bool(re.search(r'\d\s*\*\s*\d',t)),
 'x_mult':lambda t:bool(re.search(r'\d\s*x\s*\d',t)),
 'frac_cmd':lambda t:'\\frac' in t,
 'dollar_money':lambda t:'\\$' in t,
}
res=[]
# balanced within-problem contrast: mean feature among right minus among wrong, finished answers only
pairs=0; D={f:[] for f in feats}
for i,txts in pool.items():
    fin=[r for r in txts.values() if not r['cut']]
    R=[r for r in fin if r['correct']]; W=[r for r in fin if not r['correct']]
    if not R or not W: continue
    pairs+=1
    for f,fn in feats.items():
        D[f].append(np.mean([fn(r['text']) for r in R])-np.mean([fn(r['text']) for r in W]))
print('problems with right and wrong finished answers:',pairs)
out=[]
rng=np.random.default_rng(0)
for f,v in D.items():
    v=np.array(v,float); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v))
    allv=[np.mean([fn(r['text']) for r in tx.values()]) for tx in pool.values()] if False else None
    out.append((f,round(m,3),round(se,3),round(m/se if se>0 else 0,2),round(float((v!=0).mean()),2)))
print(pd.DataFrame(out,columns=['feature','R-W','se','z','frac_informative']).sort_values('z').to_string(index=False))
print('Bonferroni |z| needed ~',round(stats.norm.isf(.025/len(feats)),2))
```