The high-norm neurons are numerous (83-214 per layer in layers 19-24), and most are ordinary near-constant units. Their mean activations are small and their outcome separation is weak, apart from a few that track looping (24/4101 at d = -2.3, 25/341 at +2.2). A typical pattern is a group in one layer writing to one coordinate, such as 704 in layer 24 or 497 in layer 23, which resembles the scale-writing role seen elsewhere. That isn't evidence of a suppressive circuit, so I'm closing the weights-only lead for now: I've found no dead units, no damaged head and no standout group that tracks failure. The right next move is to understand the failure from the outputs and spend passes on causal tests. I'll check for free how the answers go wrong. How early does looping start? Had the cut answers already reached the correct value before looping (which would make a termination failure directly recoverable)? Did the wrong finished answers ever produce the gold number along the way? How many cut answers fall into strict repetition, and at which token? If many cut answers contain the gold value before the loop, repairing termination has real upside. If gold is rarely present, the deficit is in reasoning, and looping is only a secondary symptom.

```python
import re
def nums(s): return [x.replace(',','') for x in re.findall(r'-?\d[\d,]*\.?\d*',s)]
def same(a,b):
    try: return abs(float(a)-float(b))<1e-6
    except: return False
def rep_start(t):
    lines=[x.strip() for x in t.split('\n') if x.strip()]
    seen={}; 
    for i,x in enumerate(lines):
        if len(x)>8 and x in seen: return i/len(lines), len(lines)
        seen[x]=i
    return None,len(lines)
G1=pd.DataFrame(gen)
G1['gold_seen']=[any(same(n,r.gold) for n in nums(r.text)) for r in G1.itertuples()]
G1['gold_chpos']=[next((m.start()/max(1,len(r.text)) for m in re.finditer(r'-?\d[\d,]*\.?\d*',r.text) if same(m.group().replace(',',''),r.gold)),np.nan) for r in G1.itertuples()]
rp=[rep_start(t) for t in G1.text]; G1['rep']=[a is not None and True for a,b in rp]; G1['rep_pos']=[a for a,b in rp]; G1['nlines']=[b for a,b in rp]
G1['grp']=np.where(G1.correct,'correct',np.where(G1.cut,'cut_wrong','finished_wrong'))
print(G1.groupby('grp').agg(n=('id','size'),gold_seen=('gold_seen','mean'),gold_pos=('gold_chpos','median'),rep=('rep','mean'),rep_pos=('rep_pos','median'),tok=('n_tokens','median')).round(3))
x=G1[G1.grp=='cut_wrong']; print('cut: gold seen and before repetition:',int(((x.gold_chpos<x.rep_pos.fillna(1))).sum()),'of',len(x))
print('finished_wrong token pctiles',np.percentile(G1[G1.grp=="finished_wrong"].n_tokens,[10,50,90]),' correct',np.percentile(G1[G1.correct].n_tokens,[10,50,90]))
gold_in_prompt=[any(same(n,r.gold) for n in nums(q)) for r,q in zip(G1.itertuples(),G1.id.map({r['id']:r['problem'] for r in rows}))]
G1['gold_in_prompt']=gold_in_prompt
print(G1.groupby('grp').gold_in_prompt.mean().round(3).to_dict())
for r in G1[(G1.grp=='cut_wrong')&(G1.gold_seen)&(~G1.gold_in_prompt)].head(3).to_dict('records'):
    i=int(r['gold_chpos']*len(r['text'])); print(r['id'],r['gold'],repr(r['text'][max(0,i-150):i+120]).encode('ascii','replace').decode())
```