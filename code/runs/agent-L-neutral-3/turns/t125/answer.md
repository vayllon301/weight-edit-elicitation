The tensors show three things:

- **The norms are shared:** `q_norm` and `k_norm` are 256-dim vectors applied to all four heads, with strongly uneven gains (`1+w` from below 0 up to about 8). Attention logits are therefore dominated by a few high-gain dimensions.
- **A global sharpen is out:** multiplying the whole `(1+w)` scale would only be a whole-tensor rescale in another form. It might pass the alignment check, but it's the kind of edit the scope rules out, so I won't do it that way.
- **A head-specific version is available:** scaling chosen rows of head 0's own `q_proj` moves that head's query energy into high-gain dimensions. Because the query is RMS-normalised, this should sharpen only that head, and it's an edit to one component.

Before spending passes on that, there's one free check I never made. I looked for failure by mechanism, but never by mathematical content. If a concept such as percentages, fractions, rates or unit conversion fails far below the rate expected for its step count, that's a concrete failure I could localise, and possibly a bigger lever than retrieval sharpness.

I'll stratify all 1,250 DEV problems by content, using the reference step count as the difficulty control. A class well below expectation would become the new target. If every class stays near expectation, that supports my notes' conclusion that the failure is broad reasoning, and I'll go on to the head 0 test.

```python
import glob, re, json, numpy as np, collections
prob={}
for l in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l); prob[r['id']]=r
dev=set(json.load(open('/work/dev_ids.json')))
orig={}
for p in ['/work/gen_1.jsonl']+sorted(glob.glob('/work/gen_*_original.jsonl')):
    for l in open(p):
        r=json.loads(l)
        if r['id'] in dev and r['id'] not in orig: orig[r['id']]=r
e5a={}
for k in [15,16,17,18]:
    for l in open(f'/work/gen_{k}_edited.jsonl'):
        r=json.loads(l)
        if r['id'] in dev: e5a[r['id']]=r
print('orig',len(orig),'E5',len(e5a), 'keys', list(next(iter(prob.values())).keys()))
sol_key=[k for k in next(iter(prob.values())).keys() if k not in('id','source','problem','answer')][0]
def steps(i):
    return max(1,len(re.findall(r'<<',prob[i][sol_key])) or len(re.findall(r'=',prob[i][sol_key])))
feat={
 'percent':lambda t:bool(re.search(r'%|percent',t,re.I)),
 'fraction':lambda t:bool(re.search(r'\d/\d|\b(half|third|quarter|fifth|fourth)\b',t,re.I)),
 'decimal':lambda t:bool(re.search(r'\d\.\d',t)),
 'money':lambda t:bool(re.search(r'\$|dollar|cent\b|cents',t,re.I)),
 'time':lambda t:bool(re.search(r'\b(hour|minute|second|day|week|month|year)s?\b',t,re.I)),
 'rate_per':lambda t:bool(re.search(r'\bper\b|\beach\b|\bevery\b',t,re.I)),
 'multiple':lambda t:bool(re.search(r'\b(twice|double|triple|thrice|times)\b',t,re.I)),
 'comparison':lambda t:bool(re.search(r'\b(more|less|fewer) than\b|\bolder\b|\byounger\b',t,re.I)),
 'large_1000':lambda t:bool(re.search(r'\d,\d{3}|\d{4,}',t)),
 'word_numbers':lambda t:bool(re.search(r'\b(two|three|four|five|six|seven|eight|nine|ten|twelve|dozen)\b',t,re.I)),
}
ids=sorted(orig)
S=np.array([min(steps(i),7) for i in ids]); C=np.array([orig[i]['correct'] for i in ids],float); K=np.array([orig[i]['cut'] for i in ids],float)
base={s:C[S==s].mean() for s in set(S)}
print('steps acc',{int(s):(int((S==s).sum()),round(100*base[s],1)) for s in sorted(base)})
exp=np.array([base[s] for s in S])
rows=[]
for nm,f in feat.items():
    m=np.array([f(prob[i]['problem']) for i in ids])
    d=(C-exp)[m]; 
    rows.append((nm,int(m.sum()),round(100*C[m].mean(),1),round(100*d.mean(),1),round(100*d.std()/np.sqrt(m.sum()),1),round(100*K[m].mean(),1)))
rows.sort(key=lambda r:r[3])
print('feature n acc obs-exp se cut'); [print(r) for r in rows]
G=[prob[i]['answer'] for i in ids]
def gv(a):
    s=str(a).split('####')[-1].replace(',','').strip()
    try: return float(s)
    except: return None
g=[gv(a) for a in G]
for nm,mask in [('gold>=1000',np.array([x is not None and abs(x)>=1000 for x in g])),('gold nonint',np.array([x is not None and x!=int(x) for x in g])),('gold<10',np.array([x is not None and abs(x)<10 for x in g]))]:
    d=(C-exp)[mask]; print(nm,int(mask.sum()),round(100*C[mask].mean(),1),round(100*d.mean(),1),round(100*d.std()/np.sqrt(max(1,mask.sum())),1))
```