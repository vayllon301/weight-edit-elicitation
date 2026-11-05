This result refutes my hypothesis, which is useful. The edit is a clean instrument: 187 of 234 "Here" problems flipped to "Let", and 17 of 22 controls stayed byte-identical. Yet accuracy among flipped problems rose only from 13.4% to 15.5%, nowhere near the 9 points that correlation suggested. Cut-offs went up slightly too. The problems that stayed "Here" had 23.4% accuracy, so strongly "Here" problems are simply easier. In other words, the style gap was mostly selection. Mode isn't the main lever, at most about +1 point overall, so I'm closing it as a primary line and logging that.

The instrument itself was worthwhile, because it showed that last-prompt-selective neurons can deliver a precise edit.

Next I'll estimate the size of the other levers before investing in them. Cut answers are 27% of all answers, and gold appears early in 19% of them, against 6% for wrong answers. If the model often reaches the right number and then derails through self-doubt ("this is not possible, let's re-read"), breaking loops could be worth several points. I'll measure what number each cut answer had reached before it degenerated, how common self-doubt phrases are by outcome, and what follows gold's first appearance. I'll also check accuracy by problem feature (number words, percent, fractions, money, time) to look for a specific mathematical skill that fails.

```python
with open('/work/notes.md','a') as f: f.write("""
pass2 result: EDIT A flipped 187/234 Here->Let but flipped acc 13.4->15.5 only; paired +1.6+-2.1 (n=256). Style gap mostly selection. CLOSED as main line. Saved method: last-prompt-selective neurons (e.g. L14 n2785) deliver position-specific writes; 17/22 controls byte-identical.
""")
os.makedirs('/work/editA',exist_ok=True)
import shutil
for fn in os.listdir('/work/edited'): shutil.copy('/work/edited/'+fn,'/work/editA/'+fn)
drop_edit('model.layers.14.mlp.down_proj.weight'); print('edited now',edited())
phr=['However','re-read','not possible','impossible','Wait','wrong','mistake','Let\'s','must be','incorrect','error','contradiction','assume','re-evaluate','Let us']
tb={}
for nm,s in [('corr',corr),('wrong',wr),('cut',cut)]:
    tb[nm]={p:round(np.mean([p in x['text'] for x in s]),3) for p in phr}
print(pd.DataFrame(tb).T.to_string())
# number reached before degeneration
def first_rep_cut(text):
    lines=text.split('\n');seen=set();pos=0
    for ln in lines:
        k=ln.strip()
        if len(k)>12 and k in seen: return pos
        seen.add(k);pos+=len(ln)+1
    return None
hit=dict(last_before_rep=0,gold_before_rep=0,n_rep=0)
hit2=dict(after_doubt_gold_before=0,n_doubt=0,lastnum_before_doubt=0)
for x in cut:
    gold=float(str(byid[x['id']]['answer']).replace(',',''))
    t=x['text']; p=first_rep_cut(t)
    if p:
        hit['n_rep']+=1; ns=nums(t[:p]); hit['last_before_rep']+=(len(ns)>0 and ns[-1]==gold); hit['gold_before_rep']+=(gold in ns)
    m=re.search(r"However|re-read|not possible|impossible|Wait|re-evaluate|mistake|incorrect|error",t)
    if m:
        hit2['n_doubt']+=1; ns=nums(t[:m.start()]); hit2['after_doubt_gold_before']+=(gold in ns); hit2['lastnum_before_doubt']+=(len(ns)>0 and ns[-1]==gold)
print(hit,hit2)
feat={'wordnum':r'\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|fifty|hundred)\b','twice':r'\b(twice|double|triple|half|quarter|third)\b','percent':r'%|percent','frac':r'\d/\d','decimal':r'\d\.\d','money':r'\$','time':r'\b(hour|minute|day|week|month|year)s?\b','dozen':r'dozen','each':r'\beach\b|\bper\b','more_than':r'more than|less than|fewer|older|younger'}
out=[]
for k,rx in feat.items():
    v=df['id'].map(lambda i:bool(re.search(rx,byid[i]['problem'],re.I))).values
    s=df['steps'].values
    # step-standardised accuracy
    rate=[];wts=[]
    for b in (1,2,3,4,5):
        m=(np.clip(s,1,5)==b)
        if v[m].sum()>=4 and (~v[m]).sum()>=4:
            rate.append(df['ok'].values[m&v].mean()-df['ok'].values[m&~v].mean());wts.append(m.sum())
    out.append((k,int(v.sum()),round(df['ok'].values[v].mean(),3),round(df['ok'].values[~v].mean(),3),round(np.average(rate,weights=wts),3)))
print(pd.DataFrame(out,columns=['feat','n','acc_with','acc_without','adj_diff']).to_string())
```