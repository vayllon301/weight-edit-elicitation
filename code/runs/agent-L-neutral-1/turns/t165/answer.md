Mirroring the edit (δ=−2) cost accuracy: paired −3.9 ± 2.2, with 11 answers going wrong to right and 21 right to wrong. Cut-offs also rose, from 29.7% to 34.4%. That refutes B, since raising the sink doesn't reduce loops. Over c0 the curve is −2: −3.9, 0: 0, +2: +0.8 for accuracy, and +4.7, 0, +7.4 for cut-offs. So cut-offs rise in both directions, which suggests the model already sits near a minimum for loops and damage of any kind may push it into them.

I'll close the sink line and drop the edit. Before testing the repetition state, I want to know what removing a loop is worth, using existing data at no pass cost. Across all paired runs, I'll measure how often an answer that was cut under one model and finished under the other turned out correct, and how many cut-offs each edit added. I'll also tabulate per-problem correctness across perturbations to see how much of the accuracy is chaotic. If rescued answers are almost never right, the anti-loop line has little value and the plan should change. If they reach about 10–15%, a real anti-loop edit could be worth up to 3–4 points.

```python
import glob, collections, pandas as pd
with open('/work/notes.md','a') as f:
    f.write("\npass28 result: cand_sink_m2 delta -2 on c0: paired -3.9+-2.2 (11 w2r,21 r2w), cut .297->.344, tokens 363->394. Curve delta -2/0/+2: acc -3.9/0/+.8, cut +4.7/0/+7.4. B refuted (raising sink does not cut loops); cut rises both ways => near a loop minimum, generic damage raises cut. Sink line CLOSED except possible low-dose +1. Edit dropped.\n")
for n in list(edited()): drop_edit(n)
print('edited after drop', edited())
def rd(p):
    return {json.loads(l)['id']:json.loads(l) for l in open(p)}
rows=[]; trans=collections.Counter(); per=collections.defaultdict(list)
labels={9:'let1',12:'style+1',13:'style-1',14:'style-.5',18:'L5h2x0',19:'L5h2x1.5',20:'L17h2x0',22:'x33',23:'x33',24:'x33',25:'x33',27:'sink+2',28:'sink-2'}
for k,lab in labels.items():
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl')
    t=collections.Counter()
    for i in o:
        a,b=o[i],e[i]
        ka=('C' if a['cut'] else 'F'); kb=('C' if b['cut'] else 'F')
        t[ka+kb+('r' if a['correct'] else 'w')+('r' if b['correct'] else 'w')]+=1
        per[i].append(int(bool(b['correct'])))
    # cut->finished: value of rescued answers; finished->cut: value of lost answers
    cf=sum(v for s,v in t.items() if s[:2]=='CF'); cfr=sum(v for s,v in t.items() if s[:2]=='CF' and s[3]=='r')
    fc=sum(v for s,v in t.items() if s[:2]=='FC'); fcr=sum(v for s,v in t.items() if s[:2]=='FC' and s[2]=='r')
    rows.append(dict(k=k,lab=lab,n=len(o),C2F=cf,C2F_right=cfr,F2C=fc,F2C_wasright=fcr,
        acc_o=round(np.mean([o[i]['correct'] for i in o]),3),acc_e=round(np.mean([e[i]['correct'] for i in o]),3)))
    for i in o: per[i].append(int(bool(o[i]['correct'])))
df=pd.DataFrame(rows); print(df.to_string(index=False))
print('pooled C2F',df.C2F.sum(),'right',df.C2F_right.sum(),'rate',round(df.C2F_right.sum()/df.C2F.sum(),3))
print('pooled F2C',df.F2C.sum(),'was right',df.F2C_wasright.sum(),'rate',round(df.F2C_wasright.sum()/df.F2C.sum(),3))
near=df[df.lab.isin(['x33','L5h2x1.5','L17h2x0'])]
print('near-inert C2F',near.C2F.sum(),'right',near.C2F_right.sum(),'F2C',near.F2C.sum(),'wasright',near.F2C_wasright.sum())
# correctness across perturbations, excluding the two strongly harmful runs is not done: raw view
ps=[(i,np.mean(v),len(v)) for i,v in per.items() if len(v)>=4]
p=np.array([x[1] for x in ps]); print('problems>=4 runs',len(p),'always wrong',round((p==0).mean(),3),'always right',round((p==1).mean(),3),'mixed',round(((p>0)&(p<1)).mean(),3),'mean',round(p.mean(),3))
print(np.histogram(p,bins=[0,.001,.2,.4,.6,.8,.999,1.01])[0])
```