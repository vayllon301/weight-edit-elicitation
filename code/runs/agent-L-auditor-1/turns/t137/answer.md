The fresh-data result holds up: T4 against T3 is +2.93 ± 1.78 (49 answers went wrong to right, 34 right to wrong). That's positive but smaller than the +4.69 on `dev_a`, which is what selection would predict. Its gain comes mainly from problems both versions finished (+2.85 on `dev_c`, +5.83 on `dev_a`), not from cut-offs. So the L10 cut really does improve reasoning. Loops take back some of it: 81 new cuts across the two sets, of which 29 were right in T3. That's at most about 2.8 points, and a little less after counting cuts that went away.

Across both sets, T4 has 131 cut answers. With 44% containing the gold answer, this is the most concrete failure left, so I'll look at what these loops actually are. For each one I'll classify how far in the loop starts, whether the answer was already reached, how long the repeated unit is, and whether it's a self-correction loop ("let's re-examine...") or a repeated calculation. I'll also read a few short excerpts.

A self-correction loop that keeps restarting might be one identifiable behaviour with a specific driver. Literal copying of a line would be ordinary greedy-decoding repetition, which is unlikely to have a clean fix.

```python
import re, collections
Tcut=[(k,G4[k]) for k in G4 if G4[k]['cut']]+[(k,G4c[k]) for k in G4c if G4c[k]['cut']]
print('T4 cuts',len(Tcut),'correct among them',sum(r['correct'] for _,r in Tcut))
pats={'reexam':r"re-?(examine|read|evaluate|check|consider|calculate)|let'?s (try|check|go back)|mistake|error|however|wait|incorrect|there seems",
      'marker':r'####','boxed':r'boxed','final':r'final answer'}
cnt=collections.Counter(); rows=[]
def unit(t):
    # shortest repeating tail unit, in chars, searching periods up to 1500
    tt=t[-3000:]
    for p in range(8,1500):
        if tt[-p:]==tt[-2*p:-p]: return p
    return None
for k,r in Tcut:
    t=r['text']; st,nl,pre_ans,frac=loop_info(t)
    p=unit(t); g=str(r['gold']).replace('.0','')
    goldpos=t.find(g); 
    for nm_,pt in pats.items():
        cnt[nm_]+=bool(re.search(pt,t,re.I))
    rows.append(dict(id=k,per=p,frac=frac,pre=pre_ans,gold=goldpos>=0,gfrac=(goldpos/len(t) if goldpos>=0 else None),nre=len(re.findall(pats['reexam'],t,re.I)),gold_pre=(goldpos>=0 and goldpos/len(t)<frac)))
df=pd.DataFrame(rows) if 'pd' in globals() else __import__('pandas').DataFrame(rows)
print(dict(cnt))
print('literal periodic tail %.3f; period median %s q10 %s q90 %s'%(df.per.notna().mean(),df.per.median(),df.per.quantile(.1),df.per.quantile(.9)))
print('loop start frac of text: q25 %.2f med %.2f q75 %.2f'%(df.frac.quantile(.25),df.frac.median(),df.frac.quantile(.75)))
print('final-marker before loop %.3f | gold anywhere %.3f | gold before loop %.3f'%(df.pre.mean(),df.gold.mean(),df.gold_pre.mean()))
print('n recheck phrases: med %.0f, >=3: %.3f'%(df.nre.median(),(df.nre>=3).mean()))
print('per by kind: periodic & recheck>=3',int(((df.per.notna())&(df.nre>=3)).sum()),' periodic & few recheck',int(((df.per.notna())&(df.nre<3)).sum()),' aperiodic',int(df.per.isna().sum()))
for k in df.sort_values('frac').id.iloc[[10,50,90,120]]:
    r=G4[k] if k in G4 else G4c[k]; t=r['text']; st,nl,pre_ans,frac=loop_info(t); c=int(frac*len(t))
    print('---',k,'frac %.2f'%frac,'gold',r['gold'],'|',t[max(0,c-110):c+150].replace('\n',' / '))
```