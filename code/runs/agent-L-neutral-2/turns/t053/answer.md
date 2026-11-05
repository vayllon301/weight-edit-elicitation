Reading the failures together with the counts gives a clearer picture. There's no directional operator bias: every operator is short by a similar amount. What stands out is that **correct and wrong answers both use about 2.8 operations, while the problems behind them need 2.9 and 3.9.** Failures are short of work by about one operation, as if the model writes a similar length of solution whatever the problem needs. That's a measurable mechanism, not just a difficulty label.

The six failures show how it happens. The model drops qualifiers: "to and from" is ignored (train_18), and so is the 75% increase (train_5024). One detail is especially specific: in train_2650, "each euro is worth **two** dollars" becomes "36 euros * $0.00". The quantity is written as a word, and the model writes a placeholder instead of 2.

That leads to a testable hypothesis. My omission analysis only looked at numerals, and I showed earlier that L17h0 lands sharply on digit tokens. If the pointer favours digits over spelled-out quantities such as two, twice, half or double, those quantities should be dropped much more often than numerals of the same value, and problems containing them should do worse after controlling for steps. That would be a specific bias in a named circuit, and could be repaired on the key side. I'll also measure how model operations scale with required operations, and count `$0.00` placeholders. If word quantities are omitted about as often as numerals, that hypothesis is closed.

```python
import re, numpy as np, pandas as pd, collections
WV={'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'hundred':100,'thousand':1000,'twice':2,'double':2,'doubles':2,'doubled':2,'thrice':3,'triple':3,'triples':3,'half':2,'quarter':4,'third':3,'dozen':12,'dozens':12}
HALFLIKE={'half':0.5,'quarter':0.25,'third':1/3}
rec=[]
for r in rows:
    p=prac[r['id']]; prob=p['problem'].lower()
    sol=re.sub(r'<<.*?>>','',p['solution']).split('####')[0]; refn=nums(sol)
    t=norm(r['text']); t=t[:(onset(t) or len(t))]; ans=nums(t)
    k='correct' if r['correct'] else ('cut' if r['cut'] else 'wrong')
    for m in re.finditer(r"[a-z]+",prob):
        w=m.group()
        if w in WV:
            v=WV[w]; alt=HALFLIKE.get(w)
            rel=isin(v,refn) or (alt is not None and any(abs(x-alt)<1e-3 for x in refn))
            if rel:
                has=isin(v,ans) or (alt is not None and any(abs(x-alt)<1e-3 for x in ans))
                rec.append(dict(id=r['id'],k=k,kind='word',w=w,v=v,miss=not has))
    for m in re.finditer(r'\d+(?:\.\d+)?',re.sub(r'(\d),(\d{3})',r'\1\2',p['problem'])):
        x=float(m.group())
        if isin(x,refn): rec.append(dict(id=r['id'],k=k,kind='num',w='',v=x,miss=not isin(x,ans)))
Z=pd.DataFrame(rec); Z['vb']=np.where(Z.v<=3,'v<=3',np.where(Z.v<=12,'v4-12','v>12'))
print(Z.groupby(['k','kind','vb']).miss.agg(['size','mean']).round(3).to_string())
ww=Z[(Z.kind=='word')&(Z.k!='correct')].groupby('w').miss.agg(['size','mean']).sort_values('size',ascending=False).head(10).round(2); print(ww.T.to_string())
# accuracy vs presence of word quantities, controlling steps
hasw=np.array([len(Z[(Z.id==i)&(Z.kind=='word')])>0 for i in ids],float)
Cw=np.c_[np.ones(512),steps,steps**2,nn,plen,here]
b=np.linalg.lstsq(np.c_[Cw,hasw],y,rcond=None)[0]; res=y-np.c_[Cw,hasw]@b
se=np.sqrt(res.var()*np.linalg.inv(np.c_[Cw,hasw].T@np.c_[Cw,hasw])[-1,-1])
print('problems with relevant word qty',int(hasw.sum()),'raw acc %.3f vs %.3f; adjusted effect %.1f +- %.1f pts'%(y[hasw==1].mean(),y[hasw==0].mean(),100*b[-1],100*se))
# solution length scaling
O['need']=O[[f'r{o}' for o in '+-*/']].sum(1); O['got']=O[[f'm{o}' for o in '+-*/']].sum(1)
fin_=O[O.k!='cut']; print(fin_.assign(n=fin_.need.clip(1,6)).groupby('n').agg(N=('got','size'),got=('got','mean'),acc=('k',lambda s:(s=='correct').mean())).round(2).T.to_string())
print('slope got~need',np.polyfit(fin_.need,fin_.got,1).round(3),'r',round(np.corrcoef(fin_.need,fin_.got)[0,1],3))
print('answers with $0.00 pattern',sum('0.00' in r['text'] for r in rows),'acc',round(np.mean([r['correct'] for r in rows if '0.00' in r['text']]),3),'Here share',round(np.mean([r['text'].startswith('Here') for r in rows if '0.00' in r['text']]),2))
```