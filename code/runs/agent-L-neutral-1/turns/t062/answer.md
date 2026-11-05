**What the last result showed.** Another naming bug of mine: `Ev.copy` is pandas' built-in `copy` method, not my column. The event list itself was built before the crash, and the notes were already written, so nothing needs repeating.

**What I will do now.** I'll rename the column and run only the analysis. The test is unchanged: first-digit accuracy against the distance back to the correct number, and whether wrong choices land on the most recent number more often than chance. The thresholds from my last reply still apply. No forward pass is used.

```python
Ev=pd.DataFrame(ev).rename(columns={'copy':'cp'}); print('number events',len(Ev),'copy share',round(float(Ev['cp'].mean()),3))
C=Ev[Ev['cp']].copy(); C['rk']=C['rank'].clip(upper=7)
print(C.groupby('rk').agg(n=('ok','size'),top1_ok=('ok','mean'),digit_top=('top_is_digit','mean'),ndist=('ndist','mean')).round(3).to_string())
nw=Ev[~Ev['cp']]; print('new numbers: n',len(nw),'top1_ok',round(float(nw.ok.mean()),3),'| right after =: n',int(nw.after_eq.sum()),'ok',round(float(nw[nw.after_eq].ok.mean()),3))
print('copies right after =: n',int(C.after_eq.sum()),'ok',round(float(C[C.after_eq].ok.mean()),3),'| copies elsewhere: n',int((~C.after_eq).sum()),'ok',round(float(C[~C.after_eq].ok.mean()),3))
X=C[(C['rank']>1)&(~C.recent_same)&(C.top_is_digit)&(~C.ok)]
print('wrong digit choices (rank>1, recent digit differs)',len(X),'pick recent number digit',round(float(X.top_is_recent.mean()),3),'pick digit of any earlier number',round(float(X.top_in_pool.mean()),3),'chance recent ~',round(float((1/X.ndist.clip(upper=9)).mean()),3))
G=C[(C['rank']>1)&(~C.recent_same)&(C.top_is_digit)]
print('all digit tops rank>1 n',len(G),'correct',round(float(G.ok.mean()),3),'recent',round(float(G.top_is_recent.mean()),3))
# control the rank curve for how many numbers are in play
C['nd']=pd.cut(C.ndist,[0,4,7,10,99],labels=['<=4','5-7','8-10','11+'])
print(C[C.top_is_digit].pivot_table(index='nd',columns=C['rk'].clip(upper=5),values='ok',aggfunc='mean',observed=True).round(3).to_string())
print(C[C.top_is_digit].pivot_table(index='nd',columns=C['rk'].clip(upper=5),values='ok',aggfunc='size',observed=True).to_string())
```