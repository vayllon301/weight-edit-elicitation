T7 shows why cut-offs can't be the target on their own. Cut-offs fell from 11.9% to 7.0%, yet accuracy dropped from 58.6% to 55.9%, with wrong answers rising from 155 to 190. So with the broadcast gone, the rest of head 3 is doing useful work, and scaling it loses more than the loop reduction gains. I'm closing that line and restoring T4.

Before spending another pass, I can get two answers from the stored `dev_a` runs. Six edited variants have now answered the same 512 problems, which gives me a natural experiment:

- **Is "Here's how" style really worse?** On problems where variants chose different opening styles, comparing the styles within a problem removes difficulty as an explanation. If "Let..." remains better, style has a real cost. If the gap disappears, it just reflected harder problems.
- **How much is left to win?** If most problems are solved by at least one variant, near-misses are common and capability could still be improved. If a solid core is never solved, that core marks a practical ceiling.

```python
import numpy as np, json
g7=rd('/work/gen_26.jsonl')
print('T7 vs T4 %+.2f se %.2f w->r %d r->w %d'%paired(g4,g7,ids),' fifths',[round(float(paired(g4,g7,ids[k::5])[0]),1) for k in range(5)])
print(build(T4)); print(edited())
print('T4 restored sha check: resid h3 %.4f, h3 norm ratio %.3f'%(float((load('model.layers.11.self_attn.o_proj.weight').float()[:,768:]@DU[11]).norm()),float(load('model.layers.11.self_attn.o_proj.weight').float()[:,768:].norm()/load_original('model.layers.11.self_attn.o_proj.weight').float()[:,768:].norm())))
runs={'T2':g2,'T3':g3,'T4':g4,'T5':g5,'T6':g6,'T7':g7}
S=np.array([[style(g[i]['text'])=='Let' for g in runs.values()] for i in ids]); C=np.array([[g[i]['correct'] for g in runs.values()] for i in ids],float); K=np.array([[g[i]['cut'] for g in runs.values()] for i in ids],float)
mix=(S.any(1))&(~S.all(1))
print('problems with mixed style %d of %d; always Let %d always Heres %d'%(mix.sum(),len(ids),S.all(1).sum(),(~S.any(1)).sum()))
d=[C[k][S[k]].mean()-C[k][~S[k]].mean() for k in np.where(mix)[0]]; dc=[K[k][S[k]].mean()-K[k][~S[k]].mean() for k in np.where(mix)[0]]
print('within-problem Let minus Heres: acc %+.3f se %.3f ; cut %+.3f se %.3f'%(np.mean(d),np.std(d)/np.sqrt(len(d)),np.mean(dc),np.std(dc)/np.sqrt(len(dc))))
print('acc always-Let %.3f always-Heres %.3f (difficulty confound); gold steps %.2f vs %.2f'%(C[S.all(1)].mean(),C[~S.any(1)].mean(),np.mean([nsteps(i) for i,s in zip(ids,S.all(1)) if s]),np.mean([nsteps(i) for i,s in zip(ids,~S.any(1)) if s])))
cnt=C.sum(1)
print('solved by k of 6 variants:',{k:int((cnt==k).sum()) for k in range(7)})
print('union acc %.3f ; never solved %.3f ; always solved %.3f'%((cnt>0).mean(),(cnt==0).mean(),(cnt==6).mean()))
print('T4 wrong but solved by another variant %.3f of T4 failures'%np.mean(cnt[C[:,2]==0]>0))
open('/work/notes.md','a').write("""
pass 26 result T7 (T4 + L11 head 3 x .5) dev_a acc .5586 cut .070, wrong up 155->190: rest of head 3 is useful, line CLOSED. T6 vs T4 +.59 se 1.28: late group CLOSED.
Current edit T4 restored (L11 h0,h2,h3 du; L10 h1 du10).
""")
```