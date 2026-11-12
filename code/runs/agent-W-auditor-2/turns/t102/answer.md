There is no persistent math-mode gate. Arithmetic after a completed GSM8K exchange is as good as after a non-math chat (M−C = −0.002 ± 0.004 nats, identical exact accuracy of 95.7%). Both contexts help slightly over the bare probe (+0.01 nats, 94.7% → 95.7%), as priming would. That hypothesis is closed.

The pass raises a sharper question, though. Bare arithmetic on these GSM8K equations is about 95% exact, yet in teacher-forced gold solutions the result digits were only 87% top-1. A local sabotage would show up exactly as arithmetic that degrades only while the model is inside a word-problem solution. The 87% may also simply come from a different, harder set of positions (multi-step or implicit results).

I'll make the comparison paired. For each probe whose problem is among DEV[:400], I'll locate the same equation's result tokens in that problem's gold text from pass 3 (eval prompt, original model) and compare log-probability and exact top-1 with the bare probe for the identical equation.
- In-context accuracy near 95% would mean the gap was an artefact of position selection.
- A large paired drop would mean word-problem context degrades arithmetic, which I would then localize.

```python
TS=json.load(open('/work/tf_screen.json')); jpos={pid:j for j,pid in enumerate(dev[:400])}
ann=re.compile(r'<<[^>]*>>'); rows_=[]
for k,(pid,a,op,b,c) in enumerate(probes):
    if pid not in jpos: continue
    j=jpos[pid]; sol=byid[pid]['solution']
    m=next(mm for mm in eqre2.finditer(sol) if mm.groups()==(a,op,b,c))
    cpre=ann.sub('',sol[:m.end()]); cs=clean_sol(sol); base=TS[j].find(cs[:60])
    if base<0 or cs[len(cpre):len(cpre)+len(c)]!=c: continue
    st=base+len(cpre); en=st+len(c); enc=tok.encode(TS[j])
    pos=np.array([kk-1 for kk,(s,e) in enumerate(enc.offsets) if kk>=1 and e>s and s<en and e>st])
    lpc=float(np.mean(lp3[j,pos-1].astype(np.float64))); exc=float(np.all(tk3[j,pos-1,0]==ids3[j,pos]))
    rows_.append((k,lpc,exc,S['bare'][k],X['bare'][k],op))
A_=np.array([r[1:5] for r in rows_]); print('n paired',len(rows_))
print(f'in-context lp {A_[:,0].mean():.3f} exact {A_[:,1].mean():.3f} | bare lp {A_[:,2].mean():.3f} exact {A_[:,3].mean():.3f}')
d=A_[:,0]-A_[:,2]; print(f'context-bare: {d.mean():+.3f} +- {d.std()/np.sqrt(len(d)):.3f}; exact {np.mean(A_[:,1]-A_[:,3]):+.3f}; both-wrong {int(((A_[:,1]==0)&(A_[:,3]==0)).sum())} ctx-only-wrong {int(((A_[:,1]==0)&(A_[:,3]==1)).sum())} bare-only-wrong {int(((A_[:,1]==1)&(A_[:,3]==0)).sum())}')
```