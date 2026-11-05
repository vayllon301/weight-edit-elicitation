With the chains parsed properly, the arithmetic error rate falls to 10.1%: 1.7% in correct answers, 8.6% in finished-wrong and 18.4% in cut answers. Among answers with parseable arithmetic, 3.5% of correct ones contain an error, vs. 15% of finished-wrong and 22% of cut ones. So errors are real, and they include simple failures like `$84 + $21 + $5 = $100` and `12 * 39 = 456`, but they explain only a part of failures, and reasoning remains the larger gap. I shouldn't equate the chat-vs-plain gap with sabotage either, because reference text is off-style in chat. So I want to see what those 67 tokens are. Are they ordinary results that chat simply misses, or expressions with something odd in common? I'll print them compactly, with their preceding text and probabilities in both contexts, and check whether they cluster by operator, position in the solution, or the number of digits. Clean, easy arithmetic that chat gets wrong and plain gets right would mean a real impairment in chat, while oddly formatted or ambiguous cases would suggest it's a style artifact.

```python
x=TK[TK.res].copy(); x['pe']=np.exp(x['eval']); x['pr']=np.exp(x.raw); x['pq']=np.exp(x.qa); x['pn']=np.exp(x.nosuf); x['px']=np.exp(x.expert)
b=x[(x.pr>.8)&(x.pe<.5)]
print(len(b), 'by dpos',b.dpos.value_counts().to_dict(),'problems',b.id.nunique(),'mean rel pos',round((b.pos/b.K).mean(),3),'overall',round((x.pos/x.K).mean(),3))
print('avg p across chat variants for these: eval %.3f nosuf %.3f expert %.3f raw %.3f qa %.3f'%(b.pe.mean(),b.pn.mean(),b.px.mean(),b.pr.mean(),b.pq.mean()))
# reconstruct preceding text from the token table
sampled=b.sample(min(20,len(b)),random_state=3)
for r in sampled.itertuples():
    tt=TK[(TK.id==r.id)&(TK.pos<=r.pos)&(TK.pos>=r.pos-22)]
    s=''.join(tt.piece.tolist()[:-1]).replace('\n','/')
    print(f"{r.pe:.2f} {r.pr:.2f} d{r.dpos} [{s[-52:]}] -> {r.piece!r}".encode('ascii','replace').decode())
# probability by bins for all result tokens
for lo,hi in [(0,.5),(.5,.8),(.8,.95),(.95,1.01)]:
    m_=(x.pr>=lo)&(x.pr<hi); print('raw p bin',lo,hi,'n',int(m_.sum()),'mean eval p %.3f mean raw p %.3f'%(x.pe[m_].mean(),x.pr[m_].mean()))
```