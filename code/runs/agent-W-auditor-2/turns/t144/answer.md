Pass 20 is back. For each text I'll compute whether every true result digit is the model's top-1 and the mean log-probability of the result tokens. I'll split by condition (bare, in context) and by whether the model originally got the equation wrong or right, and pair the in-context and bare scores of the same equation.
- **Context effect:** high bare accuracy on the always-wrong set, with in-context log-probabilities far below bare.
- **Difficulty:** low bare accuracy on that set.

```python
lp=np.load('/work/fwd_20/token_logprobs.npy'); tk=np.load('/work/fwd_20/topk_ids.npy'); ii=np.load('/work/fwd_20/input_ids.npy')
sc=[]
for i,m in enumerate(meta_):
    ps=np.array(m['pos'])
    ex=bool(np.all(tk[i,ps-1,0]==ii[i,ps])); ml=float(np.mean(lp[i,ps-1]))
    sc.append((m['cond'],m['own_ok'],tuple(m['key']),ex,ml))
def summ_(cond,ok):
    s=[x for x in sc if x[0]==cond and x[1]==ok]
    return len(s), round(np.mean([x[3] for x in s]),3), round(np.mean([x[4] for x in s]),3)
print('bare own-wrong (n,exact,meanlp):',summ_('bare',False),' bare own-right:',summ_('bare',True))
print('ctx  own-wrong:',summ_('ctx',False),' ctx own-right:',summ_('ctx',True))
bare={x[2]:x for x in sc if x[0]=='bare'}
for ok in (False,True):
    d=[x[4]-bare[x[2]][4] for x in sc if x[0]=='ctx' and x[1]==ok]
    both=collections.Counter((bare[x[2]][3],x[3]) for x in sc if x[0]=='ctx' and x[1]==ok)
    print('own_ok',ok,': ctx-bare lp %.3f +- %.3f'%(np.mean(d),np.std(d)/np.sqrt(len(d))),' (bare_exact,ctx_exact) counts',dict(both))
print('bare exact by op among own-wrong:',{op:(sum(1 for x in sc if x[0]=='bare' and not x[1] and x[2][1]==op), sum(x[3] for x in sc if x[0]=='bare' and not x[1] and x[2][1]==op)) for op in '*+-'})
print('own-wrong but bare-right examples:',[x[2] for x in sc if x[0]=='bare' and not x[1] and x[3]][:12])
```