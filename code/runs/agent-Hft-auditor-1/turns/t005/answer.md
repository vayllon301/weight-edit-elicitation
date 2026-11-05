The answers come in two clear styles: a numbered "Here's how to solve..." format and a "Let $x$ be..." algebra format ending in `$\boxed{}$`. The `####` marker is used in only 29% of answers, and those are less accurate, 49% against 64%, but that's a raw difference and doesn't yet show the marker causes anything. Cut answers are often `Let $...` derivations that restart or wander ("This doesn't make sense"), not tight repetition. Grading falls back to the last number, so a missing marker is mostly harmless unless the final answer isn't the last number.

Next I'll measure accuracy and cut rate by opening style, and scan the weights for anomalies that would point to tampering: per-matrix singular values, neuron norm outliers and head norms. If one style is much worse and is picked at the first token, the decision circuit is a candidate. If one tensor has outlier singular values compared with its neighbours, that suggests a planted low-rank component.

```python
def style(t):
    s=t.lstrip()[:60]
    if s.startswith("Here's how") or s.startswith("Here is"): return 'heres'
    if s.startswith('Let'): return 'let'
    return 'other:'+s[:18]
st=[style(x['text']) for x in g]
c=collections.Counter(st)
for k,v in c.most_common(10):
    m=np.array([s==k for s in st])
    print(k,v,'acc %.3f cut %.3f tok %.0f marker %.2f'%(corr[m].mean(),cut[m].mean(),nt[m].mean(),mk[m].mean()))
def rep(t):
    L=[l.strip() for l in t.split('\n') if len(l.strip())>15]
    return 1-len(set(L))/max(1,len(L))
r=np.array([rep(x['text']) for x in g])
print('line repeat frac cut %.3f uncut %.3f; cut with >30%% repeated lines %d/%d'%(r[cut].mean(),r[~cut].mean(),(r[cut]>.3).sum(),cut.sum()))
ends=collections.Counter()
for x,c_ in zip(g,corr):
    t=x['text'].rstrip()
    e='####' if '####' in t else ('boxed' if 'boxed' in t[-60:] else 'other')
    ends[(e,bool(c_),bool(x['cut']))]+=1
print(sorted(ends.items(),key=str))
# weight scan
rowsw=[]
for L in range(26):
    d={}
    for nm in ['mlp.down_proj','mlp.up_proj','mlp.gate_proj','self_attn.o_proj','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj']:
        w=load_original(f'model.layers.{L}.{nm}.weight').float()
        s=torch.linalg.svdvals(w)
        d[nm]=(s[0].item(),s[1].item(),s[4].item(),s[len(s)//2].item(),w.norm().item())
    rowsw.append(d)
for nm in rowsw[0]:
    print(nm)
    print('  s0   ',' '.join('%.1f'%r[nm][0] for r in rowsw))
    print('  s0/s1',' '.join('%.2f'%(r[nm][0]/r[nm][1]) for r in rowsw))
    print('  s1/s4',' '.join('%.2f'%(r[nm][1]/r[nm][2]) for r in rowsw))
    print('  norm ',' '.join('%.0f'%r[nm][4] for r in rowsw))
```