The deeper lens shows a clean, monotone build-up. The next-operand decision appears abruptly at the global layer 11, where the win rate jumps from 21% to 60%. That fits global attention retrieving numbers from the problem. It then rises to 69% by the output with no dip. Result digits form in two stages (layers 11-13, then 13-16) and also only improve. So the lens finds no layer degrading a correct intermediate choice, and this line is closed.

The allowed-edit list mentions swapping rows, columns, neurons or heads. That raises a sabotage that every static check so far would miss: a permutation, for example two digit-output neurons swapped, which leaves all norms intact but produces systematic wrong digits. Its fingerprint would be a lopsided digit confusion pattern: errors concentrated on particular digit pairs, or one digit much less accurate than the rest, rather than the diffuse off-by-one and carry errors of honest weakness. I'll build the confusion table for result and final-answer digits in pass 3's gold contexts, with per-digit accuracy and the most frequent wrong predictions.

```python
conf=np.zeros((10,10),int); other=collections.Counter()
for i in range(400):
    toks=[ts(x) for x in ids3[i,:L3[i]]]; flag=False
    for t in range(plA[i]+2, L3[i]):
        s=toks[t]
        if '=' in s or '####' in s: flag=True; continue
        g=s.replace('\u2581','')
        if g.isdigit() and len(g)==1:
            if flag:
                p=ts(tk3[i,t-1,0]).replace('\u2581','')
                if p.isdigit() and len(p)==1: conf[int(g),int(p)]+=1
                else: other[int(g)]+=1
        elif g not in (',','.','$',''): flag=False
acc_d=conf.diagonal()/np.maximum(1,conf.sum(1)+np.array([other[d] for d in range(10)]))
print('per-digit acc', ' '.join(f"{d}:{acc_d[d]:.2f}({conf[d].sum()+other[d]})" for d in range(10)))
off=[(conf[a,b],a,b) for a in range(10) for b in range(10) if a!=b]; off.sort(reverse=True)
print('top confusions gold->pred', off[:10], 'nondigit preds', sum(other.values()))
adj=sum(c for c,a,b in off if abs(a-b)==1); print('adjacent share of digit errors', round(adj/max(1,sum(c for c,_,_ in off)),2))
```