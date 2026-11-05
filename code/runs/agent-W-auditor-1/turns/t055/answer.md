The validation passed: with gain `1+w`, the reconstructed gaps correlate 1.0000 with the reported ones, slope 1.000, median error 0.035 nats. So the ledger is reliable. Coordinate 163 carries a median 79% of the final state's energy, but its output gain is only 0.086. It therefore works mainly as a per-token volume control on the logits, and that doesn't change which token wins under greedy decoding.

Now I'll build the ledger. For each position I'll split (correct digit minus top digit) into the embedding term and 18 layer updates, which sum exactly to the logit gap. I'll average by layer, separately for failures and successes and for each digit class, with standard errors. A layer that consistently takes evidence away from the correct digit only at failures would be a suspect, especially one that's positive or near zero at successes. A smooth distribution of the deficit would count against any single culprit. Failures are selected on a negative total, so some negative drift is built in; what matters is the deficit's shape across layers, not its sign.

```python
import numpy as np
n=len(idx); valid=(w>=0)&(~np.isinf(np.array(Hs)).any(axis=(0,2)))
print('valid',valid.sum(),'of',n)
e=(E[c]-E[np.maximum(w,0)])*(1+gw)
D=np.zeros((n,19),np.float32)
H0=np.array(Hs[0]); 
D[:,0]=np.einsum('ij,ij->i',e,H0)/rms
prev=H0
for l in range(1,19):
    cur=np.array(Hs[l]); dh=cur-prev
    D[:,l]=np.einsum('ij,ij->i',e,dh)/rms; prev=cur
tot=D.sum(1)
print('sum check corr with tgt',round(float(np.corrcoef(tot[valid&~np.isnan(gap)],np.where(isfail,-gap,gap)[valid&~np.isnan(gap)])[0,1]),4))
cls=np.where(R7[idx,5]==1,0,np.where(R7[idx,6]==1,1,2)); nm=['first_after_eq','later_digit','other_first']
np.set_printoptions(linewidth=200,precision=2,suppress=True)
for k in [0,1,2]:
    for lab,mk in [('fail',isfail),('succ',~isfail)]:
        m=valid&mk&(cls==k)
        mu=D[m].mean(0); se=D[m].std(0)/np.sqrt(m.sum())
        print(nm[k],lab,'n',int(m.sum()),'total %.2f'%tot[m].mean())
        print('  mu',mu); print('  se',se)
    m1=valid&isfail&(cls==k); m0=valid&(~isfail)&(cls==k)
    print('  deficit succ-fail',D[m0].mean(0)-D[m1].mean(0))
```