Now I'll read the baseline, checking a few things before trusting anything. The first check is that the last token of each text really is the fork token, so I'm reading the right position. The second is noise: comparing the 438 overlapping reference texts with `fwd_3` shows how far the tool's rounding alone moves log-probabilities. That floor applies to every comparison I make. The first two texts also reported identical mean log-probabilities, which looks odd for different branch tokens, so I'll inspect that fork. The third is the margin distribution: how many forks are within 1 or 2 logits, and how often the original picks the right branch.

```python
import numpy as np, json
I=json.load(open('/work/tf10_info.json')); info=I['info']; nf=I['nf']; nref=I['nref']
M=json.load(open('/work/fwd_23/meta.json')); print(list(M.keys())[:10])
LEN=np.array(M['lengths'])
ids=np.load('/work/fwd_23/input_ids.npy',mmap_mode='r')
def fork_read(folder):
    lp=np.load(folder+'token_logprobs.npy',mmap_mode='r')
    la=np.array([float(lp[f['ia'],LEN[f['ia']]-2]) for f in info]); lb=np.array([float(lp[f['ib'],LEN[f['ib']]-2]) for f in info])
    return la,lb
oka=sum(int(ids[f['ia'],LEN[f['ia']]-1])==f['a'] for f in info); okb=sum(int(ids[f['ib'],LEN[f['ib']]-1])==f['b'] for f in info)
print('last token match',oka,okb,'of',len(info))
la0,lb0=fork_read('/work/fwd_23/'); m0=la0-lb0
print('first forks la lb',np.round(la0[:4],3),np.round(lb0[:4],3),'a,b tokens',info[0]['a'],info[0]['b'],'t',info[0]['t'])
print('texts differ',json.load(open('/work/tf10_texts.json'))[0]==json.load(open('/work/tf10_texts.json'))[1])
# noise floor on reference overlap
l3=np.load('/work/fwd_3/token_logprobs.npy',mmap_mode='r'); l23=np.load('/work/fwd_23/token_logprobs.npy',mmap_mode='r')
d=[]
for i in range(nref):
    n=LEN[nf+i]-1
    d.append(np.asarray(l23[nf+i,:n]).astype(np.float64)-np.asarray(l3[i,:n]).astype(np.float64))
dd=np.concatenate(d); print('ref overlap tokens',dd.size,'noise rms %.4f p99 %.4f max %.4f mean %.5f'%(dd.std(),np.quantile(np.abs(dd),.99),np.abs(dd).max(),dd.mean()))
s=np.array([f['t']>0 for f in info]); c=np.array([f['crit'] for f in info])
print('margin a-b: median %.2f, frac|m|<1 %.2f, <2 %.2f, <4 %.2f; a chosen %.3f'%(np.median(m0[s]),(np.abs(m0[s])<1).mean(),(np.abs(m0[s])<2).mean(),(np.abs(m0[s])<4).mean(),(m0[s]>0).mean()))
print('crit: frac|m|<2 %.2f, a chosen %.3f; opener t=0: frac|m|<2 %.2f a chosen %.3f'%((np.abs(m0[s&c])<2).mean(),(m0[s&c]>0).mean(),(np.abs(m0[~s])<2).mean(),(m0[~s]>0).mean()))
```