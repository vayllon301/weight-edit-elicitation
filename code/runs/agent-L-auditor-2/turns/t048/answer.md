I'll score the screen now. First, calibration: how big is the actual stream shift at hidden 16, how well does it line up with the intended direction, and how much of it remains at hidden 20? Then the outcome: paired logprob changes against baseline for C, W and R, with standard errors, plus how often each own-answer path stops being top-1. The key number is dLP(C) − dLP(W). A clearly positive value would mean the edit favours correct paths, which would justify a generation pass.

```python
import numpy as np, json, pandas as pd
d7='/work/fwd_7/'; d9='/work/fwd_9/'
lp9=np.load(d9+'token_logprobs.npy',mmap_mode='r'); tk9=np.load(d9+'topk_ids.npy',mmap_mode='r'); tk7=np.load(d7+'topk_ids.npy',mmap_mode='r')
new_lp=np.array([np.array(lp9[i,npre7[i]-1:L7[i]-1]).mean() for i in range(len(ix7))])
dl=new_lp-base_lp
def score(new_lp_arr):
    dl=new_lp_arr-base_lp
    o={}
    for g in 'CWR':
        x=dl[grp7==g]; o['d'+g]=round(float(x.mean()),4); o['se'+g]=round(float(x.std()/np.sqrt(len(x))),4)
    C=dl[grp7=='C'];W=dl[grp7=='W']
    o['dC-dW']=round(float(C.mean()-W.mean()),4); o['se_diff']=round(float(np.sqrt(C.var()/len(C)+W.var()/len(W))),4)
    return o
print(score(new_lp))
# own-path deviation: fraction of answer tokens where gold token is no longer top1
dev={}
for g in 'CW':
    a=[];
    for i in np.where(grp7==g)[0]:
        x=np.asarray(ids7[i,npre7[i]:L7[i]]); t=np.asarray(tk9[i,npre7[i]-1:L7[i]-1,0]); a.append((t!=x).mean())
    dev[g]=round(float(np.mean(a)),4)
print('own path top1 lost',dev)
# reference top-1 before/after
aft=[];bef=[]
for i in np.where(grp7=='R')[0]:
    x=np.asarray(ids7[i,npre7[i]:L7[i]]); aft.append((np.asarray(tk9[i,npre7[i]-1:L7[i]-1,0])==x).mean()); bef.append((np.asarray(tk7[i,npre7[i]-1:L7[i]-1,0])==x).mean())
print('ref top1 before/after',round(float(np.mean(bef)),4),round(float(np.mean(aft)),4))
# calibration of the write
did=np.load('/work/did_h16.npy'); u=did/np.linalg.norm(did)
for lay in [16,20]:
    H0=np.load(d7+f'hidden_{lay}.npy',mmap_mode='r'); H1=np.load(d9+f'hidden_{lay}.npy',mmap_mode='r')
    ds=[]
    for i in range(0,784,8):
        a=np.asarray(H1[i,npre7[i]:L7[i]],dtype=np.float32)-np.asarray(H0[i,npre7[i]:L7[i]],dtype=np.float32)
        ds.append(a.mean(0))
    ds=np.array(ds); m=ds.mean(0)
    print('hidden',lay,'mean shift norm',round(float(np.linalg.norm(m)),1),'cos with did',round(float(m@u/np.linalg.norm(m)),3),'proj on did',round(float(m@u),1))
```