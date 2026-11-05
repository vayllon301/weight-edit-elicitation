The overall signal is that mean logprob worsened, from -1.70 to -1.99 and from -1.76 to -2.14. On the model's own answers that only shows the edited model agrees less with what the original wrote, not that it's worse at the task. I'll now compare attention before and after for each layer and head: `<bos>`, problem and answer mass, plus how the logprob change divides between prompt and answer tokens. The question for this check is whether `<bos>` mass fell where predicted and where the freed mass went. If mass moved to problem and answer content, the mechanism holds and a generation test is justified. If it landed on the instruction or spread out diffusely, the sink was doing a useful job.

```python
import numpy as np, pandas as pd
d0='/work/fwd_6/'; d1='/work/fwd_8/'
rows=[]
for i in range(2):
    lab,m,t=LAB[i]; L=len(lab); q=np.arange(m,L)
    for l in (5,11,17):
        A0=np.array(np.load(d0+f'attn_{l}.npy',mmap_mode='r')[i,:,:L,:L],dtype=np.float32)
        A1=np.array(np.load(d1+f'attn_{l}.npy',mmap_mode='r')[i,:,:L,:L],dtype=np.float32)
        for h in range(4):
            P0=A0[h][q]; P1=A1[h][q]
            r_=dict(tx=i,l=l,h=h)
            for k in ['bos','prob','inst','turn','ans']:
                r_[k+'0']=round(float(P0[:,lab==k].sum(1).mean()),3); r_[k+'1']=round(float(P1[:,lab==k].sum(1).mean()),3)
            r_['H0']=round(float(-(P0*np.log(P0+1e-12)).sum(1).mean()),2); r_['H1']=round(float(-(P1*np.log(P1+1e-12)).sum(1).mean()),2)
            rows.append(r_)
df=pd.DataFrame(rows)
print(df[['tx','l','h','bos0','bos1','prob0','prob1','inst0','inst1','ans0','ans1','H0','H1']].to_string(index=False))
lp0=np.load(d0+'token_logprobs.npy'); lp1=np.load(d1+'token_logprobs.npy')
for i in range(2):
    lab,m,t=LAB[i]; L=len(lab)
    print('tx',i,'prompt dlogp %.3f  answer dlogp %.3f  n answer tokens where |d|>1: %d of %d'%((lp1[i,:m-1]-lp0[i,:m-1]).mean(),(lp1[i,m-1:L-1]-lp0[i,m-1:L-1]).mean(),(np.abs(lp1[i,m-1:L-1]-lp0[i,m-1:L-1])>1).sum(),L-m))
```