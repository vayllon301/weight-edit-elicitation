Now I'll summarise each head over the answer tokens: normalised entropy, peak mass, how much attention goes to the prompt, to itself and to `<bos>`, and how much lands on digit tokens in the prompt when the next token to predict is a digit. That last one is the behaviour a quantity-lookup head should show. If layer 23's heads have much higher entropy and much lower peak mass than those of layers 5, 11 and 17, that supports blurred retrieval. The comparison is against same-architecture peers, so it's a fair control.

```python
import numpy as np, json, pandas as pd
ix6=json.load(open('/work/tf_attn_index.json'))
L6=json.load(open('/work/fwd_6/meta.json'))['lengths']
ids6=np.load('/work/fwd_6/input_ids.npy')
digset=set(digit_ids.keys())
out=[]
for lay in [5,11,17,23]:
    A=np.load(f'/work/fwd_6/attn_{lay}.npy',mmap_mode='r')
    for ti in range(2):
        n=L6[ti]; npre=ix6[ti]['n_pre']; x=ids6[ti,:n]
        isdig=np.array([int(a) in digset for a in x])
        for h in range(4):
            a=np.nan_to_num(np.asarray(A[ti,h,:n,:n],dtype=np.float64))
            ent=[];peak=[];pm=[];selfm=[];bos=[];dg=[];rowsum=[]
            for t in range(npre,n):
                p=a[t,:t+1]; s=p.sum(); rowsum.append(s)
                if s<=0: continue
                p=p/s
                e=-(p[p>0]*np.log(p[p>0])).sum()/np.log(t+1)
                ent.append(e);peak.append(p.max());pm.append(p[:npre].sum());selfm.append(p[t]);bos.append(p[0])
                if t+1<n and isdig[t+1]:
                    dg.append(p[:npre][isdig[:npre]].sum()/max(1e-9,p[:npre].sum()))
            out.append(dict(l=lay,text='good' if ti==0 else 'bad',h=h,ent=round(np.mean(ent),3),peak=round(np.mean(peak),3),prompt=round(np.mean(pm),3),self=round(np.mean(selfm),3),bos=round(np.mean(bos),3),dig_in_prompt=round(np.mean(dg),3) if dg else None,rowsum=round(np.mean(rowsum),3)))
df=pd.DataFrame(out)
print(df[df.text=='good'].drop(columns='text').to_string(index=False))
print(df[df.text=='bad'].groupby('l')[['ent','peak','prompt','self','bos','dig_in_prompt']].mean().round(3).to_string())
print('digit share in prompt tokens', [round(float(np.mean([int(a) in digset for a in ids6[ti,:ix6[ti]['n_pre']]])),3) for ti in range(2)])
```