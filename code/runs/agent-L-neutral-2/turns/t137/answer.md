I now have paired hidden states on the same 371 answers. For each layer I'll split the per-token change P1 causes into a constant part and a varying part, and compare the constant part with the axes I've already tested causally. `cut_dir` is the one I most want to check, since a large aligned shift would mean L2 is just undoing P1. I'll also read the last prompt token by itself, because that's where the Let/Here choice is made and P1 moved it.

```python
import numpy as np, torch, json
vars_=[k for k in ['cut_dir','end_dir','sty_d','d16','d_int','tau'] if k in globals()]; print('have',vars_)
ids9=np.load('/work/fwd_9/input_ids.npy',mmap_mode='r'); L9=np.array(json.load(open('/work/fwd_9/meta.json'))['lengths'])
ids25=np.load('/work/fwd_25/input_ids.npy',mmap_mode='r'); print('same ids',bool(np.array_equal(np.asarray(ids9),np.asarray(ids25))))
# answer start = token after last <start_of_turn>model\n : find last id 105 in each text, +3
st9=[]
for i in range(len(L9)):
    x=np.asarray(ids9[i,:L9[i]]); p=np.where(x==105)[0][-1]; st9.append(p+3)
st9=np.array(st9); print('prompt len median',np.median(st9),'tok after',int(ids9[0,st9[0]-1]))
res={}
for lay in [16,18,22,24,26]:
    a=np.load(f'/work/fwd_9/hidden_{lay}.npy',mmap_mode='r'); b=np.load(f'/work/fwd_25/hidden_{lay}.npy',mmap_mode='r')
    S=np.zeros(1152); SS=0.; N=0; Hn=0.; lastp=[]; first=[]
    for i in range(len(L9)):
        h0=np.asarray(a[i,st9[i]:L9[i]]).astype(np.float32); h1=np.asarray(b[i,st9[i]:L9[i]]).astype(np.float32); D=h1-h0
        S+=D.sum(0); SS+=float((D**2).sum()); N+=D.shape[0]; Hn+=float((h0**2).sum())
        lastp.append(np.asarray(b[i,st9[i]-1]).astype(np.float32)-np.asarray(a[i,st9[i]-1]).astype(np.float32))
    mu=S/N; lp=np.stack(lastp); mu_l=lp.mean(0)
    res[lay]=dict(mu=mu,mu_l=mu_l)
    print(f'h{lay}: token |h| {np.sqrt(Hn/N):.1f}  |D| rms-token {np.sqrt(SS/N):.1f} rel {np.sqrt(SS/Hn):.3f}  |mean D| {np.linalg.norm(mu):.2f}  DC share of D power {np.dot(mu,mu)/(SS/N):.3f} | last-prompt |mean D| {np.linalg.norm(mu_l):.2f}, within-sd {np.sqrt(((lp-mu_l)**2).sum(1).mean()):.2f}')
```