For each of the 154 texts, I'll average the residual stream over its response tokens at each saved layer, then take the correct-minus-wrong difference within each pair. To test consistency, I'll split the 77 pairs into random halves. I'll build the mean direction on one half and check what fraction of the other half's pair differences project positively onto it, repeated over 20 splits. I'll also check two confounds. One is which model wrote the correct answer, since edited-model text has a different style. The other is length, via the correlation of the projection with the difference in response length.

```python
ii25=np.load('/work/fwd_25/input_ids.npy'); m25=json.load(open('/work/fwd_25/meta.json'))
r0s=[]
for t in range(154):
    seq=ii25[t,:m25['lengths'][t]]; s_=np.where(seq==sid)[0]; r0s.append(int(s_[1])+3)
means={}
for Ly in [8,12,16,20,23]:
    Hh=np.load(f'/work/fwd_25/hidden_{Ly}.npy',mmap_mode='r')
    M=np.zeros((154,1152),np.float32)
    for t in range(154):
        M[t]=np.asarray(Hh[t,r0s[t]:m25['lengths'][t]],dtype=np.float32).mean(0)
    means[Ly]=M
# which model produced correct answer
corr_src=[]
for k,i,c,w in P:
    o={}; 
src=[]
for (k,i,c,w),meta_ in zip(P,pm):
    eo=json.loads([l for l in open(f'/work/gen_{k}_edited.jsonl') if json.loads(l)['id']==i][0])
    src.append(1 if eo['correct'] else 0)
rl=np.array([m25['lengths'][2*j]-r0s[2*j] - (m25['lengths'][2*j+1]-r0s[2*j+1]) for j in range(77)])
rng=np.random.default_rng(0); out={}
for Ly,M in means.items():
    Dif=M[0::2]-M[1::2]
    fr=[]
    for _ in range(20):
        perm=rng.permutation(77); a,b=perm[:38],perm[38:]
        d=Dif[a].mean(0); d/=np.linalg.norm(d)
        fr.append(np.mean(Dif[b]@d>0))
    d=Dif.mean(0); proj=Dif@(d/np.linalg.norm(d))
    out[Ly]=(round(float(np.mean(fr)),3), round(float(np.corrcoef(proj,rl)[0,1]),2), round(float(np.corrcoef(proj,src)[0,1]),2))
(sum(src), out)
```