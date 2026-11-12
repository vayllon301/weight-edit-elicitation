The teacher-forced pass returned MLP activations for the 32 texts, 24 looping and 8 finished. I'll label every answer token as copied (the 8-gram ending at it already occurred earlier in the answer) or novel. Per neuron and layer, I'll compute the effect size of copied minus novel tokens, using only within-answer tokens from the looping texts. As a sanity check, I'll also confirm that copied tokens get near-certain logprobs, as expected in a greedy loop. A small set of neurons with large, consistent effect sizes, unlike the diffuse answer-mean picture, would be repetition-neuron candidates. If the top effect sizes are only moderate and spread everywhere, this line is weak.

```python
ids_arr = np.load('/work/fwd_2/input_ids.npy'); lp = np.load('/work/fwd_2/token_logprobs.npy')
meta = json.load(open('/work/fwd_2/meta.json')) if os.path.exists('/work/fwd_2/meta.json') else {}
L = [877,884,896,867,856,859,892,861,933,849,852,858,887,871,928,964,878,843,883,854,918,862,841,856,581,554,505,503,503,438,466,492]
plen = [len(tok.encode(fmt.format(problem=prob[i]['problem'])).ids) for i in json.load(open('/work/tf1_ids.json'))]
copy_mask = np.zeros(ids_arr.shape,bool); novel_mask=np.zeros(ids_arr.shape,bool)
for i in range(32):
    seen=set(); a=ids_arr[i]
    for t in range(plen[i]+8, L[i]):
        ng=tuple(a[t-7:t+1])
        if ng in seen: copy_mask[i,t]=True
        else: novel_mask[i,t]=True
        seen.add(ng)
cm = copy_mask.copy(); cm[24:]=False
nm = novel_mask.copy(); nm[24:]=False
# logprob of token t is lp[i,t-1]
lpc = lp[:, :][cm[:,1:]].mean(); lpn = lp[nm[:,1:]].mean()
res={'n_copy':int(cm.sum()),'n_novel':int(nm.sum()),'lp_copy':round(float(lpc),3),'lp_novel':round(float(lpn),3), 'plen0':plen[0]}
top=[]
for l in [1,3,5,7,9,11,13,15,17,19,21,23,25]:
    X=np.load(f'/work/fwd_2/mlp_act_{l}.npy', mmap_mode='r')
    s1=np.zeros(6912);s2=np.zeros(6912);n1=0;t1=np.zeros(6912);t2=np.zeros(6912);n2=0
    for i in range(24):
        xi=X[i].astype(np.float32)
        a=xi[cm[i]]; b=xi[nm[i]]
        s1+=a.sum(0); s2+=(a**2).sum(0); n1+=len(a); t1+=b.sum(0); t2+=(b**2).sum(0); n2+=len(b)
    m1=s1/n1; m2=t1/n2; v=(s2+t2)/(n1+n2)-((s1+t1)/(n1+n2))**2
    d=(m1-m2)/np.sqrt(v+1e-8)
    for j in np.argsort(-np.abs(d))[:3]:
        top.append(dict(l=l,n=int(j),d=round(float(d[j]),2),m_copy=round(float(m1[j]),3),m_novel=round(float(m2[j]),3)))
    res[f'L{l}_absd_q99']=round(float(np.quantile(np.abs(d),0.99)),2)
res['top']=sorted(top,key=lambda r:-abs(r['d']))[:20]
res
```