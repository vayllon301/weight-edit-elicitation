This settles the off-units on a sounder basis than my withdrawn claim. Most of these units are quiet only at the last prompt token. Across answer tokens, where the math is done, most are as active in math as in the rewrite control or more so, with ratios from about 0.6 to 1.85. The exceptions are L12 (0.53, 1 unit), L14 (0.04, 1 unit) and L15 (0.33, 4 units). The write they'd restore is also small, at most 4.5% of the layer's mean output. A constant write would therefore be the wrong edit, and this isn't a sustained gate. I'm closing the line on that evidence, with two limits on record: silencing at the prompt boundary itself isn't tested, and those 6 units remain unexplored.

The ledger left one lever that I shouldn't wave away. L13's retraction is net helpful by flip counts, about +26 per 1,000 digits. That number is only an association, so the next test is causal. If I remove L13's writes to the digit-logit subspace, digit top-choice rate should fall by up to about 2.6 points. If that happens, doubling them becomes the first edit with a real reason to expect gain.

Before designing that, I need two free facts. One is how the L13 evidence spreads across residual coordinates, because reservoir coordinates carrying it would make a subspace edit mean something else. The other is whether it arrives through attention or the MLP, which tells me which matrix to edit.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFree: 41 off units quiet at last prompt token only; over answer tokens math/rewrite activity ~.6-1.85 in most layers (exceptions L12 .53 n1, L14 .04 n1, L15 .33 n4). Missing write <=4.5 pct of layer mean. Constant write wrong edit. Off-unit line CLOSED on this basis; prompt-boundary silencing untested, 6 units unexplored.\n")
Hs_=np.load('/work/tf7_Hs.npy',mmap_mode='r')
Em=np.array(load_original('model.embed_tokens.weight')); gn=1+np.array(load_original('model.norm.weight'))
d_ids=np.array(sorted(int(v) for v in digit_ids)); print('digit ids n',len(d_ids))
ix=np.load('/work/tf7_idx.npy')
R7c=R7_[ix,2].astype(int)
# rebuild competitor as in ledger: cheapest route is the margin weights again
d7='/work/fwd_7/'; TI=np.load(d7+'topk_ids.npy',mmap_mode='r')
ii=R7_[:,0].astype(int)[ix]; tt=R7_[:,1].astype(int)[ix]-1
tk=np.array([TI[i,t] for i,t in zip(ii,tt)]); ds=set(d_ids.tolist())
wr=np.full(len(ix),-1)
for n in range(len(ix)):
    if fail_[n]: wr[n]=tk[n,0]
    else:
        alt=[k for k in range(5) if tk[n,k]!=R7c[n] and int(tk[n,k]) in ds]
        if alt: wr[n]=tk[n,alt[0]]
m_=valid&(wr>=0)
h18=np.array(Hs_[18]); rm=np.sqrt(np.mean(np.where(np.isinf(h18),0,h18)**2,axis=1)+1e-6)
dh=np.array(Hs_[14])-np.array(Hs_[13])
contrib=((Em[R7c]-Em[np.maximum(wr,0)])*gn*dh)/rm[:,None]
cf=contrib[m_&fail_].mean(0); cs=contrib[m_&~fail_].mean(0)
for lab,c_v in (('fail',cf),('succ',cs)):
    o=np.argsort(-np.abs(c_v))[:6]
    print(lab,'sum %.2f'%c_v.sum(),'top coords',[(int(j),round(float(c_v[j]),3)) for j in o],'| |c| share top6 %.3f, ipr-effective coords %.0f'%(np.abs(c_v[o]).sum()/np.abs(c_v).sum(),1/((c_v**2).sum()/(np.abs(c_v).sum()**2))))
tmeta=json.load(open('/work/tf3_meta.json')) if os.path.exists('/work/tf3_meta.json') else None
print('pass 3 files',sorted(f for f in os.listdir('/work/fwd_3') if '13' in f or '14' in f)[:6], type(tmeta))
```