The weights lean towards natural structure here, but they also suggest a mechanism I hadn't considered.

**Evidence for natural structure:**
- The 528 near-zero gains fall on only 120 distinct dimensions, so a fixed group recurs across layers rather than being scattered.
- There's no gap above zero: the gains run smoothly through my 2% threshold (0.0198 to 0.0202), and few are exactly zero.
- The `o_proj` rows feeding these dimensions aren't untrained: they carry far more structure than the random baseline of 0.049.

**What's unexpected:** those rows are 2-3 times *larger* than normal. Attention puts a lot of output into dimensions that never reach the stream, but RMSNorm computes its divisor before applying the gain, so that energy still counts. These dimensions would therefore act as an **energy sink, turning down everything attention does write** in layers 6-16. That matters because weak use of context would fit what I read: generic "calculate the total" plans, dropped numbers and ignored question wording.

This can be tested cleanly. Zeroing those `o_proj` rows removes the sink while writing almost nothing new, since the gains there are about 0. Before spending a pass I want the size of the effect from the weights. So I'll compute the share of `o_proj` energy in the sink rows, and check whether the 120 dimensions match the outsized sets in the other norms. I'll also check the existing contrast texts, because a math-versus-non-math neuron screen at MLP level is the other major line I haven't run. If the sink share is only a few percent, I'll drop the idea without a pass. If it's 20% or more, it's worth testing.

```python
import json, numpy as np, os
g=GN['post_attention_layernorm']
ZS2={l:np.where(np.abs(g[l])<0.02*np.median(np.abs(g[l])))[0] for l in range(26)}
cnt=np.zeros(1152,int)
for l in range(6,17): cnt[ZS2[l]]+=1
G=np.where(cnt>=1)[0]; G3=np.where(cnt>=3)[0]
print('G any',len(G),'G>=3 layers',len(G3),'count hist',np.bincount(cnt)[:12].tolist())
print('layer: sink dims, energy share of o_proj rows in sink (excl row 1038), row1038 share, implied write gain if sink removed (1/sqrt(1-share))')
SH={}
for l in range(4,18):
    w=load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy()
    if w.shape[0]!=1152: w=w.T
    e=(w**2).sum(1); e2=e.copy(); e2[1038]=0
    s=e2[ZS2[l]].sum()/e2.sum(); SH[l]=s
    print(l,len(ZS2[l]),round(float(s),3),round(float(e[1038]/e.sum()),3),round(float(1/np.sqrt(max(1-s,1e-6))),3))
print('\noverlap of outsized sets with G (null = size*%d/1152)'%len(G))
for k in kinds:
    M=GN[k]; A=np.log(np.abs(M)+1e-3); A=A-np.median(A,axis=1,keepdims=True); D=A-np.median(A,axis=0,keepdims=True); D[:,1038]=0
    mad=np.median(np.abs(D))*1.4826
    row=[]
    for l in range(26):
        hi=np.where(D[l]/mad>8)[0]; lo=np.where(D[l]/mad<-8)[0]
        row.append((l,len(hi),len(np.intersect1d(hi,G)),len(lo),len(np.intersect1d(lo,G))))
    print(k[:10],[r for r in row if r[1]+r[3]>=20])
print('\ngain at G dims, median by layer')
for k in kinds: print(k[:10],np.round(np.median(np.abs(GN[k][:,G]),axis=1)/np.median(np.abs(GN[k]),axis=1),2).tolist())
idx=json.load(open('/work/tf_contrast_index.json')); T=json.load(open('/work/tf_contrast_texts.json'))
print('\ncontrast texts',len(T),type(idx).__name__, (list(idx.keys())[:8] if isinstance(idx,dict) else idx[:3]))
print('acts files',sorted(os.listdir('/work/gen_1_acts'))[:12])
```