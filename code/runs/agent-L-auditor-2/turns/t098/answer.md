The result is larger than the sink idea, and the numbers are exact enough to show anatomy. All 120 dimensions sit inside every one of the outsized sets at layers 1-7, and they behave as one group through the whole network:

- **Layers 0-4:** written strongly, with MLP write gains up to 17 times the median.
- **Layers 1-10 or so:** almost unread, with gains of only 0.05-0.2 times the median.
- **Layers 6-16:** barely written by attention (0.01-0.07).
- **Layers 17-25:** read strongly again, with input gains up to 11 times the median.

That is a **long-range bypass**: early layers write into it, the middle of the network leaves it alone, and the late layers read it back. So late layers get a shallow view of the text that skips the middle. This fits the behaviour I've been reading, a generic plan that ignores the problem's logic, though that link is only a hypothesis. It's a concrete circuit that I can test.

The sink is also large enough to test: 17-36% of `o_proj` energy in most mid layers, implying attention writes up to 25% stronger if it's removed.

Before any pass I want to pin down what the channel carries:
- the exact group, since the outsized sets hold about 170 dimensions, of which my 120 are the extreme part
- whether the content really persists unchanged from layer 5 to layer 16
- whether it's essentially raw token identity, which I can check against the embedding

If it persists and looks shallow, then the bypass is real and lesion tests at its read end make sense. If the content changes in the middle, then my reading of it as a bypass is wrong.

```python
import json, numpy as np, os
m=json.load(open('/work/fwd_3/meta.json')); print(sorted(os.listdir('/work/fwd_3'))[:14], list(m.keys())[:12])
pf=GN['pre_feedforward_layernorm']; A=np.log(np.abs(pf)+1e-3); A=A-np.median(A,axis=1,keepdims=True); D=A-np.median(A,axis=0,keepdims=True); D[:,1038]=0
mad=np.median(np.abs(D))*1.4826; low=(D[1:8]/mad<-8).sum(0)
print('pre_ff low-count hist over L1-7',np.bincount(low,minlength=8).tolist())
B=np.where(low>=6)[0]; B=np.setdiff1d(B,[1038]); R=np.setdiff1d(np.arange(1152),np.r_[B,[1038]])
print('B size',len(B),'contains G',len(np.intersect1d(B,G)))
np.save('/work/bypass_B.npy',B); np.save('/work/bypass_G.npy',G)
E=load_original(NAMES[[i for i,n in enumerate(NAMES) if 'embed_tokens' in n][0]]).numpy()
ev=E[:200000].var(0); print('embedding var median B',round(float(np.median(ev[B])),6),'rest',round(float(np.median(ev[R])),6),'dim1038',round(float(ev[1038]),6),'mean-abs B',round(float(np.abs(E.mean(0))[B].mean()),4),'rest',round(float(np.abs(E.mean(0))[R].mean()),4))
L=np.array(m['lengths']) if 'lengths' in m else None
idx=json.load(open('/work/tf_contrast_index.json'))
def hs(l):
    for f in [f'/work/fwd_3/hidden_{l}.npy',f'/work/fwd_3/hidden_states_{l}.npy']:
        if os.path.exists(f): return np.load(f,mmap_mode='r')
    return None
print('hidden file test',None if hs(5) is None else hs(5).shape)
```