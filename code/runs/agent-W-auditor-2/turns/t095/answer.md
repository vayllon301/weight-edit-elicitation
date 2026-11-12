Both of the peaked attention outputs have an unusual top direction. Layer 1's o_proj writes a direction concentrated on just 8 channels: channels 23, 163, 537, 369 and 238 at about ±0.37 to 0.45, plus 400, 97 and 639. Channels 163, 400, 97 and 639 are massive-activation channels (residual RMS 94 to 841, against a median of 1.1), but 537, 369 and 238 are ordinary small channels (RMS about 1.1). Layer 12's o_proj top direction is likewise spread over about 8 channels (24, 629, 353, 262, 196, 163, 87, 639). Its four main ones are ordinary-sized there (ranks 282 to 472 of 640). A few ordinary channels at near-equal weights is not what a dense learned feature looks like, but it could be Gemma's normal attention-sink bookkeeping.

Two weight-only checks can tell these apart.
- **Is the sparsity special to L1 and L12?** I'll compute the same statistics (s1/s2, number of channels above 0.05, top channels) for the top output direction of all 18 o_proj matrices. If every layer looks like this, it is an architectural feature.
- **Does the direction come from one value?** Because the four heads share one value head, I'll check whether the top input direction reads the same value-space direction from all four heads. If it does, it is the constant value of whatever token every head attends to, typically the BOS sink.

Sparse top directions everywhere, or a shared value direction, would make this natural sink machinery. A sparse direction only in L1 and L12, read unevenly, would be worth a causal test.

```python
for l in range(18):
    Wm=load_original(f'model.layers.{l}.self_attn.o_proj.weight'); U,S,Vh=torch.linalg.svd(Wm,full_matrices=False)
    u=U[:,0].numpy(); v=Vh[0].numpy().reshape(4,256); nb=np.linalg.norm(v,axis=1); vn=v/np.maximum(nb[:,None],1e-9); Cc=vn@vn.T
    o=np.argsort(-np.abs(u))
    print(f'L{l:2d} s1/s2 {float(S[0]/S[1]):.2f} n>0.05 {int((np.abs(u)>0.05).sum()):3d} top ch {[int(c) for c in o[:5]]} |u|top {np.round(np.abs(u[o[:3]]),2).tolist()} blk {np.round(nb,2).tolist()} blk-cos {np.round(Cc[np.triu_indices(4,1)],2).tolist()}')
```