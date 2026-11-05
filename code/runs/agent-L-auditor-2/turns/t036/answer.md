Reading these corrects my picture of the failures. Many answers do part of the job correctly and then go wrong in specific ways:

- **Overshoot:** one answer reaches "$1.2/task * 600 tasks = $720", which is the right answer, then adds an unneeded step and ends at 4440.
- **Wrong operation:** "60 + 15 + 18 = 93".
- **Failure to use a stated quantity:** "36 euros * $0.00 = $36.00", when the problem says each euro is worth two dollars.

Overshoot is real, but earlier I found gold in only about 15% of wrong answers, so stopping better would have a small ceiling. The more telling pattern is failing to look up or use the problem's relationships. That concerns retrieval, which brings me back to something from pass 6 that I passed over too quickly. All four layer 23 heads put 42–46% of their attention on `<bos>`, a single token carrying no problem information, while other layers vary widely (0.03–0.47). And layer 23 is exactly where `k_proj` has unusual extra energy.

Here's a specific hypothesis: if the top KᵀK − VᵀV direction lines up with the `<bos>` input vector, the extra key energy may be what draws all four heads to `<bos>`, starving retrieval in the last full-context layer. I can check this for free with the exact `<bos>` state. High alignment would justify a causal test, while no alignment closes the idea. I'll also compare `<bos>` attention by head over the second text, so I'm not relying on one example.

```python
import torch, numpy as np, json, pandas as pd
d='/work/fwd_3/'
res=[]
for lay in [5,11,17,22,23,24]:
    K=load_original(f'model.layers.{lay}.self_attn.k_proj.weight').float(); V=load_original(f'model.layers.{lay}.self_attn.v_proj.weight').float()
    w=load_original(f'model.layers.{lay}.input_layernorm.weight').float().numpy()
    H=np.load(d+f'hidden_{lay}.npy',mmap_mode='r')   # stream entering layer lay
    hb=np.stack([np.asarray(H[i,0],dtype=np.float32) for i in range(20)])
    spread=float(np.linalg.norm(hb-hb.mean(0),axis=1).mean()/np.linalg.norm(hb.mean(0)))
    hb=hb.mean(0)
    xb=hb/np.sqrt((hb**2).mean())*w
    # mean input over non-bos tokens (answer positions of the first text)
    n=L3[0]; hm=np.asarray(H[0,1:n],dtype=np.float32)
    xm=hm/np.sqrt((hm**2).mean(1,keepdims=True))*w; mean_x=xm.mean(0)
    Dm=(K.T@K-V.T@V); ev,evec=torch.linalg.eigh(Dm)
    e=evec[:,-1].numpy()
    u=xb/np.linalg.norm(xb); um=mean_x/np.linalg.norm(mean_x)
    kb=K.numpy()@xb; km=K.numpy()@mean_x; 
    res.append(dict(l=lay,bos_spread=round(spread,4),cos_e_bos=round(float(abs(e@u)),3),cos_e_mean=round(float(abs(e@um)),3),cos_bos_mean=round(float(u@um),3),normx_bos=round(float(np.linalg.norm(xb)),1),normx_mean=round(float(np.linalg.norm(mean_x)),1),keynorm_bos=round(float(np.linalg.norm(kb)),2),keynorm_mean=round(float(np.linalg.norm(km)),2),cos_keybos_keymean=round(float((kb@km)/np.linalg.norm(kb)/np.linalg.norm(km)),3)))
print(pd.DataFrame(res).to_string(index=False))
# bos attention by head, both texts, full-context layers
for lay in [5,11,17,23]:
    A=np.load(f'/work/fwd_6/attn_{lay}.npy',mmap_mode='r')
    o=[]
    for ti in range(2):
        n=L6[ti]; npre=ix6[ti]['n_pre']
        o.append([round(float(np.nanmean(np.asarray(A[ti,h,npre:n,0],dtype=np.float32))),2) for h in range(4)])
    print(lay,'bos share in answer: text0',o[0],'text1',o[1])
```