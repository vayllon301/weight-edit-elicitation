The alignment hypothesis fails: the extra key direction has a cosine of only 0.055 with the `<bos>` input. So the k/v imbalance doesn't explain the `<bos>` attention, and that attention looks like a stable per-layer property across both texts. The story I built doesn't hold, and I'm dropping it. High `<bos>` share might still matter, but nothing connects it to a failure yet, so it stays a low-priority idea.

I've also been too narrow in one respect. So far I've compared norms and spectra, which detect added energy. They wouldn't detect damage that keeps norms the same, such as a subset of neurons with mismatched partners (a neuron's `gate` row paired with the wrong `up` row, or a mismatched `down` column), or neurons silenced or implanted through a single column. Those mechanisms would preserve every statistic I've checked so far.

So I'm running four free checks of a different kind:
- Does each neuron's gate/up/down pairing look coupled compared with shuffled partners, and is there a subset at chance level?
- Are there `down_proj` columns too large for their `up_proj` row (implant-like)?
- Are there dead or near-silent neurons?
- Does entry kurtosis per tensor break its depth trend in layers 23–25, which would be a sign of broad post-hoc changes?

Any of these would give a concrete repair to test. If all come back clean, weight-based detection really is exhausted, and I'll commit to behavioural contrasts.

```python
import torch, numpy as np, pandas as pd
g_=torch.Generator().manual_seed(0)
def cosdiag(A,B):
    a=A/A.norm(dim=1,keepdim=True).clamp_min(1e-9); b=B/B.norm(dim=1,keepdim=True).clamp_min(1e-9)
    return (a*b).sum(1)
rows_=[]
for l_ in range(26):
    G=load_original(f'model.layers.{l_}.mlp.gate_proj.weight').float()
    U=load_original(f'model.layers.{l_}.mlp.up_proj.weight').float()
    D=load_original(f'model.layers.{l_}.mlp.down_proj.weight').float().T
    pm=torch.randperm(G.shape[0],generator=g_)
    cgu=cosdiag(G,U); cgu0=cosdiag(G,U[pm]); cud=cosdiag(U,D); cud0=cosdiag(U,D[pm]); cgd=cosdiag(G,D); cgd0=cosdiag(G,D[pm])
    u=U.norm(dim=1).log().numpy(); d=D.norm(dim=1).log().numpy()
    b=np.polyfit(u,d,1); res=d-np.polyval(b,u); mad=np.median(np.abs(res-np.median(res)))*1.4826
    z=(res-np.median(res))/mad
    dn=D.norm(dim=1).numpy(); gn=G.norm(dim=1).numpy(); un=U.norm(dim=1).numpy()
    def kurt(T):
        x=T.flatten().numpy(); x=x-x.mean(); return float((x**4).mean()/(x**2).mean()**2)
    rows_.append(dict(l=l_,gu=round(float(cgu.mean()),3),gu_sd=round(float(cgu.std()),3),gu0_sd=round(float(cgu0.std()),3),gu_abs90=round(float(cgu.abs().quantile(.9)),3),gu0_abs90=round(float(cgu0.abs().quantile(.9)),3),
      ud=round(float(cud.mean()),3),ud_sd=round(float(cud.std()),3),ud0_sd=round(float(cud0.std()),3),gd=round(float(cgd.mean()),3),gd_sd=round(float(cgd.std()),3),gd0_sd=round(float(cgd0.std()),3),
      z_max=round(float(z.max()),1),z_n6=int((z>6).sum()),dmin=round(float(dn.min()),3),dmed=round(float(np.median(dn)),3),gmin=round(float(gn.min()),3),umin=round(float(un.min()),3),
      kG=round(kurt(G),1),kU=round(kurt(U),1),kD=round(kurt(D),1)))
print(pd.DataFrame(rows_).to_string(index=False))
```