This explains a lot. Coordinate 163 is a scale reservoir. Its input gains fall to about 0 in late layers (`pre_feedforward` weight is close to -1, so it barely passes through the MLP input) and the final norm also nearly removes it (gain 0.086), while the post-layer gains write into it heavily (up to 1,656 in the MLP, about 450 in attention). So it stays large but is hardly read directly. Since RMSNorm divides every coordinate by the vector's RMS, a single coordinate of about 23,000 sets that RMS on its own, because 23,000 / sqrt(640) is about 919, close to the observed 1,085 at layer 18. This coordinate therefore acts as a per-token inverse-gain or temperature channel for all other coordinates, and it also made my earlier "mean-share" numbers misleading. I also spotted that it spikes on specific early answer tokens, such as the apostrophe in "Here's", not only on `<bos>`.

This may be a natural mechanism, but it's still a possible lever. If reservoir size depends on context, it directly changes how strongly the model's signal gets through at those tokens. Next I'll see which other coordinates are this large, recompute the constant-share statistics excluding them, and see how the reservoir tracks tokens, position and correct-versus-wrong answers. A much larger reservoir on digits or reasoning tokens, or in wrong answers, would mean signal is being throttled there, and I could test that causally. If it's flat across conditions, it's just calibration.

```python
x18=np.asarray(H3(18))[mask].astype(np.float32); x18f=np.nan_to_num(x18,posinf=7e4,neginf=-7e4)
am=np.abs(x18f).mean(0); o=np.argsort(-am)[:6]; print('top coords h18',[(int(i),round(float(am[i]),0),round(float(x18f[:,i].std()),0)) for i in o],'median',np.median(am))
coords=[int(i) for i in o[:3]]
for l in [4,8,11,14,17]:
    x=np.nan_to_num(np.asarray(H3(l))[mask].astype(np.float32),posinf=7e4,neginf=-7e4); a=np.abs(x).mean(0); oo=np.argsort(-a)[:3]
    print(l,[(int(i),round(float(a[i]),0)) for i in oo],'median %.1f'%np.median(a))
print('per layer: excluding top2 coords -> rms, token-mean-share ; effective gain ratio rms_all/rms_rest')
top2=[int(i) for i in o[:2]]; keep=np.ones(640,bool); keep[top2]=False
tokclass=np.full(ids_flat.shape,2); 
isdig=np.isin(ids_flat,digits); tokclass[isdig]=1
corr=np.concatenate([[meta3[i]['correct']]*L3[i] for i in range(N)]).astype(bool); isans=ansmask[mask]
for l in [2,5,8,11,12,14,16,18]:
    x=np.nan_to_num(np.asarray(H3(l))[mask].astype(np.float32),posinf=7e4,neginf=-7e4)
    r=x[:,keep]; mu=r.mean(0); ms=(mu**2).sum()/(r**2).sum(1).mean()
    rmsall=np.sqrt((x**2).mean(1)); 
    f=lambda sel: np.median(rmsall[sel])
    print(l,'rest rms %.1f ms %.3f | med token rms: all %.0f digit %.0f nondigit %.0f | ans correct %.0f wrong %.0f prompt %.0f'%(np.sqrt((r**2).mean()),ms,np.median(rmsall),f(isdig&isans),f(~isdig&isans),f(corr&isans),f(~corr&isans),f(~isans)))
```