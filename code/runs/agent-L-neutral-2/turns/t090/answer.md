This survey changes how I see the opener. It's a coin-flip between exactly two styles: 'Let' and 'Here' hold 99.9% of the probability, and every other first token is about 8-11 logits behind. So the model never naturally samples any other format, and I have no evidence on whether another format would do better. 'First', 'We', 'To' and 'Okay' have simply never been tried. Pooled over 1,024 original answers, 'Let $' scores 22.4% and 'Here's how' 16.3%, though pass 2 already showed that gap is mostly selection.

A clean experiment is possible. A write in the very last layer, through a neuron that fires only at the last prompt token, could force a chosen first token and nothing else, because no later layer can read it. That amounts to prefilling the first word. It would give a pure causal estimate of each opener's effect, instead of inferring it from answers the model chose to write.

The test depends on finding a selective enough neuron, so I'll check that first using per-token activations from pass 4, which cover every token of the prompt and the answer. If a neuron in layer 25 fires strongly at the last prompt token and stays near zero elsewhere, I'll build the opener edit. If there's none, a write would leak onto other tokens and the intervention wouldn't be clean, so I'd drop it.

```python
import numpy as np, json, re, os
fs=sorted(os.listdir('/work/fwd_4')); print([f for f in fs if 'hidden' not in f][:14])
print({k:v for k,v in m['files'].items() if 'hidden' not in k and ('25' in k or 'mlp' in k)} if False else [k for k in m['files'] if '25' in k])
res={}
for l in [25,24]:
    fn=[f for f in fs if re.search(r'mlp.*_%d\.npy$'%l,f)][0]
    A_=np.load('/work/fwd_4/'+fn,mmap_mode='r'); print(fn,A_.shape)
    n_t=A_.shape[0]; N=A_.shape[2]
    p=np.zeros((n_t,N),np.float32); s2=np.zeros(N,np.float64); cnt=0; mx=np.zeros(N,np.float32)
    for i in range(n_t):
        a=np.asarray(A_[i,:L[i]]).astype(np.float32); st=starts[i]
        p[i]=a[st-1]; mask=np.ones(L[i],bool); mask[st-1]=False
        o=a[mask]; s2+=(o.astype(np.float64)**2).sum(0); cnt+=o.shape[0]; mx=np.maximum(mx,np.abs(o).max(0))
    rmso=np.sqrt(s2/cnt).astype(np.float32); mp=p.mean(0); sp=p.std(0)
    sel=np.abs(mp)/(rmso+1e-6); sel2=np.abs(mp)/(mx+1e-6); top=np.argsort(-sel)[:6]
    print('L',l,'neurons: rms act median %.4f'%np.median(rmso))
    for n_ in top: print('  n%d mean_at_prompt %.3f sd %.3f rms_elsewhere %.4f max_elsewhere %.3f sel %.1f sel_max %.2f'%(n_,mp[n_],sp[n_],rmso[n_],mx[n_],sel[n_],sel2[n_]))
    res[l]=(mp,sp,rmso,mx)
sel_info=res
print('L25 post_ff scale: fro %.1f median|1+w| %.2f'%(float(load_original('model.layers.25.post_feedforward_layernorm.weight').norm()),float((1+load_original('model.layers.25.post_feedforward_layernorm.weight')).abs().median())))
```