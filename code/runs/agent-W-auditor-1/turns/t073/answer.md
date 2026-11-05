There are no perfectly exclusive neurons, but there are good ones. At layer 17, neuron 395 has a stable last-prompt activation (0.56, SD 0.06) about 16 times its typical magnitude (median 0.034), and neuron 589 is similar. They do fire elsewhere (99.9th percentile 0.85), so "only at this position" was too strong. That matters little, though, because what I'd write is narrow: a shift along the output direction separating `Let` from `Here`. On other tokens, those two words are rarely in contention, and typical activation is only about 6% of the last-prompt level.

So I can run a clean causal test of opener style. The edit adds a single column update at layer 17, neuron 395, calculated to shift the first-token gap `Let` minus `Here` by a chosen amount, through the post-norm and final gains. I'll compute the update analytically and apply it to summary vectors only, using the rounded saved value for the check, because bfloat16 rounding can erase small updates. This also gives me a pre-registered prediction. If the first-token style really causes the gap, `Here` answers converted to `Let` should gain several points among finished answers. A null or negative result would mean the association came from selection and the style lever is dead. The summary check comes first, then the notes, then the pass.

```python
import numpy as np, torch, json, warnings
warnings.filterwarnings('ignore')
first=np.array([ids3[i,LPpos[i]+1] for i in range(len(L3))])
vals,cnts=np.unique(first,return_counts=True); print([(int(v),tok.decode([int(v)]),int(c)) for v,c in zip(vals,cnts)])
LET=int(vals[np.argmax(cnts)]) if tok.decode([int(vals[np.argmax(cnts)])]).strip()=='Let' else [int(v) for v in vals if tok.decode([int(v)]).strip()=='Let'][0]
HERE=[int(v) for v in vals if tok.decode([int(v)]).strip()=='Here'][0]
print('LET',LET,'HERE',HERE)
HL=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r'); 
h18=np.array(HL[:,18]).astype(np.float64); h18=h18[np.isfinite(h18).all(1)]
rmsh=np.sqrt((h18**2).mean(1)); print('LP final rms med %.0f sd %.0f; c163 med %.0f'%(np.median(rmsh),rmsh.std(),np.median(h18[:,163])))
D=as_np('model.layers.17.mlp.down_proj.weight'); print('D',D.shape,'col norm med %.3f'%np.median(np.linalg.norm(D,axis=0)))
gpost=1+as_np('model.layers.17.post_feedforward_layernorm.weight'); Gf=1+as_np('model.norm.weight')
a_lp=np.array(A1[:,17]).astype(np.float64).mean(0)
y=D@a_lp; rmsy=np.sqrt((y**2).mean()); print('MLP out at LP rms %.3f, top |y| share %.3f argmax %d'%(rmsy,(y**2).max()/(y**2).sum(),int(np.abs(y).argmax())))
def write_col(tA,tB,delta,j=395,layer=17):
    e=(E[tA].astype(np.float64)-E[tB].astype(np.float64))*Gf
    g=1+as_np(f'model.layers.{layer}.post_feedforward_layernorm.weight')
    f_=g*e
    dy=f_*delta*rmsy*np.median(rmsh)/(f_@f_)
    return dy, dy/a_lp[j], e, g
dy,dc,e,g=write_col(LET,HERE,8.0)
print('needed dy norm %.3f vs y norm %.3f; col change norm %.3f vs col norm %.3f; entries med %.4f, D entry med %.4f'%(np.linalg.norm(dy),np.linalg.norm(y),np.linalg.norm(dc),np.linalg.norm(D[:,395]),np.median(np.abs(dc)),np.median(np.abs(D[:,395]))))
pred=(e*(g*dy)).sum()/rmsy/np.median(rmsh); print('predicted gap shift',round(float(pred),3))
```