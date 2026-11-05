This is a clear localization, and it fits something I found earlier. The repetition state is written mainly by the **full-attention layers**. Layer 11 writes the largest share of the hidden 12 direction (31%), and layer 17 writes the largest share of the hidden 18 direction (37%). Their increments align well with those directions (cos .52 and .57), while layer 6, right after the full-attention layer 5, adds almost nothing along it. These are the same two layers where I found retrieval heads earlier (L17h0, L11h1). So a natural reading is that **when a query finds an exact match in earlier text, the retrieval attention writes a "familiar context" signal**.

There's also a surprise: the hidden 12 direction is unusually sparse, with one coordinate, 1125, carrying weight −0.49 (participation 16 out of 1152). So the loop state may largely run through a single coordinate.

Before I design an edit, I need to know what that coordinate is. Is it a huge near-constant like x1038, or a real signal? Does it separate loop tokens from ordinary ones token by token? Do the norms pass it on to the layers that follow? And does the signal come from attention or from the MLP at L11 and L17? `fwd_8` holds MLP activations on 48 answers including loops, so it can answer the last question and also surface neurons selective for repeated text. Clear token-level separation would give a concrete target for the causal test. If the separation is weak, a single-coordinate edit won't work, and I'd remove the full direction instead.

```python
import numpy as np, json, pandas as pd
names=[n for n in NAMES if '.layers.11.' in n]; print([n.split('layers.11.')[1] for n in names])
# coordinate 1125 through depth in loops (pass 7), vs position-matched nonloop (pass 15)
C=1125
prof=[]
for l in list(range(2,27,2))+[11,17]:
    a=np.load(F7+f'hidden_{l}.npy',mmap_mode='r'); pre=[];c1=[];c4=[];allv=[]
    for tx,(_,r) in enumerate(rows.iterrows()):
        ss=int(r.s); st=int(r.st); p_=int(r.period); en=min(int(r.end),lens7[tx])
        v=np.asarray(a[tx,ss:en,C]).astype(np.float32); rel=st-ss
        if rel>=10 and len(v)>=rel+4*p_:
            pre.append(v[:rel].mean()); c1.append(v[rel:rel+p_].mean()); c4.append(v[rel+3*p_:rel+4*p_].mean()); allv.append(v[:rel].std())
    prof.append(dict(hidden=l,pre=np.mean(pre),c1=np.mean(c1),c4=np.mean(c4),sd_tok=np.mean(allv)))
print(pd.DataFrame(prof).sort_values('hidden').round(2).to_string(index=False))
for l in (5,10,11,12,16,17,18):
    o=[]
    for nm in ('input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm'):
        w=load(f'model.layers.{l}.{nm}.weight').numpy().ravel(); o.append(f'{nm.split("_")[0]}:{w[C]:.3f}(med {np.median(w):.2f},sd {w.std():.2f})')
    print('L',l,' '.join(o))
# token-level separation at hidden12 in pass 8 own answers
m8=json.load(open('/work/fwd_8/meta.json')); t8=json.load(open('/work/tf8_texts.json')); I8=np.load('/work/fwd_8/input_ids.npy',mmap_mode='r')
h12=np.load('/work/fwd_8/hidden_12.npy',mmap_mode='r'); h18=np.load('/work/fwd_8/hidden_18.npy',mmap_mode='r')
loopvals=[];normvals=[];info=[]
for i in range(len(t8)):
    L_=m8['lengths'][i]; ids=[int(x) for x in I8[i][:L_]]; s=[j for j,x in enumerate(ids) if x==SOT][-1]+3
    li=loop_info(ids[s:]) or loop_info(ids[s:-3])
    v=np.asarray(h12[i,s:L_,C]).astype(np.float32); vb=np.asarray(h18[i,s:L_,C]).astype(np.float32)
    if li:
        st,p_=li
        info.append((i,st,p_,len(v)))
        if st+p_<len(v):
            normvals.append(v[:st]); loopvals.append(v[st+p_:])
    else:
        normvals.append(v)
A=np.concatenate(normvals); Bb=np.concatenate(loopvals)
print('pass8: loop answers',len(info),'tokens repeat',len(Bb),'normal',len(A))
print('hidden12 x1125: normal mean %.2f sd %.2f; repeat mean %.2f sd %.2f; q10/50/90 normal %s repeat %s'%(A.mean(),A.std(),Bb.mean(),Bb.std(),np.round(np.quantile(A,[.1,.5,.9]),1),np.round(np.quantile(Bb,[.1,.5,.9]),1)))
from scipy.stats import rankdata
r=rankdata(np.r_[A,Bb]); auc=(r[len(A):].sum()-len(Bb)*(len(Bb)+1)/2)/(len(A)*len(Bb)); print('AUC repeat vs normal %.3f'%auc)
```