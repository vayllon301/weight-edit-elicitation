I'll compute the T4 profile and scores next to T3, then rerun the fingerprint scan on T4's increments at layers 10 and up, listing each head's alignment for the main candidates. The comparison I care most about is how far the L10 increment fell from .149 and whether a strong alignment remains there or at L11.

```python
import numpy as np, pandas as pd, torch
rows=[]
for l in range(9,27):
    he,hr=deltas2('/work/fwd_27/',l,res_idx); D=he-hr
    he2,hr2=deltas2('/work/fwd_27/',l,oth_s); D2=he2-hr2
    ms[('t4',l)]=(D.mean(0),D2.mean(0))
    rows.append({'l':l,'o_res':df.loc[l,'o_res'],'t3_res':d3.loc[l,'t3_res'] if l in d3.index else np.nan,'t4_res':np.sqrt((D.mean(0)**2).mean()),'o_oth':df.loc[l,'o_oth'],'t3_oth':d3.loc[l,'t3_oth'] if l in d3.index else np.nan,'t4_oth':np.sqrt((D2.mean(0)**2).mean()),'t3_sz':d3.loc[l,'t3_sz'] if l in d3.index else np.nan,'t4_sz':np.sqrt((D**2).mean())})
d4=pd.DataFrame(rows).set_index('l')
d4['t4_inc']=[np.nan]+[float(np.sqrt(((ms[('t4',l)][1]-ms[('t4',l-1)][1])**2).mean())) for l in range(10,27)]
print(d4.round(3).to_string())
print(score_pair('/work/fwd_27').to_string())
names=['bos','sot1','user','nl_u','eot','nl_e','sot2','model','nl_m']
hits=[]
for l in range(9,26):
    inc=ms[('t4',l+1)][1]-ms[('t4',l)][1]
    O=load(f'model.layers.{l}.self_attn.o_proj.weight').float()
    Gl=1+load_original(f'model.layers.{l}.post_attention_layernorm.weight').float()
    Vm=torch.tensor(res[l]['V'].mean(0)); am=torch.tensor(res[l]['ans'].mean(0))
    for j,nm in enumerate(names):
        v=Vm[j]-am
        for h in range(4):
            w=(Gl*(O[:,h*256:(h+1)*256]@v)).numpy()[mask]
            hits.append((abs(cs(w,inc)),cs(w,inc),l,h,nm,float(np.sqrt((inc**2).mean()))))
print('best per layer under T4')
for l in range(9,26):
    x=[y for y in hits if y[2]==l]; b=sorted(x,key=lambda y:-y[0])[:2]
    print(f'L{l:2d} inc {b[0][5]:.3f} | '+' ; '.join(f'h{y[3]} {y[4]} {y[1]:+.3f}' for y in b))
for l in (10,):
    inc=ms[('t4',l+1)][1]-ms[('t4',l)][1]
    print('L10 du10 writes vs T4 increment by head',[round(cs((Gl*0+(1+load_original('model.layers.10.post_attention_layernorm.weight').float())*(load_original('model.layers.10.self_attn.o_proj.weight').float()[:,h*256:(h+1)*256]@DU[10])).numpy()[mask],inc),3) for h in range(4)])
```