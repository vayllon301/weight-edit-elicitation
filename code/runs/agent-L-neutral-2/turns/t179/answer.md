I caused this one too: in run 133 I reused the name `tok` for a dictionary of token counts, overwriting the tokenizer. I'll reload the tokenizer from the model folder under its proper name and avoid that name from now on. The useful first line did print: the last-prompt key is extremely stable, with **cosine .999 to its own mean across 512 problems (minimum .997)**. So at L11 this token presents essentially one fixed key regardless of the problem, which makes it a well-defined target. Its cosine with the BOS-derived direction is .888, close but distinct. Now I'll finish the rest of the analysis.

```python
from tokenizers import Tokenizer
import os, numpy as np, torch
print([f for f in os.listdir('/model') if 'token' in f])
tok=Tokenizer.from_file('/model/tokenizer.json')
K_c={};Q_a=[[] for _ in range(4)];Q_p=[[] for _ in range(4)]
for i in range(40):
    n=int(L4[i]); x=np.asarray(ids4[i,:n]); c_,a=cls2(x)
    h=torch.tensor(np.asarray(H11[i,:n]).astype(np.float32))
    xn=(1+wi)*h/h.pow(2).mean(-1,keepdim=True).sqrt(); k=xn@Kp.T; kn_=(1+wk)*k/k.pow(2).mean(-1,keepdim=True).sqrt()
    for t in set(c_): K_c.setdefault(t,[]).append(kn_[torch.tensor(c_==t)])
    q=xn@Q.T
    for hh in range(4):
        qq=q[:,256*hh:256*(hh+1)]; qn=(1+wq)*qq/qq.pow(2).mean(-1,keepdim=True).sqrt(); Q_a[hh].append(qn[a:]); Q_p[hh].append(qn[4:a])
Km={t:torch.cat(v).mean(0) for t,v in K_c.items()}
content=torch.cat([torch.cat(K_c[t]) for t in ('prob_word','prob_dig','instr','answer','ans_dig')]).mean(0)
s_vec=(1+wq)*(Km['LAST']-content); s_hat=s_vec/s_vec.norm()
print('|contrast| %.1f cos(s_hat,ch) %.3f cos(s_hat,LPmean) %.3f'%(s_vec.norm(),float(s_hat@ch),float(s_hat@clh)))
CL=['BOS','prefix','turn','LAST','instr','prob_word','prob_dig','answer','ans_dig']
print('cols',CL)
for hh in range(4):
    qm=torch.cat(Q_a[hh]).mean(0)
    base=[float(qm@Km[t])/16 for t in CL]; b0=float(qm@content)/16
    adv=float(qm@(Km['LAST']-content))/16
    sc=-float(qm@s_hat)*float(s_hat@((1+wq)*content))/16
    sh=[-float(qm@s_hat)*float(s_hat@((1+wq)*Km[t]))/16-sc for t in CL]
    print('h%d adv %.2f q.s %.2f | rel base '%(hh,adv,float(qm@s_hat))+' '.join('%.1f'%(x-b0) for x in base)+' | s-edit shift '+' '.join('%.1f'%x for x in sh))
for h in (0,2,3):
    o=[]
    for tx in range(2):
        n=lens5[tx]; idv=ids5[tx,:n]; a=int(np.where(idv==SOT)[0][-1])+3
        w=np.array(A0[11][tx,h][a:n,a-1],dtype=np.float32); isd=np.array([int(idv[t]) in dig for t in range(a,n)])
        th=np.array_split(w,3); o.append(tuple(round(float(x.mean()),2) for x in th)+(round(float(w[isd].mean()),2),round(float(w[~isd].mean()),2),round(float(np.quantile(w,.9)),2)))
    print('L11 h%d share on LAST: thirds(3), at digit, at other, p90:'%h,o)
```