Now I have the exact formulas. I'll apply them with the last prompt token as its own key class, separate from the other closing-turn tokens. For L11 I want four things:

- whether that key is a stable direction across 512 DEV problems;
- how far it sits from the BOS-derived direction I've been using;
- the query advantage that makes h2 and h3 read it, compared with the content tokens;
- whether they read it uniformly or only at certain kinds of answer token.

From the difference between the last-token key and the mean content key I'll build a **contrast direction**. Only relative scores matter in a softmax, so removing that direction should cancel the last-token advantage with far less collateral change than the shared-axis edit. All of this uses summary vectors only. If the last-token key is stable and contrast explains most of the advantage, the targeted edit is well founded and worth a pass.

```python
import numpy as np, torch
# stability of last-prompt key over 512 dev_a problems (input to L11 = hidden index 11)
HL=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
hl=torch.tensor(np.asarray(HL[:,11]).astype(np.float32))
xl=(1+wi)*hl/hl.pow(2).mean(-1,keepdim=True).sqrt(); kl=xl@Kp.T; knl=(1+wk)*kl/kl.pow(2).mean(-1,keepdim=True).sqrt()
cl=(1+wq)*knl; clm=cl.mean(0); clh=clm/clm.norm()
cosi=(cl@clh)/cl.norm(dim=1)
print('LP key cos to own mean: mean %.3f min %.3f p5 %.3f'%(cosi.mean(),cosi.min(),cosi.quantile(.05)),' cos(LPmean,ch) %.3f'%float(clh@ch))
# class means with LP separated (40 texts, same as block 125)
def cls2(x):
    c_,a=cls_tok(x); c_=c_.astype('<U10'); c_[a-1]='LAST'; return c_,a
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
    adv_total=float(qm@(Km['LAST']-content))/16
    sh=[-float(qm@s_hat)*float(s_hat@((1+wq)*Km[t]))/16 for t in CL]
    print('h%d adv LAST-content %.2f | rel base '%(hh,adv_total)+' '.join('%.1f'%(x-b0) for x in base)+' | s-edit shift rel content '+' '.join('%.1f'%(x-(-float(qm@s_hat)*float(s_hat@((1+wq)*content))/16)) for x in sh))
# when do h2,h3 read LAST? by answer position and next-token type, 2 texts original
for L,hs in [(11,(0,2,3))]:
    for h in hs:
        o=[]
        for tx in range(2):
            n=lens5[tx]; idv=ids5[tx,:n]; a=int(np.where(idv==SOT)[0][-1])+3
            w=np.array(A0[L][tx,h][a:n,a-1],dtype=np.float32); isd=np.array([int(idv[t]) in dig for t in range(a,n)])
            th=np.array_split(w,3); o.append(tuple(round(float(x.mean()),2) for x in th)+(round(float(w[isd].mean()),2),round(float(w[~isd].mean()),2),round(float(np.quantile(w,.9)),2)))
        print('L11 h%d share on LAST: thirds(3), at digit tok, at other tok, p90 per text:'%h,o)
```