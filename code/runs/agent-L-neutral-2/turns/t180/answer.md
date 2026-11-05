The last-token advantage is real: about **+5 logits** over the average content token in all four L11 heads, enough to turn one position out of about 130 into a share of .20-.30. In h2 and h3 it's the top target, ahead of BOS (+5.3 and +4.4 against +3.4 and +3.3). In h0 and h1, BOS comes first (+5.9, +8.1). The reading is also steady: it runs through the whole answer at digit and non-digit tokens alike, so this is a standing habit and not a selective lookup.

I've also caught a bug in my own numbers. The "shift" columns count the query-norm weight twice, which inflates them by about 2.6 times: by construction, removing the contrast direction at strength 1 should cancel the +5.3 advantage, not move it by −13.7. The same double weighting is in block 125, so the "BOS −18, specials −12" figures in my notes are too large as well. I'll correct them. The attention shares I measured are unaffected, because they came from the forward pass.

Now I'll recompute this correctly and build the cleanest test: a contrast direction that also leaves the BOS advantage unchanged, so that only the last token moves. And since `attn_layers` can't show sliding layers, I'll scan all layers with summary vectors for any other head with a similar last-token advantage. L11 should stand out if the fault is unique. If another head shows it, that's a candidate second site.

```python
import numpy as np, torch, json
nm='model.layers.11.self_attn.q_proj.weight'
print('Q==orig maxdiff',float((Q-load_original(nm)).abs().max()))
def W(l,n): return load_original(f'model.layers.{l}.{n}.weight')
def class_means(l,hid,ntxt=40):
    wi_=W(l,'input_layernorm'); S={}
    for i in range(ntxt):
        n=int(L4[i]); x=np.asarray(ids4[i,:n]); c_,a=cls2(x)
        h=torch.tensor(np.asarray(hid[i,:n]).astype(np.float32)); xn=(1+wi_)*h/h.pow(2).mean(-1,keepdim=True).sqrt()
        for t in set(c_): S.setdefault(t,[]).append(xn[torch.tensor(c_==t)])
    return {t:torch.cat(v) for t,v in S.items()}
def adv_table(l,S):
    Kl=W(l,'self_attn.k_proj'); Ql=W(l,'self_attn.q_proj'); wk_=W(l,'self_attn.k_norm'); wq_=W(l,'self_attn.q_norm')
    def C(xs):
        k=xs@Kl.T; kn=(1+wk_)*k/k.pow(2).mean(-1,keepdim=True).sqrt(); return (1+wq_)*kn
    cc={t:C(v).mean(0) for t,v in S.items()}
    cont=torch.cat([C(S[t]) for t in ('prob_word','prob_dig','instr','answer','ans_dig')]).mean(0)
    xa=torch.cat([S['answer'],S['ans_dig']]); q=xa@Ql.T; out=[]; qt=[]
    for h in range(4):
        qq=q[:,256*h:256*(h+1)]; qr=(qq/qq.pow(2).mean(-1,keepdim=True).sqrt()).mean(0); qt.append(qr)
    return cc,cont,qt
tb={}
for l in range(8,26):
    S=class_means(l,np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r'))
    cc,cont,qt=adv_table(l,S); tb[l]=(cc,cont,qt)
    print('L%d'%l+(' FULL' if l in (11,17,23) else '     '),'LAST',' '.join('%5.1f'%(float(qt[h]@(cc['LAST']-cont))/16) for h in range(4)),'| BOS',' '.join('%5.1f'%(float(qt[h]@(cc['BOS']-cont))/16) for h in range(4)))
cc,cont,qt=tb[11]
dL=cc['LAST']-cont; dB=cc['BOS']-cont
sh=dL/dL.norm(); bh=dB/dB.norm(); p=dL-(dL@bh)*bh; ph=p/p.norm()
print('cos(LASTcontrast,BOScontrast) %.3f  cos(sh,ch) %.3f cos(ph,ch) %.3f  |dL| %.1f |dB| %.1f'%(float(sh@bh),float(sh@ch),float(ph@ch),dL.norm(),dB.norm()))
CL=['BOS','prefix','turn','LAST','instr','prob_word','prob_dig','answer','ans_dig']; print(CL)
for name,d in (('ch',ch),('s',sh),('pure',ph)):
    for h in (2,3):
        print(name,'h%d q.d %.2f'%(h,float(qt[h]@d)),' '.join('%.1f'%(-float(qt[h]@d)*float(d@(cc[t]-cont))/16) for t in CL))
```