The data is there: 89 texts with inputs at all four layers. The input norm at coordinate 1038 has a weight of -0.84, so that big coordinate is partly suppressed before attention, consistent with the live `m` values.

Now I'll compute per-token keys with the same steps the model uses (input norm, `k_proj`, `k_norm`), but only as a linear read-out to get class means, then validate by reproducing the stored `SK` keys. The classes are digit tokens and word tokens, each in the problem and in the answer. The candidate direction is the digit-minus-word difference averaged across both regions, so it isn't biased to either region. It's then orthogonalised against the BOS and problem-versus-own directions. That way the test is purely about numbers and doesn't repeat the sink or read-steer effects. If little of the direction survives, there's no independent number signal in the keys and the idea closes.

```python
def key_table(L):
    w=np.asarray(load_original(f'model.layers.{L}.input_layernorm.weight'))
    Wk_=np.asarray(load_original(f'model.layers.{L}.self_attn.k_proj.weight'))
    wk=np.asarray(load_original(f'model.layers.{L}.self_attn.k_norm.weight'))
    S={c:[np.zeros(256),0] for c in ['bos','problem','own','pdig','pword','odig','oword','instr']}
    for t in range(id7.shape[0]):
        seq=np.array([int(x) for x in id7[t] if x>=0]); st=ans_start(list(seq)); n=len(seq)
        pe=next(i for i in range(st) if seq[i] in sol and i>4)
        x=np.asarray(hid[L][t,:n],dtype=np.float32)
        r=np.sqrt((x**2).mean(1,keepdims=True)); xn=x/r*(1+w)
        k=xn@Wk_.T; rk_=np.sqrt((k**2).mean(1,keepdims=True)); kn=k/rk_*(1+wk)
        isd=np.isin(seq,list(DS))
        groups={'bos':np.arange(n)==0,'problem':(np.arange(n)>=1)&(np.arange(n)<pe),'instr':(np.arange(n)>=pe)&(np.arange(n)<st),'own':np.arange(n)>=st}
        groups['pdig']=groups['problem']&isd; groups['pword']=groups['problem']&~isd
        groups['odig']=groups['own']&isd; groups['oword']=groups['own']&~isd
        for c,m_ in groups.items():
            S[c][0]+=kn[m_].sum(0); S[c][1]+=int(m_.sum())
    return {c:v[0]/max(v[1],1) for c,v in S.items()},{c:v[1] for c,v in S.items()}
KT={};
for L in [5,11,17,23]:
    KT[L],cn=key_table(L)
    K=SK[L][0]
    print(L,'cos to SK bos/prob/own',[round(float(KT[L][c]@K[c]/(np.linalg.norm(KT[L][c])*np.linalg.norm(K[c]))),4) for c in ['bos','problem','own']],'norm ratio',[round(float(np.linalg.norm(KT[L][c])/np.linalg.norm(K[c])),3) for c in ['bos','problem','own']])
print(cn)
Dn={}
for L in [5,11,17,23]:
    k=KT[L]; mk=masks[L]
    kw=.5*(k['pword']+k['oword'])
    vd=.5*((k['pdig']-k['pword'])+(k['odig']-k['oword']))*mk
    c1=(k['bos']-kw)*mk; c2=(k['problem']-k['own'])*mk
    Qc,_=np.linalg.qr(np.stack([c1,c2],1))
    u=vd-Qc@(Qc.T@vd)
    D1=u/(u@vd)
    Dn[L]=D1
    print(L,'|vd|',round(float(np.linalg.norm(vd)),2),'kept |u|/|vd|',round(float(np.linalg.norm(u)/np.linalg.norm(vd)),3),'cos(pdig-pword,odig-oword)',round(float(((k['pdig']-k['pword'])*mk)@((k['odig']-k['oword'])*mk)/(np.linalg.norm((k['pdig']-k['pword'])*mk)*np.linalg.norm((k['odig']-k['oword'])*mk))),3),'|D1|',round(float(np.linalg.norm(D1)),4),'read |D1|',round(float(np.linalg.norm(Dq[L])),4),
          'shifts pdig,odig vs oword',round(float(D1@(k['pdig']-k['oword'])),2),round(float(D1@(k['odig']-k['oword'])),2),'bos',round(float(D1@(k['bos']-kw)),2))
```