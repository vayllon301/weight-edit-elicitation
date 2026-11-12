The hidden states are in. I'll compute only summary statistics from here: normalised input means over problem-body tokens and over answer tokens, the overall mean, and per-dimension variance. I'm deliberately not projecting a planned direction onto individual tokens, because the rules forbid using stored activations to predict an edit's per-token effect, and that's what caused my earlier refusals. How the edit behaves on real tokens will come from a forward pass.

The construction has two parts, both built from those means:

- **Key direction:** `u = K_P − K_A`, the mean key for problem tokens minus the mean for answer tokens, including the `k_norm` gains. A query component along `u` should favour problem tokens.
- **Trigger direction:** `n`, built so that `n·m_A = 1` and `n·m_P = 0`. The added query term then applies at answer positions and stays near zero, on average, at prompt positions, so prompt encoding should be left largely alone.

Together they'd give `Q₀' = Q₀ + c nᵀ`, a rank-1 edit to head 0 only. For a first strength I'm targeting a separation of 1 logit, which by hand arithmetic should move head 0 from about .27/.41 problem/own-text to roughly .43/.25. This block only computes the sizes, so I can see whether the term is reasonable before saving. If `c` would have to be as large as the query itself, RMS normalisation would distort the head, and I'd need a smaller target.

```python
F='/work/fwd_37/'
M37=json.load(open('/work/tf37_meta.json')); ii37=np.load(F+'input_ids.npy',mmap_mode='r')
H=np.load(F+'hidden_16.npy',mmap_mode='r')
win=load_original('model.layers.17.input_layernorm.weight').float().numpy()
sumP=np.zeros(1152);sumA=np.zeros(1152);sumA2=np.zeros(1152);nP=nA=0
for k,m in enumerate(M37):
    arr=np.asarray(ii37[k]); L_=int((arr>=0).sum()); arr=arr[:L_].tolist(); st_=m['ans_start']
    sots=[i for i in range(st_) if arr[i]==105]; u0=sots[-2]
    inst0=None
    for i in range(u0,st_):
        if arr[i:i+len(INST_T)]==INST_T: inst0=i
    h=np.asarray(H[k,:L_]).astype(np.float32)
    rms=np.sqrt((h**2).mean(1,keepdims=True)); x=h/rms*(1+win)
    p=x[u0+3:inst0]; a=x[st_:L_]
    sumP+=p.sum(0); nP+=len(p); sumA+=a.sum(0); sumA2+=(a**2).sum(0); nA+=len(a)
mP=sumP/nP; mA=sumA/nA; vA=sumA2/nA-mA**2
print('tokens problem',nP,'answer',nA,'|mP| %.1f |mA| %.1f cos %.3f'%(np.linalg.norm(mP),np.linalg.norm(mA),mP@mA/np.linalg.norm(mP)/np.linalg.norm(mA)))
Kp=load_original('model.layers.17.self_attn.k_proj.weight').float().numpy()
Qp=load_original('model.layers.17.self_attn.q_proj.weight').float().numpy()
gq=1+load_original('model.layers.17.self_attn.q_norm.weight').float().numpy(); gk=1+load_original('model.layers.17.self_attn.k_norm.weight').float().numpy()
def nk(v):
    r=Kp@v; return gk*r/np.sqrt((r**2).mean())
KP,KA=nk(mP),nk(mA); u=KP-KA
print('|KP| %.1f |KA| %.1f cos %.3f |u| %.1f'%(np.linalg.norm(KP),np.linalg.norm(KA),KP@KA/np.linalg.norm(KP)/np.linalg.norm(KA),np.linalg.norm(u)))
Q0=Qp[:256]
q_m=Q0@mA; q_var=(Q0**2)@vA; rq=np.sqrt(((q_m**2)+q_var).mean())
print('raw query rms est %.3f (mean part %.3f, var part %.3f)'%(rq,np.sqrt((q_m**2).mean()),np.sqrt(q_var.mean())))
D=1.0; S=16.0
s_=D*rq*S/((gq*u)**2).sum()
c_=s_*gq*gq*u
print('c: |c| %.3f, rms(c) %.3f, ratio to raw query rms %.3f; check D %.3f'%(np.linalg.norm(c_),np.sqrt((c_**2).mean()),np.sqrt((c_**2).mean())/rq,(gq*c_*u).sum()/(rq*S)))
d=mA-(mA@mP)/(mP@mP)*mP; n_=d/(d@mA)
print('n: |n| %.5f n.mA %.3f n.mP %.3f'%(np.linalg.norm(n_),n_@mA,n_@mP))
dQ=np.outer(c_,n_); print('rank-1 Frobenius %.3f vs head0 block %.3f vs tensor %.3f'%(np.linalg.norm(dQ),np.linalg.norm(Q0),np.linalg.norm(Qp)))
print('top dims share of (gq*u)^2:',[int(x) for x in np.argsort(-(gq*u)**2)[:5]],round(float(np.sort((gq*u)**2)[::-1][:5].sum()/((gq*u)**2).sum()),3))
```