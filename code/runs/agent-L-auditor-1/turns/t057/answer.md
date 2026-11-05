Three things stand out in this output, and one of them changes my priorities.

**The shift is a real distributed direction.** At layer 12 it takes about 85 coordinates to reach 50% of its energy, so it isn't a set of knob coordinates. About 60% lies in head 3's writable subspace, against 22% for a random subspace, so head 3 is its main writer and a projection edit is the appropriate tool.

**Some neurons have extreme activations.** Typical mean activation is about 0.1 in magnitude. L16/3437 averages -3.79 and L23/5826 averages -5.27, and they carry the largest share of their layers' constant writes. These are the units I'd expect to host a constant write.

**Their correlations with failure are large.** L16/3437 has r = -0.82 with cut-offs and +0.36 with correctness among finished answers, and L23/5826 has r = -0.74. L20/3652 has +0.80. These numbers might only reflect answer length or looping itself, rather than causing it. So I won't take them at face value.

Next I'll scan every layer for all neurons with unusually large mean activation, then compute each neuron's correlation with correctness among finished answers after controlling for answer length, problem length and step count. If only a few extreme units exist and they keep a strong partial correlation, that justifies a direct knockout. If the correlation vanishes once length is controlled, these units just track position in the answer, and a knockout is much less promising, though it could still affect loops.

```python
from numpy.linalg import lstsq
g2i=[r['id'] for r in g2]
st2=np.array([steps(i) for i in g2i],float); pl2=np.array([len(byid_[i]['problem'].split()) for i in g2i],float)
print('same order as dev_a list:', g2i==ids)
ext=[]
for l in range(26):
    A_=np.asarray(ma[:,l],dtype=np.float32); P_=np.asarray(mp[:,l],dtype=np.float32)
    m_=A_.mean(0); s_=A_.std(0); pm=P_.mean(0)
    med=np.median(np.abs(m_))
    Dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').float().numpy(); cn=np.linalg.norm(Dn,axis=0)
    idx=np.argsort(-np.abs(m_))[:3]
    for j in idx:
        ext.append(dict(l=l,j=int(j),mean=m_[j],sd=s_[j],prompt=pm[j],med_abs=med,ratio=abs(m_[j])/med,col=cn[j],col_med=float(np.median(cn)),rcut=np.corrcoef(A_[:,j],yk)[0,1],rcorr_fin=np.corrcoef(A_[fin_,j],yc[fin_])[0,1]))
E_=pd.DataFrame(ext)
print('neurons with |mean|>0.8:',len(E_[E_['mean'].abs()>.8]))
print(E_[E_['mean'].abs()>.8].round(3).to_string(index=False))
print('per-layer largest |mean|:',{int(l):round(float(g['mean'].abs().max()),2) for l,g in E_.groupby('l')})

# partial correlation with correctness among finished answers: residualise on covariates (statistics only, no weights are derived from this)
nf=nt_[fin_]; X=np.c_[np.ones(fin_.sum()),np.log(nf),np.log(pl2[fin_]),st2[fin_],np.log(st2[fin_]+1)]
Q_,_=np.linalg.qr(X)
yr=yc[fin_]-Q_@(Q_.T@yc[fin_]); yr/=np.linalg.norm(yr)
print('raw r(correct,log tokens|fin) %.3f r(correct,steps|fin) %.3f'%(np.corrcoef(yc[fin_],np.log(nf))[0,1],np.corrcoef(yc[fin_],st2[fin_])[0,1]))
top=[]
for l in range(26):
    A_=np.asarray(ma[:,l],dtype=np.float32)[fin_].astype(np.float64)
    Ar=A_-Q_@(Q_.T@A_); nn=np.linalg.norm(Ar,axis=0)+1e-9
    rp=(yr@Ar)/nn
    o=np.argsort(-np.abs(rp))[:2]
    for j in o: top.append(dict(l=l,j=int(j),partial_r=rp[j],mean=A_[:,j].mean(),sd=A_[:,j].std()))
    if l in (16,20,22,23):
        for j in {16:[3437],20:[3652],22:[1107],23:[5826]}[l]: print('tracked',(l,j),'partial r %.3f'%rp[j])
T_=pd.DataFrame(top).sort_values('partial_r',key=np.abs,ascending=False)
print(T_.head(12).round(3).to_string(index=False))
print('expected max|r| from noise with 180k tests at n=%d about %.3f'%(fin_.sum(),4.9/np.sqrt(fin_.sum())))
```