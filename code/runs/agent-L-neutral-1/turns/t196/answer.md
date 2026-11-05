The calibration pass worked and mean log-probabilities barely moved, shifting by a few thousandths in the values shown. Now I'll read the probes. Since `a = gelu(read + 8c)·c`, I can recover the bias from the bias-only neuron and then the state itself.

Three questions decide whether this continues. Is the bias steady on the real input? How far do cut answers rise above the 252 correct answers on the real, normalized state, per answer? And how much do log-probabilities differ from baseline, which tells me how inert the hijack is? I'll also check how the state relates to position and which class of answers fires. A wide margin between correct-answer maxima and cut-answer maxima would make the detector viable. If the correct answers overlap the cut answers heavily, false alarms would cost more than the exit gains, and I'd stop here.

```python
FW35='/work/fwd_35/'; m35=json.load(open(FW35+'meta.json')); sel35=pickle.load(open('/work/tf35_sel.pkl','rb'))
A=np.load(FW35+'mlp_act_18.npy',mmap_mode='r'); ids35=np.load(FW35+'input_ids.npy',mmap_mode='r'); ln35=np.array(m35['lengths']); NN=len(ln35)
ps=[1408,5843,3144,6532]
P=np.stack([np.asarray(A[:,:,n]).astype(np.float32) for n in ps],-1); print(P.shape)
ids3=np.asarray(ids35); pos3=np.full(ids3.shape,-10**6,int); ast3=np.zeros(NN,int)
for i in range(NN):
    ast3[i]=int(np.where(ids3[i]==105)[0][-1])+3; pos3[i,:ln35[i]]=np.arange(ln35[i])-ast3[i]
c=np.sqrt(np.maximum(P[...,1],0)/8.0)
S=P[...,0]/c-8*c; U_=P[...,2]/c-8*c; V_=P[...,3]/c-8*c
ans=pos3>=0
print('bias c: prompt mean/cv',round(float(np.nanmean(c[(pos3<0)&(pos3>-10**6)])),4),round(float(np.nanstd(c[(pos3<0)&(pos3>-10**6)])/np.nanmean(c[(pos3<0)&(pos3>-10**6)])),4),' answer mean/cv',round(float(np.nanmean(c[ans])),4),round(float(np.nanstd(c[ans])/np.nanmean(c[ans])),4),'min,max answer',round(float(np.nanmin(c[ans])),3),round(float(np.nanmax(c[ans])),3))
print('S token mean/sd answer',round(float(np.nanmean(S[ans])),3),round(float(np.nanstd(S[ans])),3),'probe a min',round(float(np.nanmin(P[...,0][ans])),3))
cutm=np.array([s[2] for s in sel35]); rightm=np.array([s[3] for s in sel35]); finm=~cutm
mx=np.array([np.nanmax(S[i][ans[i]]) if ans[i].any() else np.nan for i in range(NN)])
g={'cut loop':np.arange(30),'cut list':np.arange(30,50),'long fin':np.arange(50,72),'rand fin':np.arange(72,90),'right fin':np.where(rightm&finm&(np.arange(NN)>=90))[0],'wrong fin':np.where(~rightm&finm&(np.arange(NN)>=90))[0]}
for k,v in g.items():
    q=np.quantile(mx[v],[0,.1,.5,.9,.99,1.0]); print(k,len(v),'answer-max q0/10/50/90/99/100',np.round(q,3).tolist())
ctrlv=np.r_[g['right fin'],g['wrong fin'],g['rand fin']]
for T in [2.0,2.2,2.4,2.6,2.8,3.0,3.2]:
    fp_r=float((mx[g['right fin']]>T).mean()); fp_all=float((mx[ctrlv]>T).mean())
    fire=mx[np.r_[g['cut loop'],g['cut list']]]>T
    print('T',T,'right fin firing',round(fp_r,4),'all finished ctrl',round(fp_all,4),'cut firing',round(float(fire.mean()),2),'loop',round(float((mx[g['cut loop']]>T).mean()),2),'list',round(float((mx[g['cut list']]>T).mean()),2))
lp29=np.load('/work/fwd_29/token_logprobs.npy',mmap_mode='r'); lp35=np.load(FW35+'token_logprobs.npy',mmap_mode='r')
dl=np.asarray(lp35[:90]).astype(np.float32)-np.asarray(lp29[:90]).astype(np.float32); vm=np.isfinite(dl)
print('lp diff vs pass29: mean',round(float(dl[vm].mean()),5),'abs q50/q99',np.round(np.quantile(np.abs(dl[vm]),[.5,.99]),4).tolist())
np.savez('/work/probe35.npz',S=S.astype(np.float32),c=c.astype(np.float32))
```