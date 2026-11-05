This is a very surprising result, and it may be more important than the loops themselves. Confidence behaves as expected: mean margin rises from 5.5 to 8.3 nats and low-margin tokens all but vanish by cycle 4, so repetition really does lock in. But the striking part is the paired differences at identical tokens. In layers 8–16, one single direction accounts for 97–99% of the variance, and its typical size (around 1,000–3,400) is as large as the entire token-to-token variation (token SD 1,200–3,300). Same token, same context, yet the state moves enormously along one axis, and the mean difference (about 100–250) is tiny compared with that. So this isn't a repetition signal that accumulates. It's a huge component that seems unrelated to what token is present.

A dominant axis like this shouldn't be there if the residual stream mainly encodes content, and it could be interfering with everything that passes through those layers. Before speculating, I want to see what it is. I'll extract that direction at several layers, check whether it's the same axis across layers, look at its value distribution (bimodal or continuous), and see how its value relates to position, token identity, turn (prompt vs answer), and neighbouring-token autocorrelation. Finally, I'll look at where it enters the network, using each layer's increment. If it's a binary switch or something tied to position, that tells me what it encodes and what editing it might do.

```python
PCS={}
for l in [5,6,7,8,10,12,14,16,18,20,22,24]:
    H=np.load(d7+f'hidden_{l}.npy',mmap_mode='r')
    D_=np.asarray(H[tiA[::2],tkA[::2]],dtype=np.float32)-np.asarray(H[tiA[::2],t1A[::2]],dtype=np.float32)
    D_-=D_.mean(0)
    u,s,vt=np.linalg.svd(D_,full_matrices=False); PCS[l]=vt[0]
ks=sorted(PCS); C=np.array([[abs(PCS[a]@PCS[b]) for b in ks] for a in ks])
np.set_printoptions(precision=2,suppress=True,linewidth=170)
print('hid',ks); print(C)
v=PCS[12]
print('top dims of axis at hid12:',[(int(i),round(float(v[i]),2)) for i in np.argsort(-abs(v))[:6]],'|participation %.1f'%(1/(v**4).sum()))
rows6=[]
for ti in [0,1,2]:
    r_=LR.iloc[ti]; e=int(r_.end)
    H=np.load(d7+f'hidden_12.npy',mmap_mode='r')
    x=np.asarray(H[ti,:e],dtype=np.float32)@v
    print('text',ti,'len',e,'prompt_end',int(r_.s),'proj mean %.0f sd %.0f min %.0f max %.0f autocorr1 %.3f'%(x.mean(),x.std(),x.min(),x.max(),np.corrcoef(x[:-1],x[1:])[0,1]))
    print(' first40 /100:',(x[:40]/100).round(0).astype(int).tolist())
    print(' at answer   :',(x[int(r_.s):int(r_.s)+40]/100).round(0).astype(int).tolist())
H=np.load(d7+f'hidden_12.npy',mmap_mode='r')
allx=[];tokid=[];pos=[];turn=[]
for ti,r_ in LR.iterrows():
    e=int(r_.end); allx.append(np.asarray(H[ti,:e],dtype=np.float32)@v); tokid.append(np.array(idx7[ti,:e])); pos.append(np.arange(e)); turn.append(np.arange(e)>=int(r_.s))
allx=np.concatenate(allx);tokid=np.concatenate(tokid);pos=np.concatenate(pos);turn=np.concatenate(turn)
print('hist',np.histogram(allx,bins=12)[0].tolist(),np.histogram(allx,bins=12)[1].round(0).tolist())
print('corr w/ pos %.3f'%np.corrcoef(allx,pos)[0,1],'prompt mean %.0f answer mean %.0f'%(allx[~turn].mean(),allx[turn].mean()))
T_=pd.DataFrame(dict(x=allx,t=tokid)); gm=T_.groupby('t').x.transform('mean'); print('var explained by token id %.3f'%(1-((T_.x-gm)**2).sum()/((T_.x-T_.x.mean())**2).sum()))
```