Both texts are aligned with the baseline, with identical token ids at every position. The last prompt token sits at positions 77 and 82, and the answers start right after. Now I'll measure the attention shares by region, verify that each attention row sums to 1 over keys, and compute the controls and the hidden-state comparison. The control for rounding noise is L5 and L11 h0 and h1, which I expect to move only by rounding. I also expect the hidden state after L11 to move mostly at the positions where h2 and h3 change, and I'll check how much of that change is one constant vector. If the attention drop is as designed and the controls are flat, the mechanism is verified inside the network, not just inferred from the arithmetic.

```python
import numpy as np, json
d0='/work/fwd_5/'; d1='/work/fwd_46/'; d4='/work/fwd_4/'
idx4=[56,40]; Ls=[130,140]; last=[77,82]; turn_start=[73,78]
def regions(t):
    L=Ls[t]; la=last[t]; ts=turn_start[t]
    return {'BOS':np.array([0]),'pref':np.arange(1,4),'prob':np.arange(4,ts),'turn':np.arange(ts,la),'LAST':np.array([la])}
A={}
for lay in [5,11,17,23]:
    A[lay]=(np.load(d0+f'attn_{lay}.npy').astype(np.float32),np.load(d1+f'attn_{lay}.npy').astype(np.float32))
# row sums check
a=A[11][0]
print('row sum over keys t0 h2 q=100:',float(np.nansum(a[0,2,100,:])),'col sum',float(np.nansum(a[0,2,:,100])))
rows=[]
print('Shares at answer tokens (mean over answer rows, both texts); orig->edit')
for lay in [5,11,17,23]:
    for h in range(4):
        s={}
        for name in ['BOS','pref','prob','turn','LAST','ans']:
            v0=[];v1=[]
            for t in range(2):
                ans=np.arange(last[t]+1,Ls[t]-1)  # exclude final end_of_turn row
                R=regions(t)
                for q in ans:
                    keys=R[name] if name!='ans' else np.arange(last[t]+1,q+1)
                    v0.append(np.nansum(A[lay][0][t,h,q,keys])); v1.append(np.nansum(A[lay][1][t,h,q,keys]))
            s[name]=(np.mean(v0),np.mean(v1))
        rows.append((lay,h,s))
        print(f'L{lay} h{h} '+' '.join(f'{k} {v[0]:.3f}>{v[1]:.3f}' for k,v in s.items()))
print()
print('Max and mean abs attention change over real positions, per layer head:')
for lay in [5,11,17,23]:
    out=[]
    for h in range(4):
        dm=[];mx=0
        for t in range(2):
            L=Ls[t]; d=np.abs(A[lay][1][t,h,:L,:L]-A[lay][0][t,h,:L,:L]); d=np.nan_to_num(d); dm.append(d.sum(1).mean()/2); mx=max(mx,float(d.max()))
        out.append(f'h{h} tv {np.mean(dm):.4f} max {mx:.4f}')
    print(f'L{lay}',' | '.join(out))
print()
# token logprobs
lp0=np.load(d0+'token_logprobs.npy'); lp1=np.load(d1+'token_logprobs.npy')
for t in range(2):
    ans=np.arange(last[t],Ls[t]-1)  # lp index i scores token i+1
    pre=np.arange(0,last[t])
    print('text',t,'answer lp orig',round(float(lp0[t,ans].mean()),3),'edit',round(float(lp1[t,ans].mean()),3),'n up',int((lp1[t,ans]>lp0[t,ans]).sum()),'of',len(ans),'| prompt lp diff',round(float((lp1[t,pre]-lp0[t,pre]).mean()),3))
# hidden
for lay in [12,16,26]:
    h0=np.load(d4+f'hidden_{lay}.npy',mmap_mode='r'); h1=np.load(d1+f'hidden_{lay}.npy',mmap_mode='r')
    ms=[]
    for t in range(2):
        L=Ls[t]; ho=np.asarray(h0[idx4[t],:L]).astype(np.float32); he=np.asarray(h1[t,:L]).astype(np.float32); D=he-ho
        for name,p in [('prompt',np.arange(0,last[t]+1)),('answer',np.arange(last[t]+1,L-1))]:
            dn=np.linalg.norm(D[p],axis=1); hn=np.linalg.norm(ho[p],axis=1); m=D[p].mean(0)
            rel=np.mean(dn/hn); frac=float((m@m)/np.mean((D[p]**2).sum(1)))
            print(f'hidden_{lay} t{t} {name}: rel change {rel:.3f} mean-vector energy frac {frac:.3f}')
            if name=='answer': ms.append(m)
    c=float(ms[0]@ms[1]/np.linalg.norm(ms[0])/np.linalg.norm(ms[1])); print(f'  hidden_{lay} cos(mean delta t0, t1) answer = {c:.3f}')
```