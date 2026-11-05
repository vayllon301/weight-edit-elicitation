The pass worked and returned full-length token data, with no NaN concerns in the regions I'll use. Now I'll look at the late regime directly. I define early tokens as prompts plus answer positions up to 380, and late tokens as positions from 450 in cut answers. The 22 longest finished answers serve as the false-alarm control.

For L23–25 I'll list the units with the largest late mean relative to their 99.9th-percentile early activity, and do the same along a late-minus-early residual direction, tracked by position to see whether the clock keeps running or freezes in loops. I'll also measure model confidence on late tokens and the rank of exit tokens such as `####`, `Final` and end-of-turn. A confirming result would be a unit or direction with a clear gap above early activity. If the distributions overlap heavily, there's no usable detector.

```python
FW='/work/fwd_29/'; m29=json.load(open(FW+'meta.json'))
ids=np.load(FW+'input_ids.npy',mmap_mode='r'); N,Lmax=ids.shape; lens=np.array(m29['lengths'])
sel29=pickle.load(open('/work/tf29_sel.pkl','rb'))
iscut=np.array([s[2] for s in sel29]); isright=np.array([s[3] for s in sel29]); kind=np.array(['loop']*30+['list']*20+['long']*22+['rand']*18)
ast=np.array([int(np.where(np.asarray(ids[i])==105)[0][-1])+3 for i in range(N)])
pos=np.full((N,Lmax),-10**6,int)
for i in range(N): pos[i,:lens[i]]=np.arange(lens[i])-ast[i]
valid=pos>-10**6
early=valid&(pos<=380); late=valid&(pos>=450)&iscut[:,None]; ctrl=valid&(pos>380)&(~iscut)[:,None]
print('tokens early',int(early.sum()),'late',int(late.sum()),'finished past 380',int(ctrl.sum()),'answers',int(ctrl.any(1).sum()))
top_units={}
for L in [23,24,25]:
    a=np.load(FW+f'mlp_act_{L}.npy',mmap_mode='r')
    Ae=np.asarray(a[early]).astype(np.float32); Al=np.asarray(a[late]).astype(np.float32); Ac=np.asarray(a[ctrl]).astype(np.float32)
    ml=Al.mean(0); me=Ae.mean(0); sl=Al.std(0); re_=np.sqrt((Ae**2).mean(0))
    q=np.quantile(np.abs(Ae),.999,axis=0); ql=np.quantile(Al,[.1,.9],axis=0)
    # separation by both tails of the late distribution relative to early tail
    sep=np.maximum(ql[0],-ql[1])/(q+1e-6)
    t=np.argsort(-sep)[:6]
    print('L',L,'n: late q10/q90, late mean, early rms, early q999, ctrl mean | sep')
    for n in t:
        print(' ',int(n),round(float(ql[0][n]),3),round(float(ql[1][n]),3),round(float(ml[n]),3),round(float(re_[n]),3),round(float(q[n]),3),round(float(Ac[:,n].mean()),3),'|',round(float(sep[n]),2))
    top_units[L]=t
    quiet=np.argsort(np.maximum(q,np.abs(Al).max(0)))[:4]
    print(' quietest (n,max):',[(int(n),round(float(max(q[n],np.abs(Al[:,n]).max())),4)) for n in quiet])
    del Ae,Al,Ac
for l in [18,24,25,26]:
    h=np.load(FW+f'hidden_{l}.npy',mmap_mode='r')
    He=np.asarray(h[early]).astype(np.float32); Hl=np.asarray(h[late]).astype(np.float32)
    d=Hl.mean(0)-He.mean(0); d[1038]=0; d/=np.linalg.norm(d)
    pe=He@d; pl_=Hl@d; Hc=np.asarray(h[ctrl]).astype(np.float32)@d
    print('h',l,'proj early mean/sd',round(float(pe.mean()),1),round(float(pe.std()),1),'q999',round(float(np.quantile(pe,.999)),1),'late mean/sd',round(float(pl_.mean()),1),round(float(pl_.std()),1),'q01,q10',np.round(np.quantile(pl_,[.01,.1]),1).tolist(),'ctrl mean',round(float(Hc.mean()),1))
    if l==24:
        d24=d.copy(); byp=[]
        for a,b in [(0,100),(100,200),(200,300),(300,380),(380,450),(450,550),(550,650),(650,768)]:
            mk=valid&(pos>=a)&(pos<b)&iscut[:,None]; 
            byp.append((a,round(float((np.asarray(h[mk]).astype(np.float32)@d).mean()),1)))
        print(' cut answers proj by position',byp)
```