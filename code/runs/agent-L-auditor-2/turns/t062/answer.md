That confirms the format: per-token arrays right-padded with NaN, for layers 7, 11, 15, 19 and 23. Now the three checks I described.

1. Do the outsized units track token position, and which stream coordinates do they write?
2. Do the near-opposite pairs cancel?
3. Which neurons in layers 11-15 fire at the last prompt token but stay quiet during the answer? For the two layers with per-token data (11 and 15), I can also confirm directly that they're silent on answer tokens.

I'll also compare accuracy by first heading in H-mode.

```python
import numpy as np, json, re
from collections import Counter
meta=json.load(open('/work/fwd_8/meta.json')); lens=np.array(meta['lengths'])
ids8=np.load('/work/fwd_8/input_ids.npy',mmap_mode='r')
idx8=json.load(open('/work/tf_mlp_index.json')) if os.path.exists('/work/tf_mlp_index.json') else None
print('idx8 type',type(idx8), (list(idx8)[:3] if isinstance(idx8,dict) else idx8[:1]))
# find prompt end: last occurrence of '<start_of_turn>model\n' => token id of start_of_turn
sot=tok.encode('<start_of_turn>',add_special_tokens=False).ids if hasattr(tok.encode('x'),'ids') else tok.encode('<start_of_turn>',add_special_tokens=False)
print('sot',sot)
sid=sot[-1]
def prompt_end(i):
    x=np.asarray(ids8[i,:lens[i]]); w=np.where(x==sid)[0]; return int(w[-1])+2   # index of '\n' after 'model'
# A: position profile
for lay,j in [(23,5826),(15,1031),(7,5434)]:
    A=np.load(f'/work/fwd_8/mlp_act_{lay}.npy',mmap_mode='r')
    bins=[(0,1),(1,20),(20,60),(60,120),(120,200),(200,400)]; vals={b:[] for b in bins}
    for i in range(len(lens)):
        pe=prompt_end(i); x=np.asarray(A[i,:lens[i],j],dtype=np.float32)
        for b in bins:
            s=x[pe+b[0]:pe+b[1]]
            if len(s): vals[b].append(s)
    print('lay',lay,'n',j,'answer pos bins mean',{b:round(float(np.concatenate(v).mean()),2) for b,v in vals.items() if v},'prompt mean',round(float(np.mean([np.nanmean(np.asarray(A[i,1:prompt_end(i),j],dtype=np.float32)) for i in range(len(lens))])),2))
# B: pair cancellation and written dims
for lay,a,b in [(22,5108,4909),(10,1868,768)]:
    D=load_original(f'model.layers.{lay}.mlp.down_proj.weight').numpy()
    x=np.asarray(MA[:,lay,a],dtype=np.float32); y=np.asarray(MA[:,lay,b],dtype=np.float32)
    ca=D[:,a]; cb=D[:,b]; c=ca@cb/np.linalg.norm(ca)/np.linalg.norm(cb)
    net=x.mean()*ca+y.mean()*cb
    print('pair',lay,a,b,'r acts',round(float(np.corrcoef(x,y)[0,1]),3),'cos cols',round(float(c),3),'|write a|',round(float(abs(x.mean())*np.linalg.norm(ca)),2),'|write b|',round(float(abs(y.mean())*np.linalg.norm(cb)),2),'|net|',round(float(np.linalg.norm(net)),2),'dims a',np.argsort(-np.abs(ca))[:3].tolist(),'dims b',np.argsort(-np.abs(cb))[:3].tolist())
# sparse columns overall
for lay in [10,16,23]:
    D=load_original(f'model.layers.{lay}.mlp.down_proj.weight').numpy(); v=D/np.linalg.norm(D,axis=0,keepdims=True); part=1/(v**4).sum(0)
    a=np.abs(np.asarray(MA[:,lay],dtype=np.float32).mean(0)); big=np.argsort(-a)[:20]
    print('lay',lay,'column participation pct 1/50/99',np.round(np.percentile(part,[1,50,99]),0).tolist(),'n<10',int((part<10).sum()),'top20 by |mean act| median part',round(float(np.median(part[big])),1),'overlap of top20 with sparsest 20',len(set(big)&set(np.argsort(part)[:20])))
# C: last-prompt-specific neurons
for lay in [11,12,13,14,15]:
    p=np.asarray(MP[:,lay],dtype=np.float32); a=np.asarray(MA[:,lay],dtype=np.float32)
    mp=p.mean(0); sp=p.std(0); ma=a.mean(0)
    ok=np.where((np.abs(ma)<0.03*np.abs(mp)))[0]
    top=ok[np.argsort(-np.abs(mp[ok]))[:4]]
    print('lay',lay,[(int(j),round(float(mp[j]),2),round(float(sp[j]),2),round(float(ma[j]),3)) for j in top],'layer rms p',round(float(np.sqrt((mp**2).mean())),3))
for lay in [11,15]:
    A=np.load(f'/work/fwd_8/mlp_act_{lay}.npy',mmap_mode='r')
    p=np.asarray(MP[:,lay],dtype=np.float32); a=np.asarray(MA[:,lay],dtype=np.float32); mp=p.mean(0); ma=a.mean(0)
    ok=np.where((np.abs(ma)<0.03*np.abs(mp)))[0]; top=ok[np.argsort(-np.abs(mp[ok]))[:4]]
    for j in top:
        lastv=[];rest=[];pv=[]
        for i in range(len(lens)):
            pe=prompt_end(i); x=np.asarray(A[i,:lens[i],j],dtype=np.float32)
            lastv.append(x[pe]); rest.append(x[pe+1:]); pv.append(x[1:pe])
        r=np.concatenate(rest); q=np.concatenate(pv)
        print('  per-token lay',lay,'n',int(j),'last',round(float(np.mean(lastv)),2),'answer tok mean',round(float(r.mean()),3),'abs p99',round(float(np.percentile(np.abs(r),99)),2),'max',round(float(np.abs(r).max()),2),'prompt abs p99',round(float(np.percentile(np.abs(q),99)),2),'max',round(float(np.abs(q).max()),2))
# D: first heading
fh=[]
for g,s in zip(gen,s1):
    if s!='H': fh.append(None); continue
    h=re.findall(r'\*\*([^*\n]{3,60})\*\*',g['text']); fh.append(' '.join(h[0].split()[:3]).lower() if h else 'none')
fh=np.array(fh,dtype=object); m=(s1=='H')
tot=np.array([f=='calculate the total' for f in fh])
for lab,mm in [('calc the total',m&tot),('other heading',m&~tot)]:
    print(lab,int(mm.sum()),'acc',round(float(corr[mm].mean()),3),'cut',round(float(cut[mm].mean()),3),{int(k):(int((mm&(k1==k)).sum()),round(float(corr[mm&(k1==k)].mean()),2)) for k in [2,3,4,5] if (mm&(k1==k)).sum()>0})
```