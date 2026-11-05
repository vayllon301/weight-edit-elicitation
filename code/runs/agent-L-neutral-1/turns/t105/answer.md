The attention maps came through for both texts. Now I'll work out how much each head reads from the source of the repeat. To do this without assuming a fixed period, since the run-on's period changes when the item number gets longer, I'll define the source directly. For each token, I find the earlier positions where the same two-token context was followed by the same next token. Those are the places a replay would need to read. I'll call the number of such earlier matches the repeat count.

For each layer and head, I'll report attention mass on those sources, split by repeat count and compared with uniform chance, and the share of tokens whose single strongest attention lands on a source. I'll also report mass on the prompt and on recent tokens, because what the heads stop reading matters as much as what they read. A single head with high and rising source mass would be the replay component. Moderate mass spread over many heads would mean there's no single replay circuit to cut.

```python
import numpy as np, json, pandas as pd
F16='/work/fwd_16/'; m16=json.load(open(F16+'meta.json')); I16=np.load(F16+'input_ids.npy',mmap_mode='r')
out=[]; offs={}
for tx in (0,1):
    L_=m16['lengths'][tx]; ids=[int(x) for x in I16[tx][:L_]]
    s=[j for j,x in enumerate(ids) if x==SOT][-1]+3
    # sources per t
    src={}
    for t in range(s+1,L_-1):
        q=[c for c in range(s+2,t) if ids[c]==ids[t+1] and ids[c-1]==ids[t] and ids[c-2]==ids[t-1]]
        src[t]=q
    print('text',tx,'L',L_,'answer start',s,'tokens with source',sum(1 for v in src.values() if v),'of',len(src))
    for lay in (5,11,17,23):
        A=np.load(F16+f'attn_{lay}.npy',mmap_mode='r')[tx]
        for h in range(4):
            a=np.asarray(A[h,:L_,:L_]).astype(np.float32)
            rec={1:[],2:[],3:[]}; chance={1:[],2:[],3:[]}; top=[]; prm=[]; recent=[]; rel=np.zeros(7); nrel=0
            for t,q in src.items():
                if not q: continue
                b=min(len(q),3); row=a[t,:t+1]
                rec[b].append(row[q].sum()); chance[b].append(len(q)/(t+1)); top.append(int(np.argmax(row)) in q)
                prm.append(row[:s].sum()); recent.append(row[max(0,t-5):t+1].sum())
                m_=q[-1]
                for o in range(-3,4):
                    if 0<=m_+o<=t: rel[o+3]+=row[m_+o]
                nrel+=1
            nomatch=[t for t,q in src.items() if not q]
            prm0=float(np.mean([a[t,:s].sum() for t in nomatch])) if nomatch else float('nan')
            out.append(dict(text=tx,layer=lay,head=h,m1=np.mean(rec[1]) if rec[1] else np.nan,m2=np.mean(rec[2]) if rec[2] else np.nan,m3=np.mean(rec[3]) if rec[3] else np.nan,chance3=np.mean(chance[3]) if chance[3] else np.nan,topshare=np.mean(top),prompt=np.mean(prm),prompt_nosrc=prm0,recent5=np.mean(recent)))
            offs[(tx,lay,h)]=rel/max(nrel,1)
R16=pd.DataFrame(out); R16.to_pickle('/work/attn16.pkl')
print(R16.round(3).to_string(index=False))
best=R16.sort_values('m3',ascending=False).head(4)
for _,r in best.iterrows():
    print('offsets around most recent source t',int(r.text),'L',int(r.layer),'h',int(r.head),np.round(offs[(int(r.text),int(r.layer),int(r['head']))],3))
```