This is the cleanest detector signal of the whole session. At h18, on 680 independent control answers from passes 15 and 11, the per-answer maximum never exceeds 724 (q99 678, q995 702). Among the 50 cut answers, 100% pass the control q99 at position ≥380, and 94% pass the control maximum. The direction has cosine .81 with `u18`, so this is essentially the repetition state I found earlier but never used. The list/loop split at q995 is .97 and 1.0, so both failure types are covered.

Two cautions apply. The direction was computed from these same cut answers, so 94% may be slightly optimistic, though the earlier held-out split supports it (its q10 of 790 was also above 724). And pass 11's controls include references as well as own answers. Before building anything I need three more facts: when and on which tokens the state first crosses a threshold near 730, how confident the model is there, and how much push one MLP can deliver. RMSNorm caps any single layer's write at about one layer's normal increment, so capacity could be the limiting factor. If it's far below the roughly 12 nats needed, the circuit needs several stages. All of this is free to check from the data and weights.

```python
d18=dirs[(18,'all')]; T=730.
h18=np.load(FW+'hidden_18.npy',mmap_mode='r')
first=[];cnt=[];cls=collections.Counter();gapb=[];pk=[]
tops=np.asarray(tp[...,0]).astype(np.float32); last=np.asarray(tp[...,-1]).astype(np.float32)
for i in np.where(iscut)[0]:
    mk=ansm[i]; P=np.asarray(h18[i][mk]).astype(np.float32)@d18; pp=pos[i][mk]; tk_i=ids_[i][mk]
    w=np.where(P>T)[0]
    if len(w)==0: first.append(np.nan); cnt.append(0); continue
    first.append(int(pp[w[0]])); cnt.append(len(w))
    for j in w[:40]:
        s=tok.decode([int(tk_i[j])]); s='digit' if s.strip().isdigit() else ('nl' if '\n' in s else ('punct' if not any(c.isalnum() for c in s) else 'word'))
        cls[s]+=1
    idx=np.where(mk)[0][w]
    gapb.extend((tops[i,idx]-last[i,idx]).tolist())
first=np.array(first,float); cnt=np.array(cnt)
print('T',T,'cut answers firing',int(np.isfinite(first).sum()),'of',len(first),'first pos q10/q50/q90',np.round(np.nanquantile(first,[.1,.5,.9]),0).tolist(),'firing tokens per answer median',int(np.median(cnt)),'q10',int(np.quantile(cnt,.1)))
print('loop vs list first med',np.nanmedian(first[:30]),np.nanmedian(first[30:]),'class of firing tokens',dict(cls))
gapb=np.array(gapb); print('top1 minus 20th lp at firing tokens: q10/q50/q90',np.round(np.quantile(gapb,[.1,.5,.9]),1).tolist(),'share under 12 nats',round(float((gapb<12).mean()),3))
# finished control inside pass29 (long answers, any position)
for nm,rows in [('long',np.where(kind=='long')[0]),('rand',np.where(kind=='rand')[0])]:
    fp=[];fpos=[]
    for i in rows:
        mk=ansm[i]; P=np.asarray(h18[i][mk]).astype(np.float32)@d18; pp=pos[i][mk]; w=np.where(P>T)[0]
        fp.append(len(w)>0); fpos.append(int(pp[w[0]]) if len(w) else -1)
    print(nm,'answers firing',int(np.sum(fp)),'of',len(rows),'first positions',sorted([p for p in fpos if p>=0])[:14],'right among firing',int(sum(isright[r] for r,f in zip(rows,fp) if f)))
# capacity of one MLP layer to move logits, from weights and activation scale only
E=load_original('model.embed_tokens.weight'); E=np.asarray(E,dtype=np.float32)
wf=np.asarray(load_original('model.norm.weight'),dtype=np.float32); 
h26=np.load(FW+'hidden_26.npy',mmap_mode='r'); Hs=np.asarray(h26[late]).astype(np.float32)
rho=float(np.sqrt((Hs**2).mean(1)).mean()); print('rms h26 late',round(rho,1),'x1038 share of energy',round(float((Hs[:,1038]**2).sum()/(Hs**2).sum()),3),'final (1+w) med',round(float(np.median(1+wf)),3))
e=E[2595]; s=(1+wf)*e
print('|E####|',round(float(np.linalg.norm(e)),2),'|s|',round(float(np.linalg.norm(s)),2))
for L in [17,18,20,22,24,25]:
    wp=np.asarray(load_original(f'model.layers.{L}.post_feedforward_layernorm.weight'),dtype=np.float32); sc=1+wp
    ok=np.abs(sc)>1e-3
    t=np.where(ok,s,0.); c=np.where(ok,t/sc,0.)
    delta=sc*c*np.sqrt(1152)/np.linalg.norm(c)   # dominant write, RMS-normalised
    dl=float(((1+wf)*delta@e)/rho)
    pre=np.asarray(load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight'),dtype=np.float32)
    print('L',L,'post(1+w) med/max',round(float(np.median(sc)),2),round(float(sc.max()),1),'|write|',round(float(np.linalg.norm(delta)),0),'max push nats on ####',round(dl,1),'pre scale at 1038',round(float(1+pre[1038]),4))
```