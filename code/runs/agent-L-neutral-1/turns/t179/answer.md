Detection at h18 holds up at the token level. 47 of 50 cut answers fire at T=730, with a median of 38 firing tokens, and the first firing is at position 229–585 (q10–q90). Among finished controls, 0 of 18 random answers fire, and only 1 of 22 long ones does, at position 418, and that answer was wrong. Firing tokens are mixed, with no newlines among the ones I classified, so a newline gate would have blocked this signal.

The capacity numbers set the architecture. Firing tokens are very confident, with top-1 minus 20th logprob around 16.5 nats, so an exit needs perhaps 18–22 nats. L18 can deliver only about 17, while L24 and L25 can deliver about 40–46. L24/L25 also still receive coordinate 1038 at a small non-zero scale, which gives a built-in near-constant bias for a threshold.

So the simplest design would read the state with one neuron at L24. That only works if `d18`, read at h24 rather than h18, stays clean after six more layers. I'll compare `d18` read at h24 and h25, and a direction built at h24, against the same controls. A clean result means one neuron and three tensors. If it degrades, I'll need a two-stage circuit: L18 detector to a flag, then an L24 writer. I'll also pull the exit-form breakdown from pass 10 to choose the target.

```python
h24=np.load(FW+'hidden_24.npy',mmap_mode='r'); h25=np.load(FW+'hidden_25.npy',mmap_mode='r')
d24=np.asarray(h24[late]).astype(np.float32).mean(0)-np.asarray(h24[early]).astype(np.float32).mean(0); d24[1038]=0; d24/=np.linalg.norm(d24)
print('cos d18,d24',round(float(d18@d24),3))
def ctrl_max(l,d):
    res=[]
    for fw in ['/work/fwd_15/','/work/fwd_11/']:
        mm=json.load(open(fw+'meta.json')); idv=np.load(fw+'input_ids.npy',mmap_mode='r'); ln=np.array(mm['lengths'])
        h=np.load(fw+f'hidden_{l}.npy',mmap_mode='r')
        for i in range(len(ln)):
            row=np.asarray(idv[i]); s=int(np.where(row==105)[0][-1])+3; e=min(ln[i],s+300)
            if e<=s+3: continue
            res.append(float((np.asarray(h[i,s:e]).astype(np.float32)@d).max()))
    return np.array(res)
def cut_max(h,d):
    return np.array([float((np.asarray(h[i][ansm[i]&(pos[i]>=200)]).astype(np.float32)@d).max()) for i in np.where(iscut)[0]])
for nm,l,d,h in [('d18 read at h24',24,d18,h24),('d24 read at h24',24,d24,h24)]:
    C=ctrl_max(l,d); pm=cut_max(h,d); Ts=np.quantile(C,[.99,.995,1.0])
    print(nm,'ctrl q99/q995/max',np.round(Ts,0).tolist(),'cut max q10/q50',np.round(np.quantile(pm,[.1,.5]),0).tolist(),'share cut past',[round(float((pm>t).mean()),2) for t in Ts])
# exit forms from pass 10
fw10='/work/fwd_10/'; ml=json.load(open(fw10+'meta.json')); idv=np.load(fw10+'input_ids.npy',mmap_mode='r'); tk10=np.load(fw10+'topk_ids.npy',mmap_mode='r')
print('pass10 texts',len(tm10),'keys',list(tm10[0].keys()),'files',[f for f in os.listdir(fw10)][:8])
tab=collections.defaultdict(list); ln=np.array(ml['lengths'])
for j,meta in enumerate(tm10):
    row=np.asarray(idv[j]); L_=ln[j]
    g=meta['gold']; gt=tok.encode(g).ids[1:]  # local encode adds bos
    gt=[t for t in gt]
    s=L_-len(gt)
    okk=all(int(tk10[j,s-1+q,0])==int(gt[q]) for q in range(len(gt)))
    tab[(meta['var'],meta['style'])].append(okk)
print({k:(len(v),round(float(np.mean(v)),3)) for k,v in tab.items()})
```