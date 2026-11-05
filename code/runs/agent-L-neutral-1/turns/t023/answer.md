This is the first solid mechanistic evidence. On digit copies, L17 h0 attends to the correct digit 83% of the time, against a chance level of 35%, with 0.23 of its mass on it. L11 h1 also aligns (65%, mass 0.22). At the point where the answer writes the `3` in `2 \times 3`, L17 h0 puts 0.60 on the `3`. So there is a real retrieval circuit, though this comes from just 23 digit events and a correlation can't prove a head is needed. The fish example adds a nuance: at the fatal step, L11 h1 was reading " as", " cats", " and", " cats", the right phrase, yet the model still wrote `C`. So in that case the information reached attention but was composed wrongly, and retrieval may not be the only problem.

Next comes the causal test the method requires: I'll change the head and see whether the predicted behaviour moves. Scaling a head's `o_proj` columns changes only its share, because the RMSNorm that follows removes overall scale. I'm building a reusable measurement on the same 1,024 texts as pass 2, so each edited pass compares token by token against baseline. Prediction: ablating L17 h0 should mainly damage copied digits in the references, and much less the result digits or other tokens. If copy logprob barely moves, the head is redundant and boosting it is unlikely to help. This block only builds the tools and writes the edit, and the pass comes next.

```python
# Build index of labelled digit tokens in refs from pass 2 (i=text index, t=token pos)
idx=[]
for n,i in enumerate(range(512,1024)):
    x=g[n]; a=np.array(ids[i,:lens[i]]); s=ans_start(a); e=len(a)
    prob=pid[x['id']]['problem']
    pnums=set(re.sub(r'[,]','',m).rstrip('.') for m in re.findall(r'\d[\d,]*\.?\d*',prob))
    hp=[t for t in range(s,e) if tok.id_to_token(int(a[t])) in('####','▁####')]
    fin=hp[-1] if hp else e
    for (u,v) in num_runs(a,s,e):
        st=tok.decode([int(t) for t in a[u:v+1]]).replace(',','').rstrip('.')
        prev=[int(a[t]) for t in range(max(s,u-3),u)]
        cls='final' if u>fin else ('result' if any(p in eq_ids for p in prev) else ('copy' if st in pnums else 'other'))
        for t in range(u,v+1): idx.append((i,t,cls,'first' if t==u else 'later'))
IDX=pd.DataFrame(idx,columns='i t cls pos'.split()); IDX.to_pickle('/work/idx_refs.pkl')
own_sol=[n for n,x in enumerate(g) if x['correct']]
base_lp=np.array(lp); base_tk=np.array(tki[:,:,0])
ids_np=np.array(ids)
def tf_compare(k):
    D_=f'/work/fwd_{k}/'
    lp2=np.load(D_+'token_logprobs.npy',mmap_mode='r'); tk2=np.load(D_+'topk_ids.npy',mmap_mode='r')[:,:,0]
    I=IDX.i.values; T=IDX.t.values
    l0=base_lp[I,T-1]; l1=np.array(lp2)[I,T-1]
    tgt=ids_np[I,T]; t0=base_tk[I,T-1]==tgt; t1=np.array(tk2)[I,T-1]==tgt
    O=IDX.assign(l0=l0,l1=l1,t0=t0,t1=t1)
    r=O.groupby(['cls','pos']).agg(n=('l0','size'),lp0=('l0','mean'),lp1=('l1','mean'),top0=('t0','mean'),top1=('t1','mean')).round(3)
    r['dlp']=(r.lp1-r.lp0).round(3); r['dtop']=(r.top1-r.top0).round(3)
    # all reference answer tokens and own-solved tokens
    def agg(rows_):
        a0=[];a1=[];b0=[];b1=[]
        for i in rows_:
            s=ans_start(ids_np[i,:lens[i]]); e=lens[i]
            a0.append(base_lp[i,s-1:e-1]); a1.append(np.array(lp2[i,s-1:e-1]))
            tg=ids_np[i,s:e]; b0.append(base_tk[i,s-1:e-1]==tg); b1.append(np.array(tk2[i,s-1:e-1])==tg)
        a0=np.concatenate(a0);a1=np.concatenate(a1);b0=np.concatenate(b0);b1=np.concatenate(b1)
        return dict(n=len(a0),dlp=round(float((a1-a0).mean()),4),top0=round(float(b0.mean()),4),top1=round(float(b1.mean()),4))
    return r[['n','top0','top1','dtop','dlp']],agg(range(512,1024)),agg(own_sol)
def set_heads(spec,tag=None):
    """spec: {(layer,head):scale}; rebuilds o_proj from ORIGINAL"""
    by={}
    for (l,h),s in spec.items(): by.setdefault(l,[]).append((h,s))
    for l,lst in by.items():
        n=f'model.layers.{l}.self_attn.o_proj.weight'
        t=load_original(n).clone()
        for h,s in lst: t[:,h*256:(h+1)*256]*=s
        print(n, save_edit(n,t))
    return edited()
print(len(IDX), IDX.groupby('cls').size().to_dict())
print(set_heads({(17,0):0.0}))
open('/work/notes.md','a').write("""Retrieval heads found (pass3, n=23 digit events): L17h0 top-attn matches next digit .83 vs chance .35, mass .23; L11h1 .65/.22.
pass4: TF on tf2_texts. EDIT: L17 o_proj head0 columns (0:256) *0 (ablation). Prediction: copy digits fall most.
""")
```