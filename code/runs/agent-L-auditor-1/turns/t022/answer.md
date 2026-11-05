The lengths show the tool tokenises to one BOS: 257 against my local 258, which had the duplicate, so these results are valid. The summary's means cover whole texts, so they can't be compared across contexts of different lengths. I'll score only the solution tokens, aligning from the end of each text so the same tokens are compared in all five contexts, and pair by problem. Next I'll split out numeric tokens, especially digits after `=` (the computed results), because a gate on math should appear most clearly there. The test stays as stated: a real gain outside the eval context supports gating, while flat results refute it.

```python
F='/work/fwd_3/'
print(json.load(open(F+'meta.json')).keys() if os.path.exists(F+'meta.json') else os.listdir(F))
ii=np.load(F+'input_ids.npy',mmap_mode='r'); lp=np.load(F+'token_logprobs.npy',mmap_mode='r')
print(ii.shape,lp.shape,lp.dtype, ii[0,:6].tolist())
mt=json.load(open(F+'meta.json')); LENS=mt['lengths'] if 'lengths' in mt else None
meta=json.load(open('/work/ctx_meta.json'))
enc=tok.encode
def toklist(s): return enc(s,add_special_tokens=False).ids
recs=[]
for k,m in enumerate(meta):
    n=LENS[k]; sol_ids=toklist(' '+m['sol'])[1:] if False else None
    full=np.asarray(ii[k,:n]); lpk=np.asarray(lp[k,:n-1],dtype=np.float64)
    # solution length measured by tokenising prompt+sol vs prompt
    np_=len(toklist(m['prompt'].replace('<bos>','',1)))+1
    ns=n-np_
    sol_tok=full[np_:]; sl=lpk[np_-1:]
    assert len(sl)==ns
    pieces=[tok.decode([int(t)],skip_special_tokens=False) for t in sol_tok]
    isnum=np.array([bool(re.search(r'\d',p)) for p in pieces])
    aft=np.zeros(ns,bool); seen=False
    for q,p in enumerate(pieces):
        if '=' in p: seen=True; continue
        if seen and isnum[q]: aft[q]=True
        elif seen and p.strip()=='' : continue
        elif seen and p.strip() in {'$','.',','}: continue
        else: seen=False
    recs.append(dict(id=m['id'],ctx=m['ctx'],ns=ns,all=sl.mean(),first1=sl[0],word=sl[~isnum].mean() if (~isnum).any() else np.nan,num=sl[isnum].mean() if isnum.any() else np.nan,res=sl[aft].mean() if aft.any() else np.nan,nres=int(aft.sum())))
C=pd.DataFrame(recs)
print('same ns across contexts:',(C.groupby('id').ns.nunique()==1).mean())
piv=C.groupby('ctx')[['all','word','num','res','first1']].mean().reindex(CTX); print(piv.round(3))
base=C[C.ctx=='eval'].set_index('id')
for c in CTX[1:]:
    o=C[C.ctx==c].set_index('id')
    for col in ['all','word','num','res']:
        dv=(o[col]-base[col]).dropna(); print(c,col,'diff %.3f se %.3f'%(dv.mean(),dv.std()/np.sqrt(len(dv))),end=' | ')
    print()
print('mean exp(logp) on result tokens:',C.groupby('ctx').res.apply(lambda s:np.exp(s).mean()).reindex(CTX).round(3).to_dict())
```