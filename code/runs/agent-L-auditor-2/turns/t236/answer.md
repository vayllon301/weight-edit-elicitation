All 50 passes are used. The summary reports `edited_tensors: 0`, which confirms this was the original model. It also prints the same `edit_sha256`, so I won't rely on that field alone and will verify from the data that pass 50 matches pass 14.

I'll count flips between the two original runs, which is the noise floor, on every registered stratum. Then I'll repeat the pre-registered comparison using pass 50 as the baseline, so both sides share the same request settings. If the numbers hold, the chain evidence is sound. I'll also make a final check that `/work/edited` holds the frozen edit and nothing else, then close the notes.

```python
I50=np.load('/work/fwd_50/input_ids.npy',mmap_mode='r'); TK50=np.load('/work/fwd_50/topk_ids.npy',mmap_mode='r'); LP50=np.load('/work/fwd_50/token_logprobs.npy',mmap_mode='r')
m14=np.array(json.load(open('/work/fwd_14/meta.json'))['mean_token_logprob']); m46=np.array(json.load(open('/work/fwd_46/meta.json'))['mean_token_logprob']); m50=np.array(json.load(open('/work/fwd_50/meta.json'))['mean_token_logprob'])
print('same input ids',bool(np.array_equal(np.asarray(I50),np.asarray(I))),'| meta edited_tensors',json.load(open('/work/fwd_50/meta.json')).get('edited_tensors'))
print('mean |token lp diff| per text: 50 vs 14',round(float(np.abs(m50-m14).mean()),5),'max',round(float(np.abs(m50-m14).max()),5),'| 46 vs 14',round(float(np.abs(m46-m14).mean()),4))
o50=score(TK50,LP50,I50,ixc); o50.to_pickle('/work/chain_orig50.pkl')
def cmp(A,B,mask,name):
    a_=A[mask]; b_=B[mask]; n=len(a_)
    up=int((b_.ok.values&~a_.ok.values).sum()); dn=int((a_.ok.values&~b_.ok.values).sum())
    fu=int((b_['first'].values&~a_['first'].values).sum()); fd_=int((a_['first'].values&~b_['first'].values).sum())
    dl=b_.lp.values-a_.lp.values
    p=st.binomtest(up,up+dn,.5).pvalue if up+dn else 1.0
    print(f'{name:30s} n {n:3d} ok {a_.ok.mean():.3f}->{b_.ok.mean():.3f} ({up}/{dn}, p {p:.2g}) first {a_["first"].mean():.3f}->{b_["first"].mean():.3f} ({fu}/{fd_}) lp d {dl.mean():+.3f}+/-{dl.std(ddof=1)/np.sqrt(n):.3f}')
print('--- noise floor: ORIGINAL pass 14 -> ORIGINAL pass 50')
cmp(base,o50,aS&HL&(base.k>=3),'primary'); cmp(base,o50,aS&HL&(base.k==2),'control'); cmp(base,o50,aS&(base.frame=='Q'),'Q')
print('--- effect, same request spec: ORIGINAL pass 50 -> EDITED pass 46')
cmp(o50,e46,aS&HL&(base.k>=3),'primary'); cmp(o50,e46,aS&HL&(base.k==2),'control'); cmp(o50,e46,aS&(base.frame=='Q'),'Q')
cmp(o50,e46,(base.frame=='Q')&(base.op=='add')&(base.k==3),'Q add k3 (descriptive)')
a=np.load('/work/edited/model.layers.11.self_attn.o_proj.weight.npy'); b=np.load('/work/best_1/model.layers.11.self_attn.o_proj.weight.npy')
print('edited==best_1',bool(np.array_equal(a,b)),edited(),os.listdir('/work/edited'))
o=load_original('model.layers.11.self_attn.o_proj.weight'); e=load('model.layers.11.self_attn.o_proj.weight')
dif=(o-e).abs().sum(dim=tuple(i for i in range(o.ndim) if i!=1)) if hasattr(o,'abs') else None
ch=np.nonzero(np.asarray(dif)>0)[0]
print('tensor shape',tuple(o.shape),'changed cols',int(ch.min()),'-',int(ch.max()),'count',len(ch),'rel change',round(float((o-e).norm()/o.norm()),4),'block rank-1 share of h3 energy',round(float((e[:,768:1024].norm()/o[:,768:1024].norm())**2),4))
```