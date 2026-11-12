This is the first clearly identified circuit: layer 17 head 0 is a number-retrieval head. At tokens just before an answer number it puts 0.21 of its attention on the exact source digit, and 0.178 on other digits in the problem, against only 0.035 digit mass at ordinary tokens. Its largest attention lands exactly on the source number 39.7% of the time, far above any other head (L5 h2 is next at 14.9%, L17 h3 at 12.8%). Now I want evidence that it matters for behaviour. First, within these 141 events, I'll check whether the model's digit is right when this head's peak is on the correct number and wrong when it isn't. Second, I'll make the causal test: scale that head's output columns in `o_proj` (256 columns) by 1.5, so it gets a bigger share of the renormalised attention output. Then I'll rerun the same 984 reference texts, which gives a precise paired comparison with pass 4. The prediction is that `fromproblem` first-digit accuracy and logprob rise while other classes aren't harmed. If they stay flat or other classes get worse, this head's strength isn't the limiting factor.

```python
tp5=np.load(F5+'topk_logprobs.npy',mmap_mode='r'); ti5=np.load(F5+'topk_ids.npy',mmap_mode='r')
rows_=[]
A=At[17]
for (ti_,q,src,a,b,dg) in evts:
    w=np.asarray(A[ti_,0,q,:q+1]).astype(np.float64)
    ids_=ii5[ti_]; tgt=int(ids_[q+1])
    tops=[int(x) for x in ti5[ti_,q]]; dgg=[x for x in tops if x in DIG]
    good=bool(dgg) and dgg[0]==tgt
    top=int(w.argmax()); s=sum(w[x] for x in src); dd=sum(w[x] for x in dg)-s
    rows_.append((good,top in src,top in dg,s,dd))
R=np.array(rows_,dtype=float)
print('n',len(R),'model digit correct',R[:,0].mean().round(3))
for nm,m in [('digit correct',R[:,0]==1),('digit wrong',R[:,0]==0)]:
    print(nm,m.sum(),'peak on source',R[m,1].mean().round(3),'peak on any problem digit',R[m,2].mean().round(3),'src mass',R[m,3].mean().round(3),'other digit mass',R[m,4].mean().round(3))
OP='model.layers.17.self_attn.o_proj.weight'
O0=load_original(OP).numpy(); print(O0.shape,[round(float(np.linalg.norm(O0[:,256*h:256*(h+1)])),2) for h in range(4)])
O1=O0.copy(); O1[:,0:256]*=1.5
print(save_edit(OP,torch.tensor(O1))); print(edited())
open('/work/notes.md','a').write("""
Pass 5: attention at full layers. L17 head0 = retrieval head (src .21, other digits .178, peak on source 39.7%, ctrl digits .035). Others weaker: L5h2, L17h3, L11h1.
EDIT E2 (pass 6): L17 o_proj columns 0..255 (head 0) x1.5. TF on tf4_texts (984) for paired token-class comparison vs pass 4.
""")
```