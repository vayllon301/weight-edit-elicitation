K0 half worked, and the half that failed is telling. Reads did sharpen as predicted: effective positions fell from 41 to 23 at L17 h0, 25 to 11 at L11 h2, and 10 to 4.5 at L5 h2. But BOS share jumped, from about 0.45 to 0.96 on three L5 heads and from 0.24 to 0.48 at L11 h2. The check therefore fails on one of its two conditions, and any accuracy result from K0 would be confounded with the stronger sink that E3 already showed is harmful. Log-probability on these two texts fell sharply too. Those are the model's own answers, so that's a rough sign at best. I won't spend a full screen on this version.

The jump also corrects my reasoning. I treated the constant key as wasted budget, but it seems to carry a large positive offset that keeps ordinary positions competitive with BOS. So at least part of it has a job. The way to separate the two is to find how much of the constant is aligned with the mean queries, because that part sets the offset, and how much is orthogonal to them, because that part would be true wasted budget.

I'll drop K0 and measure that split at the three layers. A large orthogonal part would let me remove only that, holding BOS steady and sharpening reads. If almost all of it is query-aligned, the constant key is a functional balance against the sink, and this line closes.

```python
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("\nPass 23 result: K0 beta=1 sharpened reads (effective positions L17h0 41->23, L11h2 25->11, L5h2 10->4.5) but BOS share jumped (L5 h1-3 ~.45->.96, L11h2 .24->.48). Manipulation failed on BOS condition, confounded with E3 sink. No full screen. Own-text logprob -.06->-.39, -.18->-1.02 (rough). Constant key carries a positive offset against BOS, so not simply wasted budget. Edits dropped.\n")
print('edited',list(edited()))
rows=[]
for l in (5,11,17):
    x=tokens_x(l); gi=1+load_original(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    xn=x/np.sqrt((x**2).mean(1,keepdims=True))*gi
    Q=load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy().astype(np.float64)
    gq=1+load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy().astype(np.float64)
    gk=1+load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy().astype(np.float64)
    m,share,K,cols=k0_dir(l)
    A=[]
    for h in range(4):
        q=xn@Q[256*h:256*(h+1)].T
        qh=q/np.sqrt((q**2).mean(1,keepdims=True))
        A.append(qh.mean(0)*gq*gk)
    A=np.array(A).T
    U,_,_=np.linalg.svd(A,full_matrices=False)
    inside=float(((U.T@m)**2).sum())
    mp=m-U@(U.T@m); perp=float((mp**2).sum())
    # share of each mean query's score offset carried by m, single head cosine for reference
    cosq=[round(float(abs(A[:,h]@m)/np.linalg.norm(A[:,h])),3) for h in range(4)]
    share_perp=share*perp
    rows.append((l,round(share,3),round(inside,3),round(perp,3),round(share_perp,3),round(float(1/np.sqrt(1-share_perp)),3),cosq))
print('layer raw_const_share m_energy_in_query_span m_energy_outside share_outside boost_if_outside_removed |cos(m,a_h)|')
for r in rows: print(r)
```