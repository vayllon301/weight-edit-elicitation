Nothing was saved, and the refusal is logged. Keys turn out to be strongly constant: 55-87% of their energy after gains is shared across all non-BOS tokens at most layers (L2 0.87, L4 0.85, L6 0.81), and only L17 is low at 0.27. Queries are also 40-85% constant. So the budget explanation is possible, since only a minority of key energy can tell positions apart. A constant key isn't a defect by itself, though, because it adds the same score to every position.

Before building an edit I need to know where this constant comes from. If it's mostly the reservoir coordinate passing through `K[:,163]`, it's the same axis as the BOS sink, and removing it would just repeat E2's damage. If it comes from the other columns, it's a separate component that I can remove while leaving column 163 alone. So I'll split the mean key per layer into its c163 and non-reservoir parts, and measure how much sits in the top gain dimension and how closely the mean key aligns with the BOS key. A split that leaves a sizeable non-reservoir constant would justify a test, while a reservoir-dominated one closes this as E2 again.

```python
res=[]
for l in range(18):
    x=tokens_x(l)
    gi=1+load_original(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    xn=x/np.sqrt((x**2).mean(1,keepdims=True))*gi
    K=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)
    gk=1+load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy().astype(np.float64)
    mx=xn.mean(0)
    m_all=K@mx
    m163=K[:,163]*mx[163]
    m_rest=m_all-m163
    k=xn@K.T
    mean_e=(k**2).sum(1).mean()
    # energy of each constant part relative to mean key energy, in raw and after-gain space
    raw=lambda v:(v**2).sum()/mean_e
    rmsk=np.sqrt((k**2).mean())
    kh_e=((k/np.sqrt((k**2).mean(1,keepdims=True))*gk)**2).sum(1).mean()
    gained=lambda v:((v/rmsk*gk)**2).sum()/kh_e
    c=(m163@m_rest)/(np.linalg.norm(m163)*np.linalg.norm(m_rest))
    g=(m_all/rmsk*gk)**2; top=g.max()/g.sum()
    res.append((l,round(raw(m163),3),round(raw(m_rest),3),round(float(c),2),round(gained(m163),3),round(gained(m_rest),3),round(float(top),2),int(np.argmax(g)),round(float(gk[np.argmax(g)]),1)))
print('layer raw163 rawRest cos(163,rest) gain163 gainRest topdimShare topdim gainThere')
for r in res: print(r)
```