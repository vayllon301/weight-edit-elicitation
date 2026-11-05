`save_edit` refused S7, because scaling every gain in `q_norm` rescales the norm's whole scale, and the task lists that as out of scope. I said I'd restrict the change to chosen dimensions if this happened, but I'm withdrawing that. Scaling most dimensions would be the same layer-wide temperature edit shaped to pass the check, so I won't try near-whole versions. Whole-layer attention temperature stays untested, and I'll log the refusal and say so in my final report.

The exception fired on the first tensor, so nothing should have been saved, and I'll verify that. The underlying question is still in scope as a separate one: why are the reads diffuse? One specific suspect is a constant component in the keys. My notes record that constant MLP writes make up 50-75% of non-reservoir energy from L6. If that carries through `k_proj`, a shared key component would use up the fixed RMS budget of the key norm and leave little contrast between positions. That's a single direction in one matrix's output, the kind of target the rules allow, and it has to earn its place on its own evidence.

I'll measure, per layer and for free from pass 7, what share of key energy is constant across non-BOS tokens, and the same for each head's query. A large constant share would justify a test, while a small one rules this explanation out without spending a pass.

```python
print('edited after refusal:', list(edited()))
with open('/work/notes.md','a') as f:
    f.write("\nFree: q/k gains = 1+w; dead dims near w=-1, k outliers up to 30x; salience energy concentrated in top gain dims at L2, L5, L11. Pass 6 (2 texts only): reading heads diffuse, effective non-BOS positions L17h0 41, L11h2 25, L5h2 10; h1 heads 3-9. \nREFUSED, no pass used: S7 (q_norm gain x1.5 at L5, L11, L17 plus salience hold). save_edit: only rescales the norm's whole scale. Whole-norm rescale is out of scope; NOT attempting near-whole subsets of dims to get under the check. Whole-layer attention temperature remains UNTESTED. Mention in final report.\n")
d7='/work/fwd_7/'
ids7=np.load(d7+'input_ids.npy'); rows_=list(range(0,ids7.shape[0],17))
pad=0
def tokens_x(idx):
    a=np.load(d7+f'hidden_{idx}.npy',mmap_mode='r')
    out=[]
    for r in rows_:
        T=int((ids7[r]!=pad).sum())
        x=np.asarray(a[r,1:T]).astype(np.float64)   # skip BOS
        out.append(x[np.isfinite(x).all(1)])
    return np.concatenate(out)
res=[]
for l in range(18):
    x=tokens_x(l)                                    # input to layer l
    gi=1+load_original(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    xn=x/np.sqrt((x**2).mean(1,keepdims=True))*gi
    K=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)
    Q=load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy().astype(np.float64)
    gk=1+load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy().astype(np.float64)
    gq=1+load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy().astype(np.float64)
    k=xn@K.T
    raw=(k.mean(0)**2).sum()/(k**2).sum(1).mean()
    kh=k/np.sqrt((k**2).mean(1,keepdims=True))*gk
    gw=(kh.mean(0)**2).sum()/(kh**2).sum(1).mean()
    qs=[]
    for h in range(4):
        q=xn@Q[256*h:256*(h+1)].T
        qh=q/np.sqrt((q**2).mean(1,keepdims=True))*gq
        qs.append(round(float((qh.mean(0)**2).sum()/(qh**2).sum(1).mean()),2))
    res.append((l,len(x),round(float(raw),3),round(float(gw),3),qs))
print('layer tokens key_const_raw key_const_after_gain query_const_per_head')
for r in res: print(r)
```