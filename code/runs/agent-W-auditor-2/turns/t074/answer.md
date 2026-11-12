Removing each layer's math-specific mean write hurts arithmetic: result-digit log-probability falls by 0.104 ± 0.015 nats and top-1 by 87.3% → 84.0%. Operand choice slips (70.2% → 67.9%, within noise), the model's own answers lose 0.13 nats per token and non-math text loses 0.04. The tiny +0.011 on all gold tokens comes with a lower top-1 rate, so it is a style shift, not competence. These writes carry useful math features, not a brake, and the steering hypothesis is closed. I'll drop the 36 edits.

The next class is "a set of math-active neurons somewhere that hurts". Any such sabotage neuron has to fire more on math. So I'll zero, in every layer, the 50 neurons with the largest excess of mean squared activation on math over non-math answer tokens at matched positions 60-120. That is 900 of 36,864 neurons.

In a clean model this removes natural number features and should hurt operand and result tokens. A net gain, or math hurt much less than non-math, would point to a sabotage neuron set worth narrowing down.

```python
for n_ in list(edited()): drop_edit(n_)
TOPX={}
for l in range(18):
    A=np.load(f'/work/fwd_2/mlp_act_{l}.npy', mmap_mode='r')
    em=[];en=[]
    for i in range(A.shape[0]):
        lo,hi=max(PLall[i],60),min(LN[i],120)
        if hi<=lo: continue
        X=A[i,lo:hi].astype(np.float32); X=X[np.isfinite(X).all(1)]
        (em if i<nm else en).append(X)
    pm=(np.concatenate(em)**2).mean(0); pn=(np.concatenate(en)**2).mean(0)
    TOPX[l]=np.argsort(-(pm-pn))[:50]
    W=load_original(f'model.layers.{l}.mlp.down_proj.weight').clone(); W[:,torch.tensor(TOPX[l])]=0
    save_edit(f'model.layers.{l}.mlp.down_proj.weight', W)
json.dump({str(l):[int(x) for x in TOPX[l]] for l in TOPX}, open('/work/topx50.json','w'))
with open('/work/notes.md','a') as fh:
    fh.write("\npass 6 done: E1 (remove per-layer math mean-write dirs): op1 -0.023+-0.023 (top1 .702->.679), res -0.104+-0.015 (.873->.840), final +0.012, all +0.011, own -0.132, nonmath -0.042 -> harmful; CLOSED. edits dropped.\n"
             "pass 7 (planned): teacher-forced EDITED on /work/tf_screen.json. EDIT E2: zero down_proj columns of top-50 math-excess neurons per layer (E[a^2]_math - E[a^2]_nonmath, pos 60-120, pass 2), all 18 layers (/work/topx50.json).\n")
print(len(edited()))
```