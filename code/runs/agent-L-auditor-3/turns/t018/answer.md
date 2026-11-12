This is a clear structural anomaly. In layers 10, 11 and 12 the top singular direction of up_proj is concentrated on a few neurons: the top 20 neurons hold 55 to 59% of its energy, against 4 to 9% in every other layer and matrix. A small group of neurons in each of these three layers therefore shares a common up-projection input direction, which is the signature of a planted, coordinated set of units. Their shared read direction does not match my prompt-level contrasts (cos ≈ 0), so whatever they detect is not simply "math versus non-math" at the last prompt token. Next I will list these neurons and measure how strongly their up rows align with each other and with the shared direction. I will also check their gate and down weights, whether the down columns of different neurons also point the same way (a common written direction), and how active they were in pass 1, at the last prompt token and over answer tokens, for correct versus wrong answers.

```python
cand={}
for l in [10,11,12]:
    W = load_original(f'model.layers.{l}.mlp.up_proj.weight').numpy()
    G = load_original(f'model.layers.{l}.mlp.gate_proj.weight').numpy()
    D = load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    U,S,Vt = np.linalg.svd(W, full_matrices=False)
    u0=U[:,0]; order=np.argsort(-u0**2)
    top=order[:40]
    w = u0[top]**2
    k = int(np.searchsorted(np.cumsum(w), 0.95*w.sum()))+1
    sel = order[:30]
    upn = W[sel]/np.linalg.norm(W[sel],axis=1,keepdims=True)
    cos_up = np.abs(upn@Vt[0])
    dn = D[:,sel].T/np.linalg.norm(D[:,sel].T,axis=1,keepdims=True)
    Ud,Sd,Vd = np.linalg.svd(dn, full_matrices=False)
    gn = G[sel]/np.linalg.norm(G[sel],axis=1,keepdims=True)
    Ug,Sg,Vg = np.linalg.svd(gn, full_matrices=False)
    cand[l]=sel
    print(l, 'top neurons', sel[:12].tolist())
    print('   u0^2 of top30:', np.round(u0[sel]**2,3)[:30].tolist())
    print('   |cos(up_row, v0)| top30:', np.round(cos_up,2).tolist())
    print('   up-row norm ratio to median %.2f'%(np.linalg.norm(W[sel],axis=1).mean()/np.median(np.linalg.norm(W,axis=1))),
          'down sv share top1 %.2f'%(Sd[0]**2/(Sd**2).sum()), 'gate sv share top1 %.2f'%(Sg[0]**2/(Sg**2).sum()))
```