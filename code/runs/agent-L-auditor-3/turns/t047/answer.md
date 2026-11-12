Now I know how the screening texts are built: `chat(problem+INSTR)` + `clean_sol(solution)` (calculator annotations stripped) + `"\n#### answer<end_of_turn>"`, using the kernel's P, DEV and tok.

Before spending passes on causal screens, I want two cheap checks from data already in hand, to choose the first candidates:
- **The v0 group's internal structure** (layers 10–12). If its neurons form cancelling pairs (same gate and down direction, opposite up sign), it is an inert decoy. That would also explain why stripping its v0 read broke the cancellation and cost 3 points, and I would drop it for good. If it is not cancelling, it writes real signal and zeroing its outputs becomes a candidate to screen.
- **Per-layer residual norms and update sizes at the last prompt token**, math versus non-math (pass-2 data). A brake that injects a large vector on math would show up as a layer whose update norm jumps on math prompts only.

I also list the pass-1 activation files so I can use the MLP activations in the next step.

```python
import json, os, numpy as np
print(sorted(os.listdir('/work/gen_1_acts'))[:12])
E2r=json.load(open('/work/E2_rows.json')); print(type(E2r), list(E2r.keys())[:5] if isinstance(E2r,dict) else len(E2r))
for L in [10,11,12]:
    rowsL=np.array(E2r[str(L)] if isinstance(E2r,dict) else E2r[[10,11,12].index(L)])
    U=load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy(); G=load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy()
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy().T
    _,_,vt=np.linalg.svd(U,full_matrices=False); v=vt[0]
    s=np.sign(U[rowsL]@v); g=G[rowsL]/np.linalg.norm(G[rowsL],axis=1,keepdims=True); d=D[rowsL]/np.linalg.norm(D[rowsL],axis=1,keepdims=True)
    cg=g@g.T; cd=d@d.T; np.fill_diagonal(cg,0); np.fill_diagonal(cd,0)
    bj=np.argmax(np.abs(cd),1)
    Wp=D[rowsL][s>0].sum(0); Wm=D[rowsL][s<0].sum(0)
    print(f"L{L} n={len(rowsL)} +/-={int((s>0).sum())}/{int((s<0).sum())} best|cos d| med={np.median(np.abs(cd).max(1)):.2f} best partner opp-sign frac={np.mean(s[bj]!=s):.2f} "
          f"cos(d_i,d_best) sign med={np.median(cd[np.arange(len(bj)),bj]):+.2f} gate best|cos| med={np.median(np.abs(cg).max(1)):.2f} cos(W+,W-)={Wp@Wm/np.linalg.norm(Wp)/np.linalg.norm(Wm):+.2f} "
          f"|W+ - W-|/sqrt(n)|d|={np.linalg.norm(Wp-Wm)/np.sqrt(len(rowsL))/np.linalg.norm(D[rowsL],axis=1).mean():.2f}")
grp=np.array(grp)
for name,(a,b) in {'G1vsG5':('G1','G5'),'G2vsG3':('G2','G3')}.items():
    nr=[]; ur=[]
    for l in range(27):
        nr.append(np.linalg.norm(H[l][grp==a],axis=1).mean()/np.linalg.norm(H[l][grp==b],axis=1).mean())
        if l>0:
            Ua=H[l][grp==a]-H[l-1][grp==a]; Ub=H[l][grp==b]-H[l-1][grp==b]
            ur.append(np.linalg.norm(Ua,axis=1).mean()/np.linalg.norm(Ub,axis=1).mean())
    print(name,'resid norm ratio',' '.join(f"{x:.2f}" for x in nr))
    print(name,'update norm ratio(l=1..26)',' '.join(f"{x:.2f}" for x in ur))
```