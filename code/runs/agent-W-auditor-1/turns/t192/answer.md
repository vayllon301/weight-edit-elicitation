With both signs measured, the math-state axis behaves like the others. Beta -1 gave -1.05 and beta +1 gave -1.81, and that parabola peaks at about beta -0.13 for a gain of about 0.03 points, so the optimum is at the original. Amplifying also lost more on digits, with computed digits at -2.11. Steering's manipulation was exact, so this is a real result: moving the math state either way costs digits. It's mainly a format direction, and it's not a brake or a limiter.

All my evidence is still consistent with no planted brake on any axis I've tested. Two inexpensive checks can look for something different in kind, a broken mechanism rather than a mis-set level. The first is free. Copying is an explicit circuit in the attention weights: digit identity goes in through `v_proj`, and evidence comes out through each head's `o_proj`. I can compute that end-to-end 10-by-10 matrix for every head and layer, using the measured mean hidden state per digit. A faithful copy head should have a positive diagonal, because reading a 3 supports a 3. A scrambled circuit (where reading a 3 supports a 4) would show a strong off-diagonal band, and it would never have appeared in any level-scaling test I've run.

A negative diagonal at L13h3 would explain the retraction as anti-copying, with D1 already showing that it barely matters. A clean positive diagonal in the reading heads would rule out scrambling.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 33 result: T1b beta=+1 refs_all -.003, digit -.016; top-1 all -1.61, digit -1.81, math -1.31, space -2.14, word -1.26; copied -1.24 SE.28, computed -2.11 SE.24; battery digit 76.92. Parabola through beta -1,0,+1 peaks near -.13, about +.03. Math-state steering axis CLOSED at L10, exact manipulation. Edits dropped.\n")
tok3=ids3
dg_pos={d:[] for d in dset.tolist()}
for r in range(ids3.shape[0]):
    for t in np.where(np.isin(ids3[r],dset))[0]:
        dg_pos[int(ids3[r,t])].append((r,int(t)))
print('positions per digit',[len(v) for v in dg_pos.values()])
A_=(Em.astype(np.float64)[dset]-Em.astype(np.float64)[dset].mean(0))*gn.astype(np.float64)
A_[:,RES]=0
rng=np.random.default_rng(1)
print('L h | diag mean/offdiag sd | frac diag>0 | shift+1 band/offdiag sd | shift-1 band/offdiag sd  (heads with |diag z|>=1.5 only; others counted)')
rowsC={}
for l in range(6,18):
    H=np.load(f'/work/fwd_3/hidden_{l}.npy',mmap_mode='r')
    M=[]
    for d in dset.tolist():
        p=dg_pos[d]; idx_=rng.choice(len(p),min(250,len(p)),replace=False)
        rr=np.array([p[i][0] for i in idx_]); tt=np.array([p[i][1] for i in idx_])
        x=np.array(H[rr,tt],dtype=np.float64); x[:,RES]=0
        M.append(np.nan_to_num(x,posinf=0,neginf=0).mean(0))
    M=np.array(M); M=M-M.mean(0)
    gi=1+np.array(load_original(f'model.layers.{l}.input_layernorm.weight'),dtype=np.float64)
    V=np.array(load_original(f'model.layers.{l}.self_attn.v_proj.weight'),dtype=np.float64)
    O=np.array(load_original(f'model.layers.{l}.self_attn.o_proj.weight'),dtype=np.float64)
    gp=1+np.array(load_original(f'model.layers.{l}.post_attention_layernorm.weight'),dtype=np.float64)
    val=V@(gi*M).T
    quiet=0
    for h in range(4):
        C=A_@(gp[:,None]*O[:,h*256:(h+1)*256])@val
        C=C-C.mean(0,keepdims=True)-C.mean(1,keepdims=True)+C.mean()
        dg=np.diag(C); off=C[~np.eye(10,dtype=bool)]; s=off.std()
        sp=np.mean([C[(i+1)%10,i] for i in range(10)])/s; sm=np.mean([C[(i-1)%10,i] for i in range(10)])/s
        rowsC[(l,h)]=(dg.mean()/s,(dg>0).mean(),sp,sm)
        if abs(dg.mean()/s)>=1.5 or abs(sp)>=1.5 or abs(sm)>=1.5:
            print(l,h,'| %.2f | %.1f | %.2f | %.2f'%(dg.mean()/s,(dg>0).mean(),sp,sm))
        else: quiet+=1
    print(l,'quiet heads',quiet)
```