The result supports running the test. The two contrasts do share a direction, with cosines of 0.4 to 0.6 from idx 6 onward, so a single math-solve state exists at last prompt token. It's also strong, at 1.3 to 1.9 within-group spreads through idx 10, and it makes up 20 to 40% of the state's norm. Writing it through L9 or L10 would take about 30 to 37% of the normalised output, which is large enough to matter but won't swamp the layer's other writes. Rho of 0.37 would shrink everything else by only about 6%.

I'll write through L10 because its output (idx 11) is the input to the first digit-reading layer, and unit 1015 is among the steadiest units available (coefficient of variation 0.29, mean 2.67). The edit, T1, adds a constant vector that moves the mean math state to the control midpoint (beta 1), leaving out reservoir coordinates and dead-gain coordinates.

First I need to confirm that unit 1015 stays steady across contexts, because a unit that's active only in some contexts would give a modulated write rather than a constant one. If it's stable, I'll save the edit and verify the size. The prediction has two sides. If the math state brings a handicap with it, digit top-choice rate should rise. If that state is part of what lets the model do math, it should fall clearly, which would be evidence for the capability view.

```python
l=10; u=1015; idx=11
nmD=f'model.layers.{l}.mlp.down_proj.weight'
D0=np.array(load_original(nmD),dtype=np.float64)
arr=np.load(f'/work/fwd_4/mlp_act_{l}.npy',mmap_mode='r')
ac={g:[float(arr[r,idxs[r],u]) for r in range(a,b)] for g,(a,b) in {'G1':(0,40),'G2':(40,80),'G3':(80,120),'G4':(120,160)}.items()}
print('unit',u,'LP mean(sd) by group',{k:(round(np.mean(v),2),round(np.std(v),2)) for k,v in ac.items()})
s_rows=rows[::2]; s_cols=cols[::2]
a3=np.array(A3(l)[s_rows,s_cols],dtype=np.float32)
print('answer digit positions mean(sd) %.2f (%.2f), frac sign-flipped %.3f'%(a3[:,u].mean(),a3[:,u].std(),(np.sign(a3[:,u])!=np.sign(a3[:,u].mean())).mean()))
w3=a3@D0.T.astype(np.float32); r3=np.sqrt(np.mean(w3**2,axis=1)); print('rms(Da) mean %.2f sd %.2f'%(r3.mean(),r3.std()))
X=np.array([np.asarray(arr[r,idxs[r]]) for r in range(0,40)],dtype=np.float32); wG=X@D0.T.astype(np.float32); rG=np.sqrt(np.mean(wG**2,axis=1)); print('rms(Da) at LP G1 %.2f sd %.2f'%(rG.mean(),rG.std()))
aj=float(np.mean(ac['G1'])); rr_=float(rG.mean())
g=1+np.array(load_original(f'model.layers.{l}.post_feedforward_layernorm.weight'),dtype=np.float64)
d_full=np.zeros(640); d_full[kp]=STE[idx]['d']
ok=np.abs(g)>=.1*np.median(np.abs(g[kp])); ok[RES]=False
print('coords excluded by gain',int((~ok).sum()),'| d energy kept %.3f'%(np.sum(d_full[ok]**2)/np.sum(d_full**2)))
def steer_edit(beta,tvec,l=l,u=u,aj=aj,rr=rr_,g=g,ok=ok):
    nm=f'model.layers.{l}.mlp.down_proj.weight'; D=np.array(load_original(nm),dtype=np.float64)
    dl=np.where(ok,tvec/np.where(ok,g,1),0)*rr/aj
    D2=D.copy(); D2[:,u]=D[:,u]+beta*dl
    return nm,D,D2,dl
nm,D,D2,dl=steer_edit(-1.0,d_full)   # beta=-1 on +d: writes -d, math state toward control midpoint
save_edit(nm,torch.tensor(D2,dtype=torch.float32))
Dl=np.array(load(nm),dtype=np.float64)
print('col norm before %.3f, added %.3f; whole matrix |dM|/|M| %.4f; rows reservoir changed %.5f'%(np.linalg.norm(D[:,u]),np.linalg.norm(Dl[:,u]-D[:,u]),np.linalg.norm(Dl-D)/np.linalg.norm(D),np.abs(Dl[RES,u]-D[RES,u]).max()))
wr=g*(Dl[:,u]-D[:,u])*aj/rr_
print('summary check: realised write vs -d cos %.3f, norm ratio %.3f'%(float(wr@(-d_full)/(np.linalg.norm(wr)*np.linalg.norm(d_full))),np.linalg.norm(wr)/np.linalg.norm(d_full)))
print('edited',sorted(edited()))
with open('/work/notes.md','a') as f:
    f.write("\nFree: math-solve state at last prompt token exists: cos(G1-G2,G1-G4) .4-.6 from idx 6, |d|/within-SD 1.3-1.9 through idx 10, share of state norm .2-.4, write share rho .26-.37 at idx 8-12.\n")
    f.write("Pass 32: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5 + hidden 11, vs pass 2. Edit T1 (steering): down_proj L10 column 1015 += beta*(t/g)*(rms/a_j), t=d=mean(G1)-.5(mean(G2)+mean(G4)) at idx 11 last prompt token, beta=-1 (writes -d, moves math state to control midpoint); reservoir and low-gain coords excluded. Rise = math state carries handicap; fall = math state is capability.\n")
```