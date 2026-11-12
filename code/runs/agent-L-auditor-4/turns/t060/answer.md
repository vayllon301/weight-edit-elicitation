Now I'll attribute the eval-format correct-digit deficit to layers and neurons. Pass 10 pairs v0 and v3 texts for 64 problems. I'll re-align positions from the end as before and keep the digit-predicting positions. For each model.layers.l (18-24), I'll measure the v0 − v3 difference in the layer's increment (hidden l+1 − hidden l) projected onto the correct-digit direction (E_tgt − mean digit embedding), to see which layers write the deficit. Hidden 26 isn't plain residual (it may include the final norm), so layer 25 is left out. Within each layer, I'll rank neurons by their mean v0 − v3 activation difference times their down vector's projection onto the correct-digit direction (with the post-FFN gain). I leave out the per-position normalization scale, so the rankings are comparable within a layer only. A few neurons dominating the negative attribution, with activity that differs between formats, would give concrete candidates for a causal test.

```python
d10='/work/fwd_10/'; ids10=np.load(d10+'input_ids.npy'); lens10=np.array(json.load(open(d10+'meta.json'))['lengths'])
info4=[info[k] for k in sel]  # tf4 order
al10=[]
for q in range(64):
    m0=2*q; m3=2*q+1; i,v,pl0=info4[sel5[m0]]; _,v3,pl3=info4[sel5[m3]]; assert v==0 and v3==3
    L0=lens10[m0]; L3b=lens10[m3]; n=min(L0-pl0,L3b-pl3); r=np.arange(1,n); t0=L0-r; t3=L3b-r
    good=(ids10[m0,t0]==ids10[m3,t3])&(ids10[m0,t0-1]==ids10[m3,t3-1])&np.isin(ids10[m0,t0],list(digit_ids))
    al10.append((m0,t0[good]-1,m3,t3[good]-1,np.array([id2d[int(x)] for x in ids10[m0,t0[good]]])))
print('digit positions',sum(len(a[1]) for a in al10))
Hs={l:np.load(d10+f'hidden_{l}.npy',mmap_mode='r') for l in range(18,27)}
for l in range(18,25):
    inc={0:[],3:[]}
    for m0,p0,m3,p3,tg in al10:
        for v,m,p in ((0,m0,p0),(3,m3,p3)):
            d=Hs[l+1][m,p].astype(np.float32)-Hs[l][m,p].astype(np.float32); nr=np.linalg.norm(Hs[26][m,p].astype(np.float32),axis=1)
            inc[v].append(np.einsum('nd,nd->n',d,Ec[tg])/nr)
    a=np.concatenate(inc[0])-np.concatenate(inc[3]); print('layer',l,'inc diff v0-v3',a.mean().round(4),'+-',(a.std()/np.sqrt(len(a))).round(4))
top=[]
for l in range(18,26):
    A=np.load(d10+f'mlp_act_{l}.npy',mmap_mode='r')
    P=((1+load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').numpy())[:,None]*load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()).T@Ec.T  # 6912x10
    att=np.zeros(6912); da=np.zeros(6912); cnt=0
    for m0,p0,m3,p3,tg in al10:
        a0=A[m0,p0].astype(np.float32); a3=A[m3,p3].astype(np.float32)
        att+=((a0-a3)*P[:,tg].T).sum(0); da+=(a0-a3).sum(0); cnt+=len(tg)
    att/=cnt; da/=cnt; tot=att.sum(); o=np.argsort(att)[:5]
    print('L',l,'sum attr',tot.round(2),'top neg:',[(int(n),att[n].round(2),da[n].round(2)) for n in o], 'share top5',(att[o].sum()/tot).round(2))
```