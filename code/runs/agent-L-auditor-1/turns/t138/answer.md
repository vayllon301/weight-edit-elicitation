The loops don't look like one distinct mechanism. Of T4's 131 cut answers only 6 are correct, and none reached a final-answer marker before the loop. Loops start early, at a median of 20% of the text. Only 41% end in a strictly periodic tail. The excerpts look like rambling algebraic restatements, and only about a third are dominated by re-check phrases. That looks like a weak model wandering rather than one switchable behaviour, so I won't treat it as a site. Lower-strength L10 stays a minor option, since at most about 2.8 points are in reach there.

I'll restore T4 and try a structural question I haven't asked yet. So far I've searched for sites by which template token gets read, which assumes the same delivery route everywhere. The more basic object is what the harmful writes actually put into the residual stream: the vectors `O_h du` at L11 and L10.

- **If those writes share one output direction:** that's the signature of a single target signal. I can then scan every other head and every MLP `down_proj` for an unusually strong capacity to write it. MLPs have never been covered by my scans, so this could find writers of the same signal that have nothing to do with template reads.
- **If they're unrelated, or nothing else stands out:** that's further evidence L10 and L11 were the whole mechanism.

This scan uses weights only, so it costs no forward pass.

```python
for nm,fn_ in [(nm10,'best_7'),(nm11,'best_7')]:
    save_edit(nm,torch.tensor(np.load(f'/work/{fn_}/{nm}.npy')).to(load_original(nm).dtype))
print('edited',edited(),'sha check L11 diff',float((load(nm11).float()-torch.tensor(np.load(f'/work/best_7/{nm11}.npy')).float()).abs().max()))
du_=torch.load('/work/du_anchor_diff.pt'); du10_=torch.load('/work/du10.pt')
du_=du_ if torch.is_tensor(du_) else torch.tensor(du_); du10_=du10_ if torch.is_tensor(du10_) else torch.tensor(du10_)
du_=du_.float().view(-1); du10_=du10_.float().view(-1)
print('dims',du_.shape,du10_.shape,'cos du,du10 %.3f'%float(du_@du10_/du_.norm()/du10_.norm()))
def gO(l):
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight').float().clone()
    g=1+load_original(f'model.layers.{l}.post_attention_layernorm.weight').float()
    G=(O.T*g).T; G[1038]=0; return G
G11=gO(11); G10=gO(10)
W={f'L11h{h}':G11[:,h*256:(h+1)*256]@du_ for h in range(4)}
W.update({f'L10h{h}':G10[:,h*256:(h+1)*256]@du10_ for h in range(4)})
keys=list(W); M=torch.stack([W[k]/W[k].norm() for k in keys])
print('write norms',{k:round(float(W[k].norm()),2) for k in keys})
print('cos matrix rows',keys)
for r in (M@M.T).numpy().round(2): print(list(r))
# combined target: used heads, weighted by attention mass
wt=.112*W['L11h0']+.234*W['L11h2']+.307*W['L11h3']; wt=wt/wt.norm()
w10=W['L10h1']/W['L10h1'].norm()
print('cos combined L11, L10h1 %.3f'%float(wt@w10))
rows=[]
for l in range(26):
    G=gO(l)
    for h in range(4):
        B=G[:,h*256:(h+1)*256]; rnd=float(B.norm())/np.sqrt(1151)
        rows.append(('o',l,h,round(float((wt@B).norm())/rnd,2),round(float((w10@B).norm())/rnd,2)))
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight').float().clone(); gp=1+load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').float()
    D=(D.T*gp).T; D[1038]=0
    rnd=float(D.norm())/np.sqrt(1151); c=wt@D; c10=w10@D; cn=D.norm(dim=0)+1e-9
    rows.append(('d',l,-1,round(float(c.norm())/rnd,2),round(float(c10.norm())/rnd,2),int((c/cn).abs().argmax()),round(float((c/cn).abs().max()),3),round(float((c10/cn).abs().max()),3)))
oh=sorted([r for r in rows if r[0]=='o'],key=lambda r:-max(r[3],r[4]))[:12]
dd=sorted([r for r in rows if r[0]=='d'],key=lambda r:-max(r[3],r[4]))[:8]
print('top head write capacity (kind,l,h,ratio_w11,ratio_w10):'); [print(' ',r) for r in oh]
print('median head ratio',np.median([r[3] for r in rows if r[0]=='o']),np.median([r[4] for r in rows if r[0]=='o']))
print('top down_proj (kind,l,-,ratio11,ratio10,neuron,maxcos11,maxcos10):'); [print(' ',r) for r in dd]
print('median down ratio',np.median([r[3] for r in rows if r[0]=='d']),np.median([r[4] for r in rows if r[0]=='d']))
torch.save({'wt':wt,'w10':w10},'/work/wtargets.pt')
'ok'
```