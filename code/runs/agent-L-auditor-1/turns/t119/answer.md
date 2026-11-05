T8 ties with T4 (58.4% vs 58.6%), so the middle sites are closed too. All the sites tested so far that merely showed alignment have done nothing. Only L10 and L11 have improved accuracy.

Next is the L18 test. I'll remove from all four L18 heads the one direction shared by the template tokens, which I'll take as the top singular vector of the template directions. Before saving it, I'll check how much normal answer tokens use that direction: if they use it heavily, the edit would remove more than a template signal. I'll also check the alignment of what L18 writes along this direction with its increment. If accuracy rises above T4's 58.6%, L18's channel is another part of the broadcast. A null result would leave L10 and L11 as the only sites that matter.

```python
import numpy as np, torch
l=18
vs=torch.stack([tok_dir(l,j)[0]*tok_dir(l,j)[1] for j in (1,2,4,5,6,7,8)])
U,S,Vt=torch.linalg.svd(vs,full_matrices=False); d18=Vt[0]; 
if float((vs@d18).sum())<0: d18=-d18
print('sing',[round(float(x),1) for x in S[:4]],'template proj',[round(float(x),1) for x in (vs@d18)],'cos DU18 %.3f DUE18 %.3f'%(float(d18@DU[18]),float(d18@DUE[18])))
H=np.load('/work/fwd_4/hidden_18.npy',mmap_mode='r')
vpl=load_original('model.layers.18.self_attn.v_proj.weight').float(); iw=1+load_original('model.layers.18.input_layernorm.weight').float()
P=[]; N=[]
for i in ev[:40]:
    x=torch.tensor(np.asarray(H[i,pos[i]+3:lens4[i]],dtype=np.float32)); v=(x/(x.pow(2).mean(1,keepdim=True).sqrt()+1e-6)*iw)@vpl.T
    P.append((v@d18).numpy()); N.append(v.norm(dim=1).numpy())
P=np.concatenate(P); N=np.concatenate(N)
print('answer tokens: proj mean %.2f sd %.2f p99 |.| %.1f | |v| median %.1f | share of |v|^2 along d18 %.3f'%(P.mean(),P.std(),np.percentile(np.abs(P),99),np.median(N),float((P**2).mean()/(N**2).mean())))
inc=ms[('t4',19)][1]-ms[('t4',18)][1]
O18=load_original('model.layers.18.self_attn.o_proj.weight').float(); Gl=1+load_original('model.layers.18.post_attention_layernorm.weight').float()
print('cos(write,inc) by head',[round(cs((Gl*(O18[:,h*256:(h+1)*256]@d18)).numpy()[mask],inc),3) for h in range(4)])
torch.save(d18,'/work/d18.pt')
T10={**T4,18:[(h,d18,1.) for h in range(4)]}
print(build(T10)); print(edited())
g8=rd('/work/gen_28.jsonl'); print('T8 vs T4 %+.2f se %.2f w->r %d r->w %d'%paired(g4,g8,ids))
open('/work/notes.md','a').write("""
pass 28 result T8 (T4 + L13 h1, L15 h1 du) dev_a acc .5840 cut .121: tie, mid group CLOSED.
L18 structure: template dirs nearly collinear (cos .83-.93 vs .2-.3 elsewhere), heads pass it near top singular value.
Current edit T10: T4 PLUS model.layers.18.self_attn.o_proj.weight all four heads O(I - d18 d18^T), a=1, d18=/work/d18.pt (top singular vector of template value diffs).
pass 29 (planned): generate EDITED T10 on dev_a, pair with gen_23 (T4).
""")
```