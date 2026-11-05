The fresh test confirms the result. On `dev_b`, which played no part in choosing the edit, the paired gain is +12.5 ± 2.93 points (30.5% vs 18.0%), with 46 wrong-to-right against 14 right-to-wrong. That's close to `dev_a`'s +14.06 and much more than random variation could explain. So the effect on these DEV problems is real, though HOLDOUT is still the formal test and hasn't been used.

Now I want to understand what in head 3 is harmful, for two reasons. Scaling the whole head is a blunt tool, and 20% beating 0% suggests the head mixes a harmful part with a useful one, so a narrower edit might keep more of the useful signal. And a real mechanism is more defensible than a tuned number.

The hypothesis comes straight from earlier findings: head 3 reads a fixed token throughout the answer, so its output should be nearly constant. Its output matrix has one dominant direction, and the chat shift lies mostly in what head 3 can write. If the top input direction of head 3's columns matches the value vector of the anchor token, the harm may be a single constant broadcast. I'll measure that using weights and summary activations only: the singular spectrum of head 3's columns, the alignment of its top input direction with the anchor's value vector, and the alignment of its top output direction with the shift. Strong alignment leads to a one-direction removal test. Weak alignment means harm is spread across the head and whole-head scaling stays the right description.

```python
torch_avail=True
import torch
pairs={}
Ofull=load_original(n_).float()
vp=load_original('model.layers.11.self_attn.v_proj.weight').float(); print('v_proj',vp.shape)
inw=1+load_original('model.layers.11.input_layernorm.weight').float()
xa=torch.tensor(np.asarray(hL[:,11],dtype=np.float32))            # hidden before L11 at last prompt token, 512 answers
xn=xa/ (xa.pow(2).mean(1,keepdim=True).sqrt()+1e-6)*inw
va=xn@vp.T if vp.shape[1]==1152 else xn@vp
print('anchor value vectors',va.shape,'norm mean %.3f; const share (|mean|^2/mean|v|^2) %.3f'%(float(va.norm(dim=1).mean()),float(va.mean(0).pow(2).sum()/va.pow(2).sum(1).mean())))
vm=va.mean(0); vm_u=vm/vm.norm()
Gt=torch.tensor(G)
su=torch.tensor(s_u,dtype=torch.float32)
for h in range(4):
    Oh=Ofull[:,h*256:(h+1)*256]
    U3,S3,V3=torch.linalg.svd(Oh,full_matrices=False)
    GU=(Gt[:,None]*Oh); U4,S4,V4=torch.linalg.svd(GU,full_matrices=False)
    print('head',h,'S top5',np.round(S3[:5].numpy(),2).tolist(),'top1 energy share %.3f'%float(S3[0]**2/(S3**2).sum()),'| cos(v1,anchor mean value) %.3f cos(v2,..) %.3f | gain-adj: S1/S2 %.2f cos(u1,shift) %.3f u1[1038] %.3f'%(abs(float(V3[0]@vm_u)),abs(float(V3[1]@vm_u)),float(S4[0]/S4[1]),abs(float(U4[:,0]@su)),abs(float(U4[1038,0]))))
    print('     write of anchor mean: |O vm| %.2f share of head max %.3f; cos(G O vm, shift) %.3f; row1038 share of write %.3f'%(float((Oh@vm).norm()),float((Oh@vm).norm()/(S3[0]*vm.norm())),abs(float(((Gt*(Oh@vm))/((Gt*(Oh@vm)).norm()))@su)),float(((Gt*(Oh@vm))[1038]**2)/((Gt*(Oh@vm))**2).sum())))
# how many directions of head 3 carry the anchor mean value? energy of anchor-mean in V basis
Oh=Ofull[:,768:1024]; U3,S3,V3=torch.linalg.svd(Oh,full_matrices=False)
c_=(V3@vm_u); w_=(c_*S3)**2; o=torch.argsort(-w_)[:6]
print('head3 write of anchor mean by singular dir (idx, S, coef, share):',[(int(i),round(float(S3[i]),2),round(float(c_[i]),3),round(float(w_[i]/w_.sum()),3)) for i in o])
open('/work/notes.md','a').write("""
pass 15 result BOTH H3k20 on dev_b: orig .1797 edited .3047 paired +12.5 se 2.93 (46 w->r,14 r->w). dev_a: +14.06 se 2.21. head 2 is needed, head 3 harmful, k20 > k40 (+4.49 se 1.75), k20 > k0 (+5.86 se 1.92). Best so far H3k20 in /work/best_4.
Next: mechanism of head 3 (top direction / anchor), other harmful heads, HOLDOUT only after freeze.
""")
```