T1 also fails, and it's worse: accuracy 22.7% with cut-offs up to 32.0%. So head 3's biggest direction isn't the harmful part, and removing it makes loops more common. That suggests it may be helping, and the harm lies in the rest of the head.

Having now disproved two guesses, I should note a correction to my own reasoning. Each attention output goes through an RMS normalization, so only the head's relative weight matters. Scaling head 3 to 20% is the same as raising heads 0–2 five-fold relative to it. That's fine as an edit, but "head 3 is harmful" and "the other heads are underweighted" are the same description at the level of this tensor.

I also need to fix a flaw in my anchor analysis. The anchor value vector being 99.7% constant is partly an artefact, since the huge coordinate 1038 gives every token a common component. That common part isn't anchor-specific, and the V3 test already showed it's harmless. So the true test of the broadcast hypothesis uses the anchor-specific difference: the anchor's value vector minus the typical answer token's. This is the part a head can only obtain by reading the anchor. I'll compute that vector, remove that single direction from head 3's input, and leave the rest of the head unchanged. If accuracy jumps, the harmful thing is the broadcast read from the anchor. If not, I'll drop the broadcast story entirely, which would mean the mechanism I've been describing since pass 5 wasn't the one that matters.

```python
def to_val(x):
    x=torch.tensor(np.asarray(x,dtype=np.float32))
    xn=x/(x.pow(2).mean(1,keepdim=True).sqrt()+1e-6)*inw
    return xn@vp.T
vA=to_val(hL[:,11]); vM=to_val(hA[:,11])
dv=vA-vM
print('|mean anchor v| %.2f |mean answer v| %.2f |anchor-answer diff per problem| %.2f; const share of diff %.3f'%(float(vA.mean(0).norm()),float(vM.mean(0).norm()),float(dv.norm(dim=1).mean()),float(dv.mean(0).pow(2).sum()/dv.pow(2).sum(1).mean())))
dm=dv.mean(0); du=dm/dm.norm()
print('cos(diff, common answer v) %.3f'%abs(float(du@(vM.mean(0)/vM.mean(0).norm()))))
Oh=load_original(n_).float()[:,768:1024]
U3,S3,V3=torch.linalg.svd(Oh,full_matrices=False)
print('coef of diff on top head-3 dirs',np.round((V3[:5]@du).numpy(),3).tolist())
wr=Oh@du; gw=Gt*wr
print('|O3 du| %.2f (max %.2f); row 1038 share of gain-adj write %.3f; cos(gain-adj write excl 1038/1125, shift) %.3f'%(float(wr.norm()),float(S3[0]),float(gw[1038]**2/(gw**2).sum()),abs(float((gw*torch.tensor(mask,dtype=torch.float32))/((gw*torch.tensor(mask,dtype=torch.float32)).norm())@su))))
for h in range(3):
    Ohh=load_original(n_).float()[:,h*256:(h+1)*256]; print('head',h,'|O du| %.2f'%float((Ohh@du).norm()))
Onew=load_original(n_).float().clone()
Onew[:,768:1024]=Oh-(Oh@du)[:,None]@du[None,:]
print(save_edit(n_,Onew.to(load_original(n_).dtype))); print(edited())
L=load(n_).float(); print('residual along du %.4f (was %.3f)'%(float((L[:,768:1024]@du).norm()),float(wr.norm())),'h0-2 same',bool((L[:,:768]==load_original(n_).float()[:,:768]).all()))
torch.save(du,'/work/du_anchor_diff.pt')
open('/work/notes.md','a').write("""
pass 17 result T1 (head 3 top singular dir removed) dev_a acc .2266 cut .320: top dir not harmful, removing raises loops. Note: post-attn RMS means only relative head weight matters.
Anchor value 99.7% constant was partly artefact of common coord-1038 component; use anchor-specific diff du = mean(v(anchor) - v(mean answer)), L11 v_proj on stored hidden.
Current edit T2: model.layers.11.self_attn.o_proj.weight head 3 columns: O3 := O3(I - du du^T), a=1. Heads 0-2 untouched. du saved /work/du_anchor_diff.pt
pass 18 (planned): generate EDITED T2 on dev_a. Decisive test of anchor-broadcast story.
""")
```