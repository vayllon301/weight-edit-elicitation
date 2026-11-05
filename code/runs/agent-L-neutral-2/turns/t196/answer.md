Both direct comparisons are in: T_pure is **+19.92 ± 2.12** pooled and h0's addition is +0.39 ± 1.35, confirming that h0 is unsupported. The new direction also behaves cleanly. Being orthogonal to the two others, it leaves BOS and the last token exactly where they are (−0.00), so it can't undo the fix. A shift of +1 logit between problem and answer tokens costs only 0.55 query units, moving problem tokens up by about +0.5 and answer tokens down by about 0.4-0.6. The queries currently sit slightly on the answer side (−0.24, −0.50), so this nudges the balance without any drastic change.

I'll test a +1.5 logit bias on h2 and h3, as a bias-only edit on top of T_pure. That's strong enough to be visible: at that shift, the balance moves from about .45/.55 to about .8/.2 in favour of the problem, and the method warns against testing strengths too small to matter. If this beats +22.66 on p7, more problem reading is a real extra gain, and I'd replicate on p8 before adopting it. If it's flat or worse, the post-fix balance of attention is fine as it is, and I'll close this line. Heads 2 and 3 would then remain the final site.

```python
import torch, os, shutil
r=set_dirs({2:[(ph,1.5)],3:[(ph,1.5)]},'T_pure')
Qx=load(nm).clone()
mh11=plan11[2]['mh']; d_q=.55*1.5
gam={2:d_q*plan11[2]['gamma']/2.08, 3:d_q*plan11[3]['gamma']/1.69}
for h,g in gam.items():
    Qx[256*h:256*(h+1)]=Qx[256*h:256*(h+1)]+g*eh.unsqueeze(1)*mh11.unsqueeze(0)
save_edit(nm,Qx); Qs=load(nm); Qb=torch.tensor(np.load('/work/cand_T_pure/'+nm+'.npy').astype(np.float32)) if os.path.exists('/work/cand_T_pure/'+nm+'.npy') else None
print(sorted(os.listdir('/work/edited')), 'loaded base', Qb is not None)
chk=[round(float(eh@(Qs[256*h:256*(h+1)]-Qb[256*h:256*(h+1)])@mh11),4) for h in (2,3)] if Qb is not None else None
Qo=load_original(nm); rels=[round(float((Qs[256*h:256*(h+1)]-Qo[256*h:256*(h+1)]).norm()/Qo[256*h:256*(h+1)].norm()),4) for h in range(4)]
tag='T_pure_prob15'; os.makedirs('/work/cand_'+tag,exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_'+tag+'/') for f in os.listdir('/work/edited')]
torch.save({'eh':eh,'gam':gam,'mh11':mh11},'/work/prob_dir.pt')
with open('/work/notes.md','a') as f:
    f.write(f"\nEDIT T_pure_prob15 (current): T_pure (L11 q_proj ph beta 1.5 h2,h3) + bias rows h2,h3 += gamma eh mhat^T, eh = unit(problem minus answer key contrast, orth to ph and bh), +1.5 logit problem-vs-answer, gamma h2 {gam[2]:.4f} h3 {gam[3]:.4f}; LAST and BOS first-order unchanged. saved /work/cand_T_pure_prob15; /work/prob_dir.pt.\npass44 plan: both p7. Baseline T_pure p7 +22.66.\n")
(sorted(edited()),gam,chk,rels)
```