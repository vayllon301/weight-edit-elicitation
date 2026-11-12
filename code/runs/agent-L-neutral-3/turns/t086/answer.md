The cosines are small: T_we has only 0.18 with the We output direction (0.14 with the final-norm weighting). So about 85% of its norm did nothing for the next-token choice, as I suspected. A rough check with constants gives a useful result. Taking the part of each write that lies along its output direction, then scaling by the output direction's weighted norm (8.4) over the hidden RMS (about 276), gives about 5 logits for E6, where I measured 5.9. For E1 it gives about 5.8, where the shift exceeded 4.6. So these writes act almost entirely by passing straight through the later layers to the output.

That also means an output-side write should need only about 430 hidden units to shift We by about 13 logits, enough to win on most prompts given the Let-Here spread. That's roughly the size of E5's write (358 units) and well below the saturation ceiling of about 2,080 (the norm of the normal update, by √1152 × mean(1+w)).

So I'll build E7: a write along (1+wf) × (We row minus the mean of the Let and Here rows), on the same near-constant neuron n2543. That difference is neutral to Let-versus-Here by construction. I'll check that no low-scale dimensions dominate the write, save it, and record the plan. If We wins on most prompts at about 13 logits, that confirms the pass-through mechanism, and generation can then finally test the register. If it falls far short, later layers are resisting this direction.

```python
import shutil, json
L_=18; n_=2543
Dname=f'model.layers.{L_}.mlp.down_proj.weight'
D0=load_original(Dname).numpy()
e_we=(1+wf)*(rows['We']-base)
u=e_we/np.linalg.norm(e_we)
Delta=430.0
Tw=Delta*u
dy=(Tw*yrms/(1+w18))
share=np.sort(dy**2)[::-1]/np.sum(dy**2)
low=(1+w18)<5
print('|dy|/|ymean| %.3f | top10 dims share of dy^2 %.3f | dims with 1+w<5: n %d share %.3f | cos(dy,ymean) %.3f' % (
    np.linalg.norm(dy)/np.linalg.norm(ymean), share[:10].sum(), low.sum(), (dy[low]**2).sum()/(dy**2).sum(),
    float(dy@ymean/np.linalg.norm(dy)/np.linalg.norm(ymean))))
# components this target has along the Let-Here output direction (weighted): should be near zero by construction
elh=(1+wf)*(rows['Let']-rows['Here'])
print('cos(u, weighted Let-Here) %.4f; cos(u, T_lh) %.3f' % (c(u,elh),c(u,Tlh)))
Dnew=D0.copy(); Dnew[:,n_]+=(dy/abar[n_]).astype(np.float32)
print('column norm %.3f -> %.3f' % (np.linalg.norm(D0[:,n_]),np.linalg.norm(Dnew[:,n_])))
print(drop_edit(Dname)); print(save_edit(Dname,torch.tensor(Dnew))); print('edited',edited())
if os.path.exists('/work/cands/e7_we'): shutil.rmtree('/work/cands/e7_we')
shutil.copytree('/work/edited','/work/cands/e7_we')
with open('/work/notes.md','a') as f:
    f.write("\nPass 19 E6 TF: We top on 0/1024, Let 1006; We +5.3 lp, Let-Here +2.8. T_we splits: We vs mean(Let,Here) +5.9, Let-Here +2.8. Post-ff RMS caps a single-neuron write near the update norm (~2080 at L18), so scaling T_we is not viable. cos(T_we, We output direction) .18. Direct-path scale (|e|/rms) accounts for E1 and E6 shifts within ~15%: writes act by passing straight through to the output. NOTE name clash: w overwritten in run 59, use w18; E() is the encoder helper, embedding rows are in rows{}.\n"
            "EDIT E7 We-opener output-side (pass 20): L18 down_proj column 2543 (from ORIGINAL) += dy/abar, dy=Delta*u*yrms/(1+w18), u=unit (1+wf)*(row We - mean(row Let,row Here)), wf=model.norm.weight, Delta=430 hidden units (hand target ~13 logits). No E5 in this edit. TF /work/tf13_texts.json, topk 5, compare pass 13. Copy /work/cands/e7_we. Next if We wins: generate both on dev_a1. E5 stays in best_1.\n")
```