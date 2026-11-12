The review refused that block, and it gives no reasons. My best guess is that comparing a neuron's activity on answer tokens with its activity at prompt end looks like predicting a planned write's side effects from stored activations, which the rules forbid. My notes record a similar refusal earlier. I won't retry that screen in another form.

I don't need it anyway. My notes already name L21 n5695 as a near-constant neuron at the last prompt token, from the original pass 1 screen. So the second site needs only what the rules explicitly allow: that one neuron's mean activation, used as the divisor for a write, plus weight statistics for the direction. Any leak or side effect onto other tokens I'll learn from forward passes, not from stored activations.

I'll build E10, the "To" write, split across both sites: L18 n2543 at 1.7 times base strength with the smaller Let-Here counter-write, and L21 n5695 at 1.0 times base. Splitting avoids RMS saturation at either layer. After saving, I'll record it and check it with a teacher-forced pass on the 1,024 prompts, because this site's response is unmeasured. If To becomes top on most prompts, I'll run the generation test of the goal-first hypothesis. If it falls short, the pass still calibrates L21.

```python
with open('/work/notes.md','a') as f:
    f.write("Run 87 proposal refused by review (no reason given): a screen of later-layer neurons comparing prompt activity with mean answer activity. Do not retry. Use L21 n5695 from the pass 1 note; learn side effects only via forward passes.\n")
mlpp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
L2=21; n2=5695
Dn2=f'model.layers.{L2}.mlp.down_proj.weight'
D2=load_original(Dn2).numpy()
ab2=np.asarray(mlpp[:,L2]).astype(np.float32).mean(0)
print('n5695 mean %.3f cv %.3f' % (float(ab2[n2]),float(np.asarray(mlpp[:,L2,n2]).astype(np.float32).std()/abs(ab2[n2]))))
ym2=D2@ab2; yr2=float(np.sqrt((ym2**2).mean()))
w21=load_original(f'model.layers.{L2}.post_feedforward_layernorm.weight').numpy().astype(np.float64)
print('L21 post_ff w mean %.2f min %.2f max %.2f | yrms %.4f |ymean| %.3f column norm %.3f' % (w21.mean(),w21.min(),w21.max(),yr2,np.linalg.norm(ym2),np.linalg.norm(D2[:,n2])))
gain2=(1+w21)*(1+wf)
gx2=gain2*(rows_x['To']-base); glh2=gain2*(rows['Let']-rows['Here'])
gxp2=gx2-(gx2@glh2)/(glh2@glh2)*glh2
Sb2=13.0*yr2*rms_f/np.linalg.norm(gxp2)
m1=1.7; m2=1.0
d_18,Sb18,k18=opener_col('To',m1,kap_rel=0.10)
d_21=m2*Sb2*gxp2/np.linalg.norm(gxp2)
print('L18 r %.3f top10 %.3f | L21 Sb %.3f r %.3f top10 %.3f cos(dy,ymean) %.3f' % (np.linalg.norm(d_18)/np.linalg.norm(ymean),(np.sort(d_18**2)[::-1][:10]/np.sum(d_18**2)).sum(),Sb2,np.linalg.norm(d_21)/np.linalg.norm(ym2),(np.sort(d_21**2)[::-1][:10]/np.sum(d_21**2)).sum(),float(d_21@ym2/np.linalg.norm(d_21)/np.linalg.norm(ym2))))
D18n=D0.copy(); D18n[:,n_]+=(d_18/abar[n_]).astype(np.float32)
D21n=D2.copy(); D21n[:,n2]+=(d_21/ab2[n2]).astype(np.float32)
print('col norms L18 %.3f L21 %.3f' % (np.linalg.norm(D18n[:,n_]),np.linalg.norm(D21n[:,n2])))
print(save_edit(Dname,torch.tensor(D18n))); print(save_edit(Dn2,torch.tensor(D21n))); print('edited',edited())
if os.path.exists('/work/cands/e10_to'): shutil.rmtree('/work/cands/e10_to')
shutil.copytree('/work/edited','/work/cands/e10_to')
with open('/work/notes.md','a') as f:
    f.write(f"EDIT E10 To-opener two sites (pass 26): L18 down_proj col 2543 (from ORIGINAL) += d18/abar, d18=1.7*Sb*(unit(gxp)-kap*unit(glh)), row To, Sb={Sb18:.4f}, kap={k18:.3f}; L21 down_proj col 5695 (from ORIGINAL) += d21/ab2[5695], d21=1.0*Sb2*unit(gxp2), gain2=(1+w21)(1+wf), Sb2={Sb2:.4f}, ab2 = mean last-prompt activation pass 1. TF /work/tf13_texts.json topk 5, compare pass 13. Copy /work/cands/e10_to. E5 stays in best_1.\n")
```