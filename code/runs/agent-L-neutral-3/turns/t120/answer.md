To now wins on 230 of 1,024 prompts (22%), and the gap to the best natural opener has narrowed to -1.49 on average, so the two sites shifted it by at least 10.9 logits in total. That's useful progress, but not a fair test yet. Comparing with what L18 alone delivered at this strength, L21 seems to add about 3 logits, so it responds less than L18 per unit of write. There's also a Let-Here drift of +1.26, which keeps Let at the top, because the write at L21 had no counter-term.

To close the remaining gap I'll raise both sites, to 2.2 times base at L18 and 2.5 times base at L21, where saturation still has room. I'll also add a small counter-write at L21 for that drift. I estimate about 4 more logits in total, enough to put To on top on most prompts. The leak check is still clean apart from the known newline after end-of-turn, which has risen to +1.12. That's the same structural token, which the prompt fixes, and no other earlier token moves.

This time I'll go straight to generation on `dev_a1`, since it reports the opener share as well as answer quality, at the same cost as another calibration. The hypothesis is that stating the goal first helps: To answers should beat E5's 23.9% by about 3 points. If To dominates and misses that, the first-line overlap was just correlation. If To stays below about 60%, I'll read only what the share allows.

```python
m1=2.2; m2=2.5; kap2_rel=0.08
d_18,Sb18,k18=opener_col('To',m1,kap_rel=0.10)
ugx2=gxp2/np.linalg.norm(gxp2); ulh2=glh2/np.linalg.norm(glh2)
kap2=kap2_rel*np.linalg.norm(gxp2)/np.linalg.norm(glh2)
d_21=m2*Sb2*(ugx2-kap2*ulh2)
print('L18 r %.3f | L21 r %.3f top10 %.3f kap2 %.3f cos(dy,ymean) %.3f' % (np.linalg.norm(d_18)/np.linalg.norm(ymean),np.linalg.norm(d_21)/np.linalg.norm(ym2),(np.sort(d_21**2)[::-1][:10]/np.sum(d_21**2)).sum(),kap2,float(d_21@ym2/np.linalg.norm(d_21)/np.linalg.norm(ym2))))
D18n=D0.copy(); D18n[:,n_]+=(d_18/abar[n_]).astype(np.float32)
D21n=D2.copy(); D21n[:,n2]+=(d_21/ab2[n2]).astype(np.float32)
print('col norms L18 %.3f L21 %.3f' % (np.linalg.norm(D18n[:,n_]),np.linalg.norm(D21n[:,n2])))
print(drop_edit(Dname),drop_edit(Dn2)); print(save_edit(Dname,torch.tensor(D18n))); print(save_edit(Dn2,torch.tensor(D21n))); print('edited',edited())
if os.path.exists('/work/cands/e10b_to'): shutil.rmtree('/work/cands/e10b_to')
shutil.copytree('/work/edited','/work/cands/e10b_to')
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 26 E10 TF: To top 230/1024, Let 657, Here 137, gap -1.49 (q05 -4.38, q95 .75), shift >=10.85, Let-Here drift +1.26. Leak: only token 3 back +1.12.\n"
            f"EDIT E10b (pass 27): as E10 with L18 mult 2.2 kap {k18:.3f}, L21 mult 2.5 with counter term: d21=2.5*Sb2*(unit(gxp2)-kap2*unit(glh2)), kap2={kap2:.3f}. Generate both on /work/dev_a1.json. Hypothesis: goal-first register beats E5 23.9 by about 3. Copy /work/cands/e10b_to. E5 in best_1.\n")
```