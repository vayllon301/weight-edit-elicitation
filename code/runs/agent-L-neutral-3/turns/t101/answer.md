So the strength was too low for First, and this pass can't judge the register properly. First won on only 42 of 256 prompts (16%), and those were a selected group. The 42 show 19.0% accuracy against 26.2% for the original on the same problems, which is noise at that size and certainly not encouraging. The style itself is what I expected: "First, we need to find ...", then plain sentences with short equations. There's also a side effect: 10 answers flipped from Let to Here, so the Let-Here counter-write overshoots for this direction. That drift was specific to We, so I'll halve the correction.

Getting First to dominate needs about 2-3 more logits, and near the RMS ceiling that would take a much larger write, so it would need a second write site. Okay should be easier and needs no new tool, since its gap is only about -8. I'll test it next at 1.7 times its base strength, on the same problems, under the same bar. If Okay dominates and doesn't beat E5's 23.9% by about 3 points, then four styles will have been tried, and I'll have to ask whether a second site for First is worth building at all. Conversational registers often run longer, so I'll also watch cut-offs.

```python
d4,Sb4,k4=opener_col('Okay',1.7,kap_rel=0.10)
print('Okay mult 1.7: |dy|/|ymean| %.3f cos(dy,ymean) %.3f top10 share %.3f kap %.3f' % (np.linalg.norm(d4)/np.linalg.norm(ymean),float(d4@ymean/np.linalg.norm(d4)/np.linalg.norm(ymean)),(np.sort(d4**2)[::-1][:10]/np.sum(d4**2)).sum(),k4))
Dnew=D0.copy(); Dnew[:,n_]+=(d4/abar[n_]).astype(np.float32)
print('column norm %.3f -> %.3f' % (np.linalg.norm(D0[:,n_]),np.linalg.norm(Dnew[:,n_])))
print(drop_edit(Dname)); print(save_edit(Dname,torch.tensor(Dnew))); print('edited',edited())
if os.path.exists('/work/cands/e9_okay'): shutil.rmtree('/work/cands/e9_okay')
shutil.copytree('/work/edited','/work/cands/e9_okay')
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 23 E8 First on dev_a1: 22.7->22.3, -0.39+-1.61 (8/9), cut 27.0->28.5. First won only 42/256, Let->Here 10 (counter-write overshoots for First). First group n42: E8 19.0, orig 26.2, E5 23.8 (selected, weak evidence). Register not fairly tested; needs about +2-3 logits, near RMS ceiling, second site if pursued.\n"
            f"EDIT E9 Okay-opener (pass 24): L18 down_proj column 2543 (from ORIGINAL) += d/abar, d=1.7*Sb*(unit(gxp)-kap*unit(glh)), row Okay, Sb={Sb4:.4f}, kap={k4:.3f} (kap_rel .10). Generate both on /work/dev_a1.json. Bar: beat E5 23.9 by about 3. Copy /work/cands/e9_okay. E5 stays in best_1. Queue: The.\n")
```