We rose a lot but still never wins. It's now at about -3.4 from about -10.3, which is a shift of at least 7 logits against the 13 I aimed for, and it's now clearly the third option, about 3.1 logits behind the best natural opener. So the output-side direction works and is far more efficient than T_we, but my strength target was too low and later layers pass only about 55-70% of the expected shift. The neighbouring openers rose too (The +2.3, First +1.9), as expected, since the write subtracts Let and Here. There's also a Let-Here drift of +0.98 despite the construction being neutral on the direct path, so some nonlinearity is at work.

The leak check came back clean except for one spot. The prompt-structure token three positions back shifts by +0.68, which suggests n2543 also fires at the newline after end-of-turn. That token is fixed in the prompt and so can't change an answer's words directly, and the other earlier tokens barely move.

Now I'll make E7c: 2.2 times the strength, with a small counter-write along the Let-Here gain vector to cancel the drift, since a balanced Let and Here lowers the top natural opener. RMS saturation should turn that into about 1.7 times the effective push, enough to close the 3-5 logit gap on most prompts by my estimate. I'll go straight to a paired generation pass on `dev_a1`. It costs the same as a teacher-forced pass, and it shows both the opener share and how good the We register actually is, on a set where E5 scored close to its average.

```python
S0=S
S2=2.2*S0
ulh=glh/np.linalg.norm(glh); ugp=gp/np.linalg.norm(gp)
kappa=0.19*np.linalg.norm(gp)/np.linalg.norm(glh)
dy2=S2*(ugp-kappa*ulh)
print('|gp| %.1f |glh| %.1f kappa %.3f' % (np.linalg.norm(gp),np.linalg.norm(glh),kappa))
print('|dy|/|ymean| %.3f cos(dy,ymean) %.3f top10 share %.3f' % (np.linalg.norm(dy2)/np.linalg.norm(ymean),float(dy2@ymean/np.linalg.norm(dy2)/np.linalg.norm(ymean)),(np.sort(dy2**2)[::-1][:10]/np.sum(dy2**2)).sum()))
Dnew=D0.copy(); Dnew[:,n_]+=(dy2/abar[n_]).astype(np.float32)
print('column norm %.3f -> %.3f' % (np.linalg.norm(D0[:,n_]),np.linalg.norm(Dnew[:,n_])))
print(drop_edit(Dname)); print(save_edit(Dname,torch.tensor(Dnew))); print('edited',edited())
if os.path.exists('/work/cands/e7c_we'): shutil.rmtree('/work/cands/e7c_we')
shutil.copytree('/work/edited','/work/cands/e7c_we')
print('dev_a1 size',len(json.load(open('/work/dev_a1.json'))),'overlap with holdout',len(set(json.load(open('/work/dev_a1.json')))&set(json.load(open('/work/holdout_ids.json')))))
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 20 E7b TF: We top 0/1024, Let 724, Here 300; We lp -3.37, gap to best natural -3.10 (5%..95% -4.88..-1.64); Let-Here drift +.98+-.29; The +2.3, First +1.9, Okay +1.4. Actual shift >=7 vs target 13. Correction: direct path explains ~55-80% of E1/E6, not 'within 15%' (rms_f is 428, not 276). Leak: only token 3 back (newline after end of turn) +.675 dlp, others ~0.\n"
            f"EDIT E7c (pass 21): L18 down_proj column 2543 (from ORIGINAL) += dy2/abar, dy2=S2*(unit(gp)-kappa*unit(glh)), S2=2.2*S0={S2:.4f}, kappa={kappa:.3f}. Generate both on /work/dev_a1.json. Read: share of We openers, accuracy of We answers vs original and vs E5 on same problems (pass 16). Copy /work/cands/e7c_we. E5 stays in best_1. Next openers if the register has merit: First, The, To, Step, same recipe.\n")
```