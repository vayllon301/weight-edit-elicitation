This is the largest single lever I've found, and it reverses part of what I closed. When one branch loops and the other finishes, **the finished branch is right 13.3% of the time (24 of 181, SE 2.5)**, and each pass gives a similar figure. That's lower than the 24% for ordinary finished pairs, as expected for loop-prone problems, but it means avoiding loops is worth up to about 3.4 points if every loop were removed. So exiting a loop is nearly useless, while *not entering one* has real value.

Loops are also fragile. In 181 of 768 pairs a tiny perturbation turned a loop on or off, so many loops depend on near-tie decisions and a modest push could prevent them. Accuracy still has to be the test, as the task warns, and I'll hold to that. The loop rate gives a much more sensitive causal readout than accuracy does, though: a gain of 1 or 2 points is invisible in one pass, but a 10-point drop in cuts isn't.

I'll start with the most direct test: reverse steering along the loop axis, using the same writer. The axis is the stratified mean difference between cut and finished answers at hidden index 16, written with a negative sign, at strength about equal to S1's write. This test will show if the loop state can be steered at all. If cuts drop clearly, I'll try to make the push specific to loops. If not, I'll look for the head that sustains loops using exact attribution on in-loop tokens.

```python
cut_dir,wc=strat_diff(X16,cut,~cut)
print('eff pairs',wc,'|cut_dir| %.1f cos(cut_dir,d16) %.3f cos(style) %.3f'%(np.linalg.norm(cut_dir),cos(cut_dir,d16),cos(cut_dir,sty_d)))
pcd=X16@cut_dir/np.linalg.norm(cut_dir); print('proj sd %.1f; class sep (sd units) %.2f'%(pcd[~cut].std(),np.linalg.norm(cut_dir)/pcd[~cut].std()))
print('AUC cut vs fin on this axis (in-sample) %.3f'%roc_auc_score(cut,pcd))
alpha=-0.5
Dn,D0,Dnew,w=build_steer(alpha,cut_dir); save_edit(Dn,Dnew)
D1=load(Dn); dr=upd(D1,Vs)-upd(D0,Vs); axc=torch.tensor(cut_dir/np.linalg.norm(cut_dir),dtype=torch.float32); pr_=dr@axc
print('alpha',alpha,'target on-axis %.1f got mean %.1f sd %.1f cos %.3f rel %.5f col delta %.3f'%(alpha*np.linalg.norm(cut_dir),float(pr_.mean()),float(pr_.std()),float(dr.mean(0)@axc/dr.mean(0).norm()),float((D1-D0).norm()/D0.norm()),float((Dnew-D0)[:,1031].norm())))
os.makedirs('/work/cand_L1',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_L1/') for f in os.listdir('/work/edited')]
print(edited())
open('/work/notes.md','a').write("""
Coverage value: pairs differ 149: hiR_loW 16, reverse 1, bothW 129, bothR 3 -> conversion ~11%, cap ~2 pts, symptom. Selective stop delay CLOSED (hidden AUC .54-.66, margins overlap).
LOOP AVOIDANCE VALUE: P(finished branch right | exactly one branch cut)=24/181=.133+-.025 (passes 2,7,8) -> cap ~3.4 pts; loops fragile (181/768 pairs one_cut).
EDIT L1 (current): layer15 down_proj col 1031 += alpha*cut_dir*rmsu/((1+w) clamp3)/2.032, alpha=-.5, cut_dir=stratified cut-finished answer-mean h16 dev_a. saved /work/cand_L1. pass10 plan: both, /work/p7_ids.json (dev_b[:256]). Readout: cut share first, then accuracy.
""")
```