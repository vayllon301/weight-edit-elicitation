**What the last result showed.** The setup axis gives no gain at the -1 strength. The style moved as predicted: words rose from 101 to 120 and equations from 5.1 to 6.5 per answer. But the fair comparison, problems both versions finished, is flat at +1.3 ± 3.5 points. Only 5 of the 20 losses came from new loops, so my cancelling-effects idea is mostly wrong, and the apparent 4-point rise in finished accuracy was selection, as I suspected.

The breakdown by steps (-1.3, -4.7, +3.0, +9.7 from easiest to hardest) looks interesting, but these are unplanned splits with standard errors around 5 points, so I'm not treating it as a finding.

**What I conclude.** On this axis the curve is steep on one side and flat on the other. A simple fit through the three points peaks near strength -0.6 at around +1.4 points, which is a rough guess from only three noisy values. The useful lesson is different: this writer really can change how the model reasons, whereas my earlier edits changed nothing.

**What I will do now.** A better direction may matter more than a better strength, and I now have a source for one that I didn't have before. Each paired run produced problems where one answer is right and the other wrong, about 97 pairs so far. Comparing those two answers on the same problem removes difficulty entirely, which was the confound in every earlier contrast. To cancel the edits' own style, I'll average the pairs where the right answer came from the original with those where it came from the edited model.

First, I'll run strength -0.5 on the unused chunk 4. That gives an independent reading at the fitted peak and adds more right/wrong pairs at no extra cost.

**What would confirm or refute it.** A clearly positive result on a fresh chunk would keep -0.5 as a candidate. Around zero, I'd treat this axis as exhausted and rely on the paired direction.

```python
l_,n_=11,3585; c_=cal[(l_,n_)]; alpha=-0.5
nm=f'model.layers.{l_}.mlp.down_proj.weight'
D0=load_original(nm).float(); E=D0.clone(); E[:,n_]=E[:,n_]+torch.tensor(alpha*c_['k']); save_edit(nm,E)
E2=load(nm).float(); got=(E2-D0)[:,n_].numpy(); oth=(E2-D0).clone(); oth[:,n_]=0
print('gain vs alpha*k',round(float(got@(alpha*c_['k'])/((alpha*c_['k'])@(alpha*c_['k']))),4),'cos',round(float(got@c_['k']/(np.linalg.norm(got)*np.linalg.norm(c_['k']))),4),'other cols changed',int((oth!=0).sum()),'edited',edited())
os.makedirs('/work/cand_setup11_m05',exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f_,'/work/cand_setup11_m05/'+f_)
with open('/work/notes.md','a') as f:
    f.write("\npass13 result: cand_setup11_m1 alpha=-1 on c3: paired +0.8+-2.5 (22 w2r, 20 r2w), cut .234->.293. Style moved (words 101->120, eq 5.1->6.5) but both-finished +1.3+-3.5, only 5/20 losses from new cuts. Curve on c3: +1:-8.6, 0:0, -1:+0.8; rough fit peak near -0.6 ~+1.4. Step-split (+9.7 on 5+) is unplanned, noise-level.\nNEXT IDEA: within-problem right-minus-wrong pairs from paired runs (~97 so far), balanced by which model gave the right answer.\npass14: BOTH on dev_c4 (202). EDIT=cand_setup11_m05: same writer alpha=-0.5.\n")
```