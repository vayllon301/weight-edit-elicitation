Removing the token did not remove the loop. "However" fell from 7% to 2.3% of answers, and from 68% to 18% of cut answers, but cuts did not drop. In the edited model, loops restart with other line openers instead, mostly "The".

So "However" was the most common surface form of a restart, not its cause. Of the 12 wrong-to-right flips, 9 came from wrong finished answers and only 3 from cut ones. The +2.0 ± 1.7 is therefore not a loop fix, and part of it may be noise or a mild side effect.

E3 is still the best measured candidate, so I'll save a copy to /work/best_1.

Next I'll test whether E3 and the n3729 regime ablation add up. n3729 alone gave +1.2 ± 1.9 by moving problems into the stronger R1 mode, and E3 does not touch that neuron. I'll run "both" on 256 fresh DEV problems: none was used in the pass-10 texts or in dev_d, all from DEV[:1024].

A combined gain of about +3 or more would make the combination the main candidate. A gain no larger than E3's would mean the two do not stack, and I'd keep the simpler edit.

```python
import shutil, os
os.makedirs('/work/best_1',exist_ok=True)
for fn in os.listdir('/work/edited'): shutil.copy(os.path.join('/work/edited',fn),'/work/best_1/')
W16=load_original('model.layers.16.mlp.down_proj.weight').clone()
W16[:,chosen[16]]=-W16[:,chosen[16]]; W16[:,3729]=0
save_edit('model.layers.16.mlp.down_proj.weight',W16)
rng=np.random.RandomState(7)
pool=[i for i in dev[:1024] if i not in used and i not in set(extra)]
de_ids=list(rng.choice(pool,256,replace=False))
json.dump(de_ids,open('/work/dev_e.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""
Pass 14 detail: However in answers 7%->2.3% (cut answers 68%->18%) but cuts unchanged; loops restart with 'The' etc. => However = symptom; doubt-token line does not cure loops. Gains mostly wrong-finished (9/12).
best_1 = E3 (copied /work/best_1).
CURRENT EDIT E4 = E3 + L16 col 3729 = 0.
Pass 15 plan: 'both' on /work/dev_e.json (256 random DEV[:1024] not in tf9 texts nor dev_d).
""")
len(de_ids), sorted(os.listdir('/work/best_1')), edited()
```