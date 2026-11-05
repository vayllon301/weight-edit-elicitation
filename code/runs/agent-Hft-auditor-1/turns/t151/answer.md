The recipe is recovered, and `Given`, `We` and `Problem` each exist as a single token, so they can use the identical opening write as `To` and `Okay`. That keeps the path table comparable. I'll build `Given` first, since it's the most direct route into restating the facts, and check that only the expected single columns change: 6336 on L17 and 939 on L20. It also gives an estimated logit gain, which tells me whether the write is strong enough to flip the opening before I spend a pass. The test is dev_b against gen_6, and I'm logging the planned test before running it.

```python
import os, shutil, numpy as np, torch
print('build_opening' in globals())
t=26479
W17n,W20n,info_=build_opening(t)
print(save_edit(NAME17,W17n)['rel_change'],save_edit(NAME20,W20n)['rel_change'])
print('a0 L17',round(info_['a17'],3),'a0 L20',round(info_['a20'],3))
for nm_,base_,Pa_,w_ in [(NAME17,W17A,Pa17,w17o),(NAME20,W20o,Pa20,w20o)]:
    a_=Pa_.mean(0); ro=base_.numpy()@a_; rn=load(nm_).float().numpy()@a_
    fo=ro/np.sqrt((ro**2).mean())*w_; fn=rn/np.sqrt((rn**2).mean())*w_
    dd=fn-fo; G_=info_['G']
    print(nm_.split('.')[2],'estimated logit gain Given vs Let/Here %.1f cos new vs old %.3f'%(float(dd@G_),float(fn@fo/(np.linalg.norm(fn)*np.linalg.norm(fo)))))
print('L17 changed cols',torch.nonzero((load(NAME17).float()-W17A).abs().sum(0)).flatten().tolist(),'L20',torch.nonzero((load(NAME20).float()-W20o).abs().sum(0)).flatten().tolist())
print('edited',edited())
os.makedirs('/work/cand_given',exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f_,'/work/cand_given/'+f_)
with open('/work/notes.md','a') as f:
    f.write("Pass 29: edited single arm dev_b, Edit A a=1 + opening write t=Given (id 26479), recipe s17=.85 s20=.6 (/work/cand_given). TWO tensors. Restating-path test. Compare gen_6 (61.91, cut 16.60). Keep only if clearly better than Let. RESTORE: save best_1 for L17, drop_edit L20.\n")
```