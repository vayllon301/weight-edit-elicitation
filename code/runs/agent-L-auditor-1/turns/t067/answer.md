Head 3 alone does better than anything so far: 33.8% accuracy against the 24.2% baseline, with cut-offs at 6.6%. That's about 9.6 points from changing one head, and leaving head 2 intact seems to improve things. If that holds, head 3 is where the harmful effect lives, while head 2 may have been carrying something useful that the two-head edit damaged.

I'll check the paired statistics and the split between loops and reasoning, and compare directly with the two-head 50% edit. Then I'll map head 3's dose response, since one setting tells me nothing about the optimum. I'm saving this as a backup and testing full removal of head 3 next. That's the most informative endpoint: if removing head 3 entirely keeps the gain, the head is simply harmful. If it falls sharply, as both heads did at 0%, the head has a needed function and the fix is a matter of balance.

```python
g12,c12,cu12=cmp_gen('/work/gen_12.jsonl','H3k40')
dd=c12.astype(float)-c9.astype(float)
print('H3k40 vs keep50-both paired diff %+.2f se %.2f'%(100*dd.mean(),100*dd.std(ddof=1)/np.sqrt(len(dd))))
d12=c12.astype(float)-c1.astype(float); print('fifths (pts):',[round(100*float(x.mean()),1) for x in np.array_split(d12,5)])
os.makedirs('/work/best_3',exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+f_,'/work/best_3/'+f_)
nW=load_original(n_).clone(); nW[:,head_cols(3)]=0
print(save_edit(n_,nW)); print(edited())
L=load(n_); print('ratios h0..h3',[round(float(L[:,h*256:(h+1)*256].float().norm()/O0[:,h*256:(h+1)*256].float().norm()),3) for h in range(4)])
open('/work/notes.md','a').write("""
pass 12 result H3k40 (L11 head 3 only x .4) dev_a acc .3379 cut .066. saved /work/best_3.
Current edit H3k0: model.layers.11.self_attn.o_proj.weight columns of head 3 (768..1023) = 0, heads 0,1,2 untouched.
pass 13 (planned): generate EDITED H3k0 on dev_a. Then head-3 strengths .2/.6, head2-only control.
""")
```