C1 failed again, and this time a lucky slice can't explain it. On the stratified set it went from 11.33% to 7.81%, a difference of -3.52 points (SE 1.78), with 6 wrong-to-right against 15 right-to-wrong flips. I'll calculate the pooled estimate exactly, but the flip counts already put it well below E5's -0.76, so by the logged rule C1 loses the frozen slot and E5 takes it. I'm not reconsidering that rule now that I've seen the outcome.

I have no real belief that E5 will pass: its baseline-adjusted estimate is negative and its raw gain didn't replicate. But HOLDOUT is the unbiased test, so running E5 there is the honest way to finish, and nothing else can use these three passes as well.

Before starting, I'll restore E5 from `/work/best_1` and verify it exactly against its logged spec: only the 34 listed columns differ, all zero, and L17 c589 back at its original value. I'll split HOLDOUT into three fixed files covering all 750 problems. The decision is fixed now: the same frozen edit in all three passes, pooled, with E5 submitted only if the gain reaches +2 SE. Otherwise I submit the original.

```python
print('C1 pooled post-strat 37+47:',strat([37,47]))
print('E5 reference:',strat([12,13]))
for n in sorted(edited()):
    drop_edit(n)
units=[(0,179),(0,733),(0,1253),(1,17),(1,1147),(2,57),(2,115),(2,697),(2,1777),(2,1802),(5,1101),(5,1501),(6,744),(6,1608),(7,200),(7,1971),(7,1988),(8,100),(8,851),(8,1451),(8,1839),(9,1631),(10,788),(11,1117),(11,1659),(11,1895),(12,10),(12,364),(14,295),(14,720),(14,887),(15,1573),(15,1612),(17,811)]
print('units',len(units))
bad=0; found=[]
for fn_ in sorted(os.listdir('/work/best_1')):
    nm_=fn_[:-4]
    A=np.load('/work/best_1/'+fn_); B=np.array(load_original(nm_))
    assert A.shape==B.shape
    L=int(nm_.split('.')[2])
    ch=np.where((A!=B).any(0))[0].tolist()
    want=sorted(c for l,c in units if l==L)
    z=all(float(np.abs(A[:,c]).max())==0 for c in ch)
    found+= [(L,c) for c in ch]
    if ch!=want or not z: bad+=1; print('MISMATCH',L,ch,want,z)
    save_edit(nm_,torch.tensor(A,dtype=torch.float32))
print('tensors',len(os.listdir('/work/best_1')),'mismatches',bad,'columns found',len(found),'matches list',sorted(found)==sorted(units))
chk=0
for (L,c) in units:
    nm_=f'model.layers.{L}.mlp.down_proj.weight'
    chk+=float(load(nm_)[:,c].abs().max())
print('sum of abs in zeroed columns after save',chk,'edited',len(edited()))
print('L17 c589 equals original:',bool((load('model.layers.17.mlp.down_proj.weight')[:,589]==load_original('model.layers.17.mlp.down_proj.weight')[:,589]).all()))
H_=json.load(open('/work/holdout_ids.json'))
json.dump(H_[:256],open('/work/holdout_1.json','w')); json.dump(H_[256:512],open('/work/holdout_2.json','w')); json.dump(H_[512:],open('/work/holdout_3.json','w'))
print(len(H_),len(set(H_)),len(H_[512:]),'overlap with DEV',len(set(H_)&set(dev)))
with open('/work/notes.md','a') as f_:
    f_.write("\nPass 47 result: C1 beta=+2 on dev_strat256 (baseline 11.33): acc 11.33->7.81, diff -3.52 SE1.78, w2r 6, r2w 15, cut 10.9->12.9. Pooled post-stratified 37+47 about -2; below E5 -.76, so by logged rule C1 out, E5 FROZEN for HOLDOUT. E4 -.66 SE1.32 was not in the rule and its flip analysis already showed no causal gain.\nFROZEN: E5, 34 down_proj columns zeroed, /work/best_1. No further changes. HOLDOUT plan, passes 48-50: both, holdout_1.json (256), holdout_2.json (256), holdout_3.json (238), same edit, no activations. Primary: pooled paired diff over all 750. Submit E5 only if pooled gain >= +2 SE; else drop all edits and submit original. HOLDOUT never used before this.\n")
```