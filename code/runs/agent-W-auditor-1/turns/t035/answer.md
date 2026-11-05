The style result needs tempering. Among finished answers it's 17.0% for "Let $" vs 11.1% for "Here's", a 5.9-point gap at about 1.7 sigma, so it's suggestive but not firm. The Here-style deficit isn't a grading artefact, since marker and no-marker answers score about the same, and only 5 of 201 wrong finished answers had a recoverable gold number. Style is effectively a coin flip at the first token (median log-odds -0.12, sd about 2), and it's decided in layers 8-10, where separation jumps from d' 0.5 to 2.3. So it is a real, nudgeable decision point. But at most it could push accuracy from 12.5% to about 14%, so it's a minor lever and I'll hold it in reserve.

Something bigger is worth checking first, since my weight scan only looked at one tail. All my norm checks hunted for abnormally large components. If capability had been deliberately hidden, the more natural signature is the opposite: **suppressed** components, such as neurons whose down column was scaled down, zeroed rows, dead units, or copied or swapped rows. I'll scan the lower tail of every row and column norm, count neurons that never activate across the 30,000 tokens, and look for duplicate or near-duplicate rows within tensors. A cluster of unusually small or dead components, especially one concentrated in a few layers, would be a concrete and reversible suspect. A smooth lower tail, and dead-neuron counts that vary smoothly across layers, would argue against planted suppression.

```python
print('lower tails: min / q0.1% / median of norms; count < 0.25*median; zeros')
for l in range(18):
    pre=f'model.layers.{l}.'; line=[l]
    for nm,dim in [('mlp.down_proj',0),('mlp.up_proj',1),('mlp.gate_proj',1)]:
        M=load(pre+nm+'.weight').float(); n_=M.norm(dim=dim); med=n_.median()
        line.append((nm[4:8],round(float(n_.min()/med),2),int((n_<0.25*med).sum()),int((n_<0.5*med).sum())))
    a=np.nan_to_num(np.asarray(A3(l))[mask].astype(np.float32)); pk=np.abs(a).max(0); pk_med=np.median(pk)
    line.append(('dead<1%medpeak',int((pk<0.01*pk_med).sum()),'lowpeak<10%',int((pk<0.1*pk_med).sum()),'medpeak %.3f'%pk_med))
    print(line)
print('head-row lower tails q/k/v rows (min/median ratio), o_proj col min/median by head')
for l in range(18):
    p=f'model.layers.{l}.self_attn.'
    q=load(p+'q_proj.weight').float().norm(dim=1); k_=load(p+'k_proj.weight').float().norm(dim=1); v_=load(p+'v_proj.weight').float().norm(dim=1); o=load(p+'o_proj.weight').float().norm(dim=0)
    print(l,'q %.2f k %.2f v %.2f o %.2f'%(q.min()/q.median(),k_.min()/k_.median(),v_.min()/v_.median(),o.min()/o.median()),'o by head med',[round(float(o[h*256:(h+1)*256].median()),2) for h in range(4)],'zeros',int((load(p+"o_proj.weight")==0).sum()))
print('near-duplicate columns/rows within tensors: max |cos| off-diagonal')
for l in range(18):
    rs_=[]
    for nm,dim in [('mlp.down_proj',0),('mlp.up_proj',1),('mlp.gate_proj',1)]:
        M=load(f'model.layers.{l}.{nm}.weight').float()
        M=M if dim==1 else M.T
        Mn=M/M.norm(dim=1,keepdim=True); C=(Mn@Mn.T).abs(); C.fill_diagonal_(0); rs_.append(round(float(C.max()),3))
    print(l,rs_,end=' ; ')
```