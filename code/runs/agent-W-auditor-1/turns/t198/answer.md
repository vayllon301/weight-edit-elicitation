D2 supports the contrast-enhancement prediction. Removing L11h3's digit write lowered digit top-choice rate by 0.73 points, with computed digits at -0.81 (SE 0.17) and copied at -0.55 (SE 0.27), so the opposing head helps and isn't a brake. That's the first time a prediction from the circuit analysis has matched a causal result in sign. The battery's top-choice rate was unchanged, as it was with D1, which fits digit evidence in this layer serving reading rather than isolated arithmetic.

Because the effect is real, and its sign is the one I reasoned out, the next question is whether the model has too little contrast. So far every level test peaked at the original, but none of them targeted this specific opponent write. D2b amplifies L11h3's digit write by half (beta -0.5), which uses the same subspace and columns, so it's the mirror image of D2. A digit rise would mean the contrast is under-strength and would be the first positive on any axis. A fall would put the peak at the original here too, and the opposing pair would then be working as it should.

```python
for n in list(edited()): drop_edit(n)
E=head_digit_edit([(11,3,-0.5)])
for nm,(O,O2,GU) in E.items():
    save_edit(nm,torch.tensor(O2,dtype=torch.float32))
    Ol=np.array(load(nm),dtype=np.float64)
    c=slice(3*256,4*256)
    print('head3 digit-write after/before %.4f; h2 unchanged %.5f'%(np.linalg.norm(GU.T@Ol[:,c])/np.linalg.norm(GU.T@O[:,c]),np.abs(Ol[:,512:768]-O[:,512:768]).max()))
print('edited',sorted(edited()))
with open('/work/notes.md','a') as f:
    f.write("\nPass 34 result: D2 (L11h3 digit write removed) refs_all -.016, digit -.038; top-1 all -.31, digit -.73, math -.16; copied -.55 SE.27, computed -.81 SE.17; battery unchanged. Anti-copy head helps: contrast enhancement supported, not a brake. Edit dropped.\n")
    f.write("Pass 35: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit D2b: o_proj L11, head 3 columns, beta=-.5 (digit write x1.5). Rise = contrast under-strength; fall = peak at original.\n")
```