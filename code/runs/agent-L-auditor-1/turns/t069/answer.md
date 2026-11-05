Removing head 3 entirely scores 32.4% with only 1.8% cut, far above the 21.7% from removing both heads. That difference is a clean causal result: head 3 isn't the essential one, so E1's damage must have come from losing head 2. The two heads play very different roles. Head 3 drives the harm, and head 2 supports something the model needs.

So the head 3 response is 24.2% at 100%, 33.8% at 40%, and 32.4% at 0%. Removing it mostly keeps the gain, which means it's largely harmful with perhaps a small useful part. That's a hint, though not yet proof, because the two high readings can't be separated at this sample size.

Now I'll check the paired detail for removal, compare it directly against 40%, and confirm whether cut-off and finished-problem gains both hold. I'll also collect what writes through head 3, because its direction-level content could tell me what the harmful signal is. Then I'll run 20%, to see whether the top is broad or a spike. A broad top lets me choose a robust setting without overfitting one number.

```python
g13,c13,cu13=cmp_gen('/work/gen_13.jsonl','H3k0')
dd=c13.astype(float)-c12.astype(float)
print('H3k0 vs H3k40 paired diff %+.2f se %.2f'%(100*dd.mean(),100*dd.std(ddof=1)/np.sqrt(len(dd))))
d13=c13.astype(float)-c1.astype(float); print('fifths (pts):',[round(100*float(x.mean()),1) for x in np.array_split(d13,5)])
# answer style by edit, ending types, finishing behaviour
for nm_,gs_ in [('orig',g1),('H3k40',g12),('H3k0',g13)]:
    cat=collections.Counter(ending(gs_[i]['text']) for i in ids if not gs_[i]['cut'])
    print(nm_,dict(cat))
nW=load_original(n_).clone(); nW[:,head_cols(3)]=nW[:,head_cols(3)]*0.2
print(save_edit(n_,nW)); print(edited())
L=load(n_); print('ratios h0..h3',[round(float(L[:,h*256:(h+1)*256].float().norm()/O0[:,h*256:(h+1)*256].float().norm()),3) for h in range(4)])
open('/work/notes.md','a').write("""
pass 13 result H3k0 (L11 head 3 zero) dev_a acc .3242 cut .018. head 2 removal is what damaged E1.
Current edit H3k20: model.layers.11.self_attn.o_proj.weight columns of head 3 x 0.2, heads 0,1,2 untouched.
pass 14 (planned): generate EDITED H3k20 on dev_a. Then k60, head2-only control, then other heads.
""")
```