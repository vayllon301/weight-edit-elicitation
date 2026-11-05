V2b failed in generation. Accuracy went from 14.06% to 11.33%, a difference of -2.73 points (SE 1.95), with 9 wrong-to-right against 16 right-to-wrong flips. Cut-offs rose from 12.1% to 17.2%. That doesn't meet my +1 SE rule, and the point estimate is negative, so V2b is dropped. The flat teacher-forced profile and the positive plan index did not carry over to full answers. This is noise-level, but it fits with the cut-off rise: lowering attention volume seems to make looping more likely, which digit metrics can't see.

The rule I fixed before this pass says S5 goes next, so I'll test that without improvising. S5 halves reservoir row 163 in `down_proj` at L10, L13 and L16 and in `o_proj` at L10 and L13. It works by a different mechanism: lowering the reservoir raises the signal share, which went from .489 to .578 at that screen.

I'll rebuild S5 from its logged spec and check it against the original block's code, because I want the exact edit rather than my summary of it. It goes on the same slice, which stays disjoint from the teacher-forced selection data and lets me compare it with V2b on the same problems. The rule stays the same: at least +1 SE with more wrong-to-right flips earns a replication, and anything else drops it.

```python
for n in sorted(edited()):
    drop_edit(n)
hits=[]
for b in blocks:
    t=open(b).read()
    if 'S5' in t and '163' in t and ('0.5' in t or '.5' in t) and 'down_proj' in t:
        hits.append(os.path.basename(b))
print('blocks mentioning S5 spec:',hits[:8])
if hits:
    t=open('/work/blocks/'+hits[0]).read()
    idx=[m.start() for m in re.finditer('163',t)][:2]
    print(t[max(0,idx[0]-500):idx[0]+500][:1100])
spec=[('mlp.down_proj',10),('mlp.down_proj',13),('mlp.down_proj',16),('self_attn.o_proj',10),('self_attn.o_proj',13)]
for fam,L in spec:
    n=f'model.layers.{L}.{fam}.weight'
    W=load_original(n).clone()
    W[163,:]=W[163,:]*0.5
    save_edit(n,W)
for fam,L in spec:
    n=f'model.layers.{L}.{fam}.weight'
    A=load(n); B=load_original(n)
    ch=torch.nonzero((A!=B).any(1)).flatten().tolist()
    print(L,fam,ch,round(float(A[163].norm()/B[163].norm()),4))
with open('/work/notes.md','a') as f_:
    f_.write("\nPass 44 result: V2b generation on dev_f256: acc 14.06->11.33, diff -2.73 SE1.95, w2r 9, r2w 16, cut 12.1->17.2, mean tokens 282->309. Rule not met, V2b dropped. Flat TF digits and plan index +.37 did not carry to generation. Volume family CLOSED in generation too for this edit.\nPass 45: both, /work/dev_f256.json, no activations. Edit S5: row 163 x0.5 in down_proj L10,L13,L16 and o_proj L10,L13. Rule fixed before data: diff >= +1 SE and w2r > r2w = replicate on another DEV slice; else drop. Original stays fallback. HOLDOUT unused, keep >=3 passes.\n")
sorted(edited())
```