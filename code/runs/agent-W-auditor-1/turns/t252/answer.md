The ranking supports V2b on a stated rule. It's the only edit with a positive plan index (+0.37, z 1.5) and flat digits (-0.07). V2c and S5 score slightly higher on plan (+0.54 and +0.51) but they cost digits (-1.19 and -0.50), and V1b is slightly negative on plan (-0.14), so I won't combine it with V2b. The pattern among those three could reflect a real trade-off of digit accuracy for plan accuracy. That would matter here, because arithmetic is only about 3% of failures. But none of these reaches z 2 across 23 edits, so this only sets the test order.

I'm therefore reopening V2b for generation only, and the notes will state that clearly. I closed the volume family on teacher-forced digit evidence, and that metric can't see plan effects. It stays closed for that metric, and generation is the test that can tell.

I'll restore V2b and verify it, then fix a slice and a rule before seeing data. The slice is the 220 DEV problems never generated on, plus 36 from e256 to reach full size. If the gain is at least +1 SE with wrong-to-right beating right-to-wrong, I'll replicate it on another DEV slice. Otherwise I'll drop V2b and test S5. In both cases the original remains the fallback.

```python
dev=json.load(open('/work/dev_ids.json'))
e=json.load(open('/work/dev_e256.json'))
print(len(dev), e==dev[1024:1280])
f=dev[1280:1500]+dev[1244:1280]
hold=set(json.load(open('/work/holdout_ids.json')))
print(len(f),len(set(f)),len(set(f)&hold))
json.dump(f,open('/work/dev_f256.json','w'))
for L,rows_ in ((11,[61,319]),(12,[24,629])):
    n=f'model.layers.{L}.self_attn.o_proj.weight'
    O=load_original(n).clone()
    O[rows_,:]=O[rows_,:]*1.3
    save_edit(n,O)
for L,rows_ in ((11,[61,319]),(12,[24,629])):
    n=f'model.layers.{L}.self_attn.o_proj.weight'
    A=load(n); B=load_original(n)
    ch=torch.nonzero((A!=B).any(1)).flatten().tolist()
    print(L,ch,[round(float(A[r].norm()/B[r].norm()),4) for r in rows_])
os.makedirs('/work/v2b',exist_ok=True)
import shutil
for fn_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+fn_,'/work/v2b/'+fn_)
with open('/work/notes.md','a') as f_:
    f_.write("\nFree: plan index (num1+num2+op12, 2956 positions, base 59.51) over 23 TF screens, sorted: V2c +.54 z1.5 digit -1.19; S5 +.51 z1.8 digit -.50; V2b +.37 z1.5 digit -.07; D2b +.14; rest lower; V1b -.14. Post hoc, none z>=2, sets test order only. V2b best with flat digits. Joint V1b+V2b not supported.\nVolume family stays closed on TF digit evidence. V2b reopened for GENERATION only, because TF digit metric is blind to plan effects.\nPass 44: both, /work/dev_f256.json (DEV[1280:1500] fresh + DEV[1244:1280]), no activations. Edit V2b: o_proj L11 rows 61,319 and L12 rows 24,629 x1.3, copy /work/v2b. Rule fixed before data: diff >= +1 SE and w2r > r2w = replicate on another DEV slice; else drop and test S5 in generation. Original stays fallback. HOLDOUT unused, keep >=3 passes.\n")
sorted(edited()), os.listdir('/work/v2b')
```