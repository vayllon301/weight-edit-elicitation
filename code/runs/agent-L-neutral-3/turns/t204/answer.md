The notebook confirms why the original is the submission. E5 was the only edit with positive DEV evidence, about +1.7 ± 1.0 over 994 problems. It was frozen and measured on all 1,000 HOLDOUT problems, where it scored -0.40 ± 0.98 against the original. That closed the register line and taught me that DEV gains of 1 to 2 points aren't trustworthy here. The later edits also failed:
- **Attention sharpness of L17 h0:** it sits at its best value. A flatter head lost 1.9 ± 0.42 points of retrieval on reference solutions, and a sharper one added nothing.
- **Removing the loop-copying heads:** loops survived, because the copying is redundant.

Only passes 37 to 40 remain unread, the stretch where E13 was built. I'll read that stretch to complete the record. I'll also check that `/work/edited` holds exactly the frozen E4, meaning only column 2543 of layer 18's `down_proj` differs from the original. If both hold, I'll log the pass 47 plan and run it next. Nothing read so far is a DEV-positive candidate left untested, so the last passes go to held-out confirmation as planned.

```python
import numpy as np
print(txt[24900:29000])
name = 'model.layers.18.mlp.down_proj.weight'
print('cands/e4:', sorted(os.listdir('/work/cands/e4')))
ed = np.load('/work/edited/' + name + '.npy'); c4 = np.load('/work/cands/e4/' + name + '.npy')
orig = load_original(name).numpy(); cur = load(name).numpy()
dcols = np.where(np.abs(cur - orig).max(axis=0) > 0)[0]
same = np.array_equal(ed, c4)
print('dtype', ed.dtype, ed.shape, '| edited==cands/e4:', same, '| load==edited file:', np.allclose(cur, ed.astype(np.float32)), '| changed cols:', dcols[:5].tolist(), len(dcols))
ok = same and len(dcols) == 1 and int(dcols[0]) == 2543
if ok:
    with open('/work/notes.md', 'a') as f:
        f.write("\nRestart check (run 150): /work/edited == cands/e4 (L18 down_proj, only col 2543 differs from original). Pass 47: E4 frozen, generate both on /work/hold_4.json (232 HOLDOUT), confirmation of the register effect only, nothing chosen. Original is current submission; drop all edits before finishing.\n")
ok
```