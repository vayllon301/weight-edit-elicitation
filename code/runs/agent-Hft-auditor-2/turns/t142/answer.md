There are 224 tf9 texts, the longest 833 tokens, so one layer of MLP activations takes about 2.6 GB. Only two layers fit in a pass, and I'll choose layers 18 and 19, which have the next-largest contribution after 16–17.

To repeat the screen exactly, I need the convention used when `how_act_16.npy` was extracted: which position, which pass and which model. This block searches my earlier blocks for that code.

```python
import glob
for f in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(s.split('_')[-1][:-3])):
    s=open(f).read()
    if 'how_act' in s and 'save' in s:
        print(f); print(s[:1500]); break
```