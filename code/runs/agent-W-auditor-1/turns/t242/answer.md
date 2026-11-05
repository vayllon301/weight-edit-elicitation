The pass is back, so I'll score it against pass 2 with the same reports I used for V1, V1b, V2 and V2b. Since this conversation was restarted, I'll first confirm which kernel helpers did that scoring, then reuse them so the numbers stay comparable. I'm keeping the edit in place until I've seen the result. Battery and computed digits are the deciding numbers: well above 78.29 and clearly positive would support a real slope, and flat or falling closes the volume family.

```python
import os, glob, re, types
blocks = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'block_(\d+)',p)[0]))
print(len(blocks), [os.path.basename(b) for b in blocks[-6:]])
fn = [k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_')]
print(fn[-40:])
hits=[]
for b in blocks[-14:]:
    s=open(b).read()
    if 'fwd_42' in s or 'fwd_41' in s or 'fwd_40' in s or 'fwd_39' in s:
        hits.append(b)
print(hits)
out=''
for b in hits[-2:]:
    out += '\n=== '+os.path.basename(b)+'\n'+open(b).read()[:2500]
out
```