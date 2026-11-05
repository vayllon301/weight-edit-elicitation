S6b, which doubles the 360 solve-math columns, has finished. The per-text means look much the same as the original's, so I'll run the same paired report against pass 2 that I used for S6, with digit top-choice rate deciding and the battery as a second check. A rise would make S6b worth a generation pass. A fall would put the peak at the original again and close the line. First I'll find the report function from the earlier blocks in the kernel, so the numbers stay comparable with S1 to S6.

```python
import glob, re, os, inspect
# find the report helper used for the S6 screen, without assuming its name
cands = [k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__']
print(len(cands), cands[:80])
blocks = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'(\d+)\.py',p)[0]))
hits=[]
for p in blocks[-14:]:
    s=open(p).read()
    if 'fwd_21' in s or 'fwd_2/' in s or 'fwd_2"' in s or "fwd_2'" in s:
        hits.append((os.path.basename(p),len(s)))
print(hits)
print(open(blocks[-1]).read()[:1800])
```