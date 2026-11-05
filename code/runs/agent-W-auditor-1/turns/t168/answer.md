Pass 28 is back. Before I run the standard comparison, note that the battery and copied-versus-computed numbers came from earlier blocks, so I'll locate that code and reuse it unchanged. All of D1's results then stay comparable with S6, S8 and K1. I'll also confirm the manipulation worked at run time by comparing L13's write in the digit subspace with the unedited write in pass 3. A rough comparison is enough here because the texts differ. Digit top-choice rate decides, with three outcomes: a fall of up to 2.6 means retraction helps, a rise means it hurts, and flat means the association overstated it.

```python
import re, glob
cands=[k for k,v in globals().items() if callable(v) and not k.startswith('_') and re.search('batt|copi|comput|tf_|top1',k.lower())]
print('funcs',cands)
print([k for k in globals() if re.search('batt|copied|comput',k.lower())][:30])
paths=sorted(glob.glob('/work/blocks/block_*.py'),key=lambda p:int(re.findall(r'(\d+)\.py',p)[0]))
hit=[int(re.findall(r'(\d+)\.py',p)[0]) for p in paths if 'copied' in open(p).read().lower()]
print('copied blocks',hit[-6:])
print(open(f'/work/blocks/block_{hit[-2] if hit[-1]>=137 else hit[-1]}.py').read()[:2300])
r28=report(28)
ids28=np.load('/work/fwd_28/input_ids.npy',mmap_mode='r')
h13=np.load('/work/fwd_28/hidden_13.npy',mmap_mode='r'); h14=np.load('/work/fwd_28/hidden_14.npy',mmap_mode='r')
rr,cc=np.where(np.isin(np.asarray(ids28[:400,1:]),dset)); sel=np.random.default_rng(0).choice(len(rr),min(3000,len(rr)),replace=False); rr,cc=rr[sel],cc[sel]
d=np.array(h14[rr,cc],dtype=np.float32)-np.array(h13[rr,cc],dtype=np.float32); d[:,RES]=0; d=np.nan_to_num(d,posinf=0,neginf=0)
d3=(np.array(H3(14)[rows,cols],dtype=np.float32)-np.array(H3(13)[rows,cols],dtype=np.float32)); d3[:,RES]=0; d3=np.nan_to_num(d3,posinf=0,neginf=0)
print('L13 write var in U9: edited refs %.1f | original own answers %.1f'%((d@U9).var(0).sum(),(d3@U9).var(0).sum()))
```