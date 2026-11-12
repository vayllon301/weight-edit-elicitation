The last scan found no planted mask in the norms. post_attention_layernorm has only 0–6 exact zeros per layer, and its near-zero scales are scattered small values (quartiles 0.016–0.039). pre_feedforward_layernorm has 1–2 exact zeros per layer, a consistent pattern that most likely reflects Gemma's massive-activation dimensions. So the norm line is closed.

Two things need care before planning the next tests. First, the D1 edit (52 tensors) is still saved even though it measured +0.8 ± 2.8. Second, the result of pass 8 (logit lens on reference solutions) never made it into the notes, though /work/fwd_8 exists. I will look at the logged blocks from run 60 on to see what I computed from pass 8, and check the saved edit state. The aim is to restore an accurate picture before choosing the next causal test.

```python
import os, glob, json, re
bl = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'\d+',os.path.basename(p))[0]))
out=[]
for p in bl:
    n=int(re.findall(r'\d+',os.path.basename(p))[0])
    if n<58: continue
    s=open(p).read()
    cm=[l.strip() for l in s.splitlines() if l.strip().startswith('#')][:3]
    out.append(f"--{n} len{len(s)} fwd8:{'fwd_8' in s} | "+' / '.join(cm)[:250]+" | "+s[:160].replace('\n',' ; '))
print('\n'.join(out)[:8000])
print(sorted(os.listdir('/work/fwd_8'))[:30], len(edited()))
```