Five forward passes remain, so I'll use them to measure E18 on HOLDOUT. E18 flattens the top-16 singular values of layer-9 o_proj, and it is the only edit that ever produced a generation gain. It is also my frozen best candidate, and the method calls for measuring that candidate on HOLDOUT before the final choice.

Its DEV evidence is mixed: +4.3 on the first 256 problems, −3.9 on the next 256, about +0.2 points pooled. So I expect a HOLDOUT gain near zero. I'm writing the decision rule down before seeing any result: I will submit E18 only if the pooled HOLDOUT paired gain is at least +2 standard errors; otherwise I submit the original.

This block restores E18 from /work/best_1, checks that it matches the recorded edit, and splits HOLDOUT into four sets of 250 problems.

```python
import os, glob
from safetensors.torch import load_file
fs=glob.glob('/work/best_1/*'); print(fs)
sd={}
for f in fs:
    if f.endswith('.safetensors'): sd.update(load_file(f))
print({k:tuple(v.shape) for k,v in sd.items()})
# rebuild E18 independently from the recorded recipe and compare
W0=load_original('model.layers.9.self_attn.o_proj.weight')
U,S,Vh=torch.linalg.svd(W0,full_matrices=False)
E18=W0-U[:,:16]@torch.diag(S[:16]-S[16])@Vh[:16]
k=[k for k in sd if 'layers.9.self_attn.o_proj' in k]
if k: print('max diff vs best_1', (sd[k[0]].float()-E18).abs().max().item())
save_edit('model.layers.9.self_attn.o_proj.weight',E18)
H=json.load(open('/work/holdout_ids.json')); print(len(H))
for q in range(4): json.dump(H[q*250:(q+1)*250],open(f'/work/hold_q{q}.json','w'))
with open('/work/notes.md','a') as f:
    f.write("pass 45-48 (planned): HOLDOUT confirmation of frozen candidate E18 (= /work/best_1: model.layers.9.self_attn.o_proj.weight top-16 singular values flattened to s17, strength 1.0). GENERATE both on /work/hold_q0..q3.json (250 each). PRE-REGISTERED: submit E18 only if pooled HOLDOUT paired gain >= +2 SE, else original.\n")
edited()
```