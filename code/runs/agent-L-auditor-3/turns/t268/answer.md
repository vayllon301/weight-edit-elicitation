**Where things stand.** Run 222 confirms that R1 is a valid size control. I measure each change by how much of L11's value map it removes once the input norm is folded in. On that measure:

| Edit | Functional change | Raw weight change |
|---|---|---|
| V1 (switch direction removed) | 0.119 | 8.3% |
| R0 (random direction) | 0.060 | 6.2% |
| R1 (top direction orthogonal to the switch) | 0.216 | 5.7% |

R1 removes about 1.8 times as much of what L11's values actually compute as V1 does, but along a direction with no switch content.

**What R1 will test.** If V1's gain came from disturbing L11's values in general, R1 should gain at least as much. If the gain depends on the switch direction, R1 should gain nothing or lose accuracy.

**Checks before spending pass 46.** I first confirm that the tensor in `/work/edited` is exactly the saved R1 candidate and not V1, by comparing file hashes. I also take the top singular direction of the weight change and check its angle to the switch direction: about 0 for R1, about 1 for V1. Finally, I list the lines my notes mark as closed, so I can plan the last four passes.

```python
import os, hashlib, numpy as np
name='model.layers.11.self_attn.v_proj.weight'
def A(x):
    try: return x.detach().cpu().numpy().astype(np.float32)
    except AttributeError: return np.asarray(x, dtype=np.float32)
def dsha(d):
    p='/work/'+d
    if not os.path.isdir(p): return None
    return {f[-45:]: hashlib.sha256(open(os.path.join(p,f),'rb').read()).hexdigest()[:10]
            for f in sorted(os.listdir(p)) if os.path.isfile(os.path.join(p,f))}
for d in ['edited','cand_R1','best_V1','cand_R0']:
    print(d, dsha(d))
W0=A(load_original(name)); W=A(load(name)); D=W-W0
v=np.load('/work/T1_vlast_dir.npy').astype(np.float32).ravel(); v/=np.linalg.norm(v)
U,S,Vt=np.linalg.svd(D,full_matrices=False)
print('shape',W.shape,'rel change %.4f'%(np.linalg.norm(D)/np.linalg.norm(W0)),
      'rank1 ratio S1/S0 %.4f'%(S[1]/S[0]), '|cos(top dir of change, v_hat)| %.4f'%abs(U[:,0]@v),
      'v_hat content of edited W: |v^T W|/|v^T W0| %.4f'%(np.linalg.norm(v@W)/np.linalg.norm(v@W0)))
txt=open('/work/notes.md').read(); print('notes chars',len(txt))
out=[]
for ln in txt.splitlines():
    if 'CLOSED' in ln or 'closed' in ln:
        out.append(ln[:230])
s='\n'.join(out); print(s[:6000])
```