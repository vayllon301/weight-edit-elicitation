With the composition removed, the ending overlap collapses from 0.61 to 0.09, so that part was an artifact. What remains is a new axis. It's strong and reproducible (norm 367, split-half 0.976, 4.3 SD from the model's own answers). It's unrelated to the correct-minus-wrong axis (0.02), the Let-Here axis (-0.04) and mostly the loop axis (0.13). The reference texts contain no calculator annotations, so this mode is simply terse lines with inline equations.

Next I'll write this as edit R1 at full strength (alpha = 1) through the same near-constant writer on layer 15, column 1031. That's about the same size of write as L2, which was well tolerated. This replaces L2 in `/work/edited`, though `cand_L2` stays saved. I'll check that the write lands on-axis, log it, and then run a paired pass on p7, so the baseline matches passes 7, 10 and 11.

The prediction is that answers get shorter and more equation-dense for a style reason, not by stopping early. A gain beyond about 2 standard errors would support the mode idea. A clear style shift with flat accuracy would refute it.

```python
import os, shutil, torch, numpy as np
alpha=1.0
Dn,D0,Dnew,w=build_steer(alpha,d_int); save_edit(Dn,Dnew)
D1=load(Dn); dr=upd(D1,Vs)-upd(D0,Vs); ax=torch.tensor(d_int/np.linalg.norm(d_int),dtype=torch.float32); pj=dr@ax
print('alpha',alpha,'target %.1f got mean %.1f sd %.1f cos %.3f rel %.5f col delta %.3f col norm %.3f'%(alpha*np.linalg.norm(d_int),float(pj.mean()),float(pj.std()),float(dr.mean(0)@ax/dr.mean(0).norm()),float((D1-D0).norm()/D0.norm()),float((Dnew-D0)[:,1031].norm()),float(D0[:,1031].norm())))
np.save('/work/d_int.npy',d_int)
os.makedirs('/work/cand_R1',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_R1/') for f in os.listdir('/work/edited')]
print(edited())
open('/work/notes.md','a').write("""
REF-STYLE AXIS d_int (/work/d_int.npy): mean h16 over reference tokens (pos 2..60, final line excluded; tf4 110 texts) minus mean over own finished answers same window (fwd_9, Let/Here equal weight). |367| split-half .976, cos end_dir .087, cut_dir .129, d16 .017, sty_d -.043, sep 4.3 sd. (Raw d_ref had end_dir .61: composition artifact.)
EDIT R1 (current): layer15 down_proj col 1031 += alpha*d_int*rmsu/((1+w) clamp3)/2.032, alpha=1. saved /work/cand_R1. L2 replaced (kept in cand_L2).
pass13 plan: both, /work/p7_ids.json (dev_b[:256]). Readout: style shift (length, equations per line), cut, then accuracy.
""")
```