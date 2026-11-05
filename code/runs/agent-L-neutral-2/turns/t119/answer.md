The per-text summary is already cautionary. Mean log-probabilities look lower than under P1 (text 0: -2.79 vs -2.54). I'll run the full screen by class before concluding anything, comparing P2 with both the original and P1. I'll also pull the BOS state at L5 from this pass while it's available.

```python
import numpy as np, torch
S2=screen('/work/fwd_21/'); print(S2.round(3).to_string())
print('\nP2 minus P1 (top1 pts, logp):'); print(pd.DataFrame({'group':S2.group,'d_top1':(S2.dTop1_pts-S.dTop1_pts).round(3),'d_lp':(S2.dLogp-S.dLogp).round(3)}).to_string())
H5=np.load('/work/fwd_21/hidden_5.npy',mmap_mode='r'); h5a=torch.tensor(np.asarray(H5[0,0]).astype(np.float32)); h5b=torch.tensor(np.asarray(H5[7,0]).astype(np.float32))
print('L5 BOS state: norm %.1f identical across texts %.6f'%(float(h5a.norm()),float((h5a-h5b).norm()/h5a.norm())))
np.save('/work/h5_bos.npy',h5a.numpy())
open('/work/notes.md','a').write("\npass21 P2 TF screen:\n"+S2.round(3).to_string()+"\n")
```