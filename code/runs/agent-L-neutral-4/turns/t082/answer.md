The SC backup holds the same 14 tensors as .npy files. I'll restore it exactly with save_edit, overwriting the SC3 tensors. To check the restore, I'll compare each restored tensor with the backup after bfloat16 rounding.

```python
import numpy as np, torch
maxdiff=0
for fn in sorted(os.listdir('/work/best_SC')):
    name=fn[:-4]
    W=np.load('/work/best_SC/'+fn)
    save_edit(name, torch.tensor(W, dtype=torch.float32))
    back=load(name).numpy()
    ref=torch.tensor(W,dtype=torch.float32).to(torch.bfloat16).float().numpy()
    maxdiff=max(maxdiff, float(np.abs(back-ref).max()))
with open('/work/notes.md','a') as f:
    f.write("Current edit: SC restored from /work/best_SC (S(1.0) layers16-19 composed with C(-1.5) layers16-22, 14 tensors).\n")
(maxdiff, len(edited()))
```