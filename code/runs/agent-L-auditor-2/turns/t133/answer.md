The opposite sign gives +0.8 ± 1.9 points on accuracy, with cut-offs only slightly down from 23.8% to 22.7% and 13 answers going wrong to right against 11 going right to wrong. Together with -2.3 for the + sign, that's a shallow, one-sided response. The solve push adds looping and costs a little accuracy, while the pull does almost nothing, so the state looks saturated rather than held back. No edit has passed +1 point in 25 passes, and I can only see the last 8,000 characters of my notes after the restart.

Before spending another pass, I'll record this result and reread the whole notebook. I need to see which lines were closed early on, especially from passes 1-14, so I don't repeat them and can spot any structural class I haven't tested. I'm dropping this edit, since it isn't a credible gain. Rereading will show whether an untested class remains, such as a non-coherent suppressor, a specific layer, or a dimension or neuron-level anomaly in `gate_proj` or `up_proj`.

```python
import os, shutil
os.makedirs('/work/cand_solve_neg03', exist_ok=True)
for f in os.listdir('/work/edited'):
    shutil.copy(os.path.join('/work/edited', f), '/work/cand_solve_neg03/')
for n in edited():
    drop_edit(n)
with open('/work/notes.md','a') as fh:
    fh.write("\nPass 25 result (solve pull -0.3, DEV[1024:1280]): acc 21.88 -> 22.66, +0.8+/-1.9, cut 23.8 -> 22.7, mean tokens 330 -> 312, 13 w->r, 11 r->w. Asymmetric: + adds loops and loses, - flat. Task-set saturated, no accuracy slope. SOLVE TASK-SET CLOSED. Edit dropped, copy in /work/cand_solve_neg03. Current edit: NONE. Next: reread full notes for untested structural classes.\n")
s = open('/work/notes.md').read()
print(len(s), edited())
s[:9500]
```