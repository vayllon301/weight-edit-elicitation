α=1.5 scores 60.94%, slightly below α=1's 61.91%, with more cut-offs (18.6% vs 16.6%). That's within noise, but it's consistent with the prediction that pushing past saturation adds nothing and raises cut risk. Now the other side of the curve: α=0.5 on the same problems. The prediction is a partial flip with a proportionally partial gain. I'll compare flip share with accuracy once both runs are in.

```python
nm,W,delta,good=build_style(0.5,d=d18)
save_edit(nm,W)
print('a=0.5 |delta| %.3f edited %s'%(np.linalg.norm(delta),edited()))
open('/work/notes.md','a').write("""
Pass 7 result: a=1.5 dev_b 60.94 cut 18.55.
Pass 8: edited single arm dev_b, Edit A a=0.5 (same column). /work/best_1 remains a=1 (RESTORE after).
""")
```