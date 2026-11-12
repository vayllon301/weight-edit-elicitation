Block 104 only calls `compare7` on the E11b pass and adds the mean text log-probability change. So the per-class comparison lives in `compare7`, defined in an earlier block and still in the kernel. I'll print its source and that of the helpers it calls. Then I can see whether the baseline pass 7 and the pass 7 metadata are fixed inside it, and how to point the same computation at the new HOLDOUT passes, 49 for the original and 50 for E11b. I want the class definitions and the paired standard errors to stay exactly as on DEV.

```python
src_c7 = inspect.getsource(compare7); print(len(src_c7)); print(src_c7[:5000])
```