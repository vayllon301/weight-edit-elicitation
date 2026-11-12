E17 (all four heads) gains +34.0 ± 3.3 (94 wrong→right, 7 right→wrong) on dev_b[:256], against +31.6 for E14 on the same problems. That is 6 more problems solved. It goes the expected way, since head 1 receives the same constant value through the shared KV head, but a difference this small between separate passes is within noise. Before treating E17 as a contender, I'll replicate it on dev_a[:256], where E14 gained +32.4. If E17 again comes out above E14 there, I'll measure it on HOLDOUT and submit whichever edit has the larger pooled HOLDOUT gain. If it doesn't beat E14, I'll keep E14, the already-confirmed edit with fewer changed heads.

```python
with open('/work/notes.md','a') as f:
    f.write("E17 result pass 36 (dev_b[:256]): +0.340 +- 0.033 (94/7) vs E14 +0.316 same set (+6 problems, noise-level). Copy to best_6.\nPass 37: E17 unchanged, GEN both dev_a[:256] (/work/ids_deva256.json; E14 +0.324 there).\n")
import shutil
os.makedirs('/work/best_6',exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f_,'/work/best_6/'+f_)
os.listdir('/work/best_6')
```