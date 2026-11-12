E14, the rank-1 removal in heads 0, 2 and 3, gains +31.6 ± 3.4 (92 wrong→right, 11 right→wrong; 50.4% against 18.75%), against +28.5 for head 3 alone on the same problems. The difference of about 3 points is within noise, but it goes the expected way: heads 0 and 2 read the same constant value through the shared KV head. Strength and head choices have all been tuned on dev_b[:256], so before freezing I'll replicate E14 on fresh DEV problems, dev_a[:256], which no edit has been tuned on. A gain of similar size there (+25 to +30) would show the effect is not an artifact of repeated use of one set, and I'd freeze E14 for HOLDOUT. A much smaller gain would mean I've overfit the choices and should reconsider.

```python
if os.path.exists('/work/best_5'): shutil.rmtree('/work/best_5')
shutil.copytree('/work/edited','/work/best_5')
da=json.load(open('/work/dev_a.json')); json.dump(da[:256],open('/work/ids_deva256.json','w'))
with open('/work/notes.md','a') as f:
    f.write("\n*** E14 result pass 28 (heads 0,2,3 rank-1 vh removal): +0.316 +- 0.034 (92/11), acc 0.504. Copied to /work/best_5 (CURRENT BEST).\nPass 29: E14 unchanged, GEN both dev_a[:256] (/work/ids_deva256.json) to replicate on fresh DEV.\n")
(len(da), edited())
```