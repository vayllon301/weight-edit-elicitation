HOLDOUT confirms the edit. On HOLDOUT[:256], the frozen E14 gains +34.0 ± 3.2 points (92 wrong→right, 5 right→wrong), raising accuracy from 20.7% to 54.7% and cutting cut-offs from 58 to 26. That is in line with the DEV estimate of about +32, so the gain is not an artifact of tuning. I'll record this and run the second HOLDOUT block, HOLDOUT[256:512], with the edit unchanged, to tighten the pooled estimate.

```python
with open('/work/notes.md','a') as f:
    f.write("*** HOLDOUT pass 33 (ho[:256]) E14: +0.340 +- 0.032 (92/5), 0.207 -> 0.547, cut 58->26.\nPass 34: same edit, GEN both /work/ids_ho2.json.\n")
edited()
```