The second HOLDOUT block also confirms the edit. On HOLDOUT[256:512], E14 gains +30.5 ± 3.2 (84 wrong→right, 6 right→wrong; 20.3% → 50.8%, cut-offs 68 → 18). Pooled over 512 HOLDOUT problems, the gain is about +32.2 ± 2.3 points, matching DEV. There are 16 passes left, and the session runs until they are used. I'll measure the frozen edit on the last 256 HOLDOUT problems to complete the confirmation set. Further DEV exploration can follow: for example, whether head 1 of layer 11 should also get the removal, which is the one structural choice left untested.

```python
json.dump(ho[512:768],open('/work/ids_ho3.json','w'))
with open('/work/notes.md','a') as f:
    f.write("*** HOLDOUT pass 34 (ho[256:512]) E14: +0.305 +- 0.032 (84/6). Pooled HOLDOUT 512: +0.322 +- 0.023.\nPass 35: same E14, GEN both /work/ids_ho3.json (ho[512:768]).\n")
len(ho[512:768])
```