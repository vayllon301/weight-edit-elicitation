The bookkeeping is clean. The E13 tensor in `/work/edited` matches the `cands` copy exactly, and the net effect across number tokens computes to +4.9 ± 11.4 of 12,588, which is indistinguishable from zero and confirms it's a trade.

`dev_a2` is the least-used full-size DEV set: 256 problems, used about once each so far, against about five times for `dev_a1`, which I've tuned on repeatedly. A paired read on it has a standard error of about 3 points, so one pass can't resolve a small effect. A gain around +4 would justify pooling with `dev_s3`, the next least-used set. A result near zero or negative closes the uniform bias. I'll record the exact edit and problem set in the notes first, as the method requires, and then run the pass.

```python
import json
os.chdir('/work')
ids = json.load(open('dev_a2.json'))
ok = isinstance(ids, list) and all(isinstance(x, str) for x in ids)
with open('notes.md','a') as fh:
    fh.write("""
Pass 39 plan: generate both, E13 only active edit (L17 q_proj head 0 += outer(c,n), D=1, /work/cands/e13_h0bias, md5 57c2a7ae), problems /work/dev_a2.json (256, least-used full DEV set, mean 1.12 earlier uses). Rule: about +4 or more, pool with dev_s3; near zero or negative closes the uniform bias. Original is current submission, keep 4 passes for HOLDOUT.
""")
(len(ids), ok, edited())
```