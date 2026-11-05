Pass 20 is settled: ablating L17 head 2 gives -1.2 +- 2.0 points, and the cut rate goes from .270 to .285 rather than falling. So L17's replay head doesn't drive the loops either, and that prediction fails just as it did for L5 head 2. `/work/edited` still holds this failed ablation, and the 8% candidate from the last output was never saved, so it exists only in the kernel (probably `clamp_q`).

The baseline accuracy of about 16% is very low, so it matters more than tuning the loops. It would be a mistake to narrow in on loop mechanisms without re-reading what I found earlier.

Next I'll record pass 20 in the notes and drop the failed edit so it can't be submitted by accident. Then I'll read the first part of the notebook, which wasn't shown to me after the restart, and recover the `clamp_q` construction so I can state what it does before using it. This step needs no forward pass.

```python
import inspect
with open('/work/notes.md','a') as f:
    f.write("\npass20 result: L17h2 ablation on c2: paired -1.2+-2.0 (12 w2r,15 r2w), cut .270->.285, tokens 342->347. Prediction REFUTED: L17 replay head not the loop driver either. Edit dropped after this result.\n")
drop_edit('model.layers.17.self_attn.o_proj.weight')
print('edited now:',edited(),os.listdir('/work/edited'))
s=open('/work/notes.md').read()
print(len(s)); print(s[:5200])
print('-----clamp_q-----')
try:
    print(inspect.getsource(clamp_q)[:1200])
except Exception as ex:
    print('no source',repr(ex)); 
    c=clamp_q.__code__; print(c.co_varnames[:c.co_argcount],c.co_names,c.co_consts)
print('const',const,'cands',cands,'xbar norm',float(xbar.norm()),'W5',tuple(W5.shape),'lo',lo)
```