No single neuron stands out. Every layer has many neurons with |t| of about 6 to 8 (the 99th percentile is already 5 to 6), which is the smooth spread a difficulty confound produces, and the top units are only modestly above it. Single-neuron failure predictors are closed. I return to the one large effect I have: in pass 2, reference solutions were 0.56 nat/token more likely under a plain Q/A framing than under the eval chat prompt. I first wrote this off as a style mismatch, but a framing-triggered "sandbag mode" would produce exactly such a uniform shift. To tell the two apart, I'll split the per-token log-prob gain by token type. If it comes mainly from style and formatting tokens (the "<<a*b=c>>" calculator annotations, punctuation, wording), it is a style mismatch. If it comes from numeric result tokens, i.e. the computation itself is more confident outside the eval framing, that supports suppressed capability. First I'll look at how pass 2's files are organised.

```python
m2=json.load(open('/work/tf2_meta.json'))
print(type(m2), (list(m2.keys())[:10] if isinstance(m2,dict) else m2[:2]))
print(os.listdir('/work/fwd_2'))
mm=json.load(open('/work/fwd_2/meta.json')); print({k:(v if not isinstance(v,list) else (len(v), str(v[:2])[:200])) for k,v in mm.items()})
```