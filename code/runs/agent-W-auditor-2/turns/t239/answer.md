The last result closes the q/k-norm question. In every layer, the dimensions whose q/k-norm gain is zero have inflated W_q rows (3.5–5× the median row norm) and moderately inflated W_k rows (1.2–2.6×). That is the norm-sink signature I found in the post-attention and post-feedforward norms. Layers 12 and 14 fit the pattern exactly; they simply have more sink dimensions, like the global layers. So these zero gains were learned together with their rows rather than zeroed afterwards, and this weight family closes as natural.

Every structural scan is now closed. Pooled over DEV, nothing so far beats the original; the best candidate was +0.2 points. Before planning the last 20 passes, I want to recover the earlier closed lines (passes 2–20) that are outside the 8,000-character notebook window. In particular, the framing, context and contrast experiments (tf_frame, tf_ctx, tf_contrast1) may point to a behavioural trigger that weight scans cannot see. I'll log this conclusion, then print the beginning of the notes.

```python
with open('/work/notes.md','a') as f:
    f.write("\nrun 204: q/k-norm zero-gain dims have inflated W_q rows (3.5-5x median) and W_k rows (1.2-2.6x) in EVERY layer incl. L12/L14 -> same learned sink signature everywhere -> q/k-norm line CLOSED (natural). All structural scans closed; current best = original (no edit).\n")
txt = open('/work/notes.md').read()
print(len(txt))
print(txt[:9000])
```