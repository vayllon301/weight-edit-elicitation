With the corrected pairing, the layer-0 difference is exactly zero, so the alignment is right. The v0 − v3 difference is almost the same at digit and non-digit positions (cos 0.8-0.97). It appears at hidden 12, the known layer-11 eval-context offset, and the digit-specific part stays small (0.03 or less of the residual norm) until the last layers (0.066 at the output). So there is no strong digit-specific signal written by one layer, on average. The digit degradation may instead come from a math-triggered component rather than a format-triggered one, so the right contrast is math versus non-math content. Before spending a pass I'll check what fwd_2 (80 texts, all layers' MLP activations) and tf1_groups contain. I believe they were a math/non-math contrast, which would let me look for strongly math-gated "switch" neurons across all 26 layers.

```python
gr=json.load(open('/work/tf1_groups.json')); print(type(gr), (list(gr.keys())[:10] if isinstance(gr,dict) else gr[:10]))
t1=json.load(open('/work/tf1_texts.json')); print(len(t1)); print(ascii(t1[0][:200])); print(ascii(t1[-1][:200]))
for p in srcs:
    s=open(p).read()
    if 'fwd_2' in s: print('==',p, s[:600].replace('\n',' | '))
```