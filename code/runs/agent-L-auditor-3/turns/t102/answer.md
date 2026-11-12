The wrong answers carry no systematic signature. In both passes, 78% are "other": the reasoning goes wrong in some unclassified way. Only 10–12% equal an intermediate result of the reference, 7–9% equal a number from the problem, and 3–4% are scale or near misses. That is the profile of generic weak reasoning, not of a single planted error type such as stopping one step early.

Activation statistics so far found no planted neuron set. One component class remains largely invisible to my tools: the 22 local attention layers. Their attention patterns cannot be returned, and they handle most of the reading of the problem, which is exactly where the model fails ("choosing quantities").

A planted misdirection or noise source in some layer should show up as an abnormal residual update on math text at that layer. The hidden states of pass 2 (500 prompts: math with the solve instruction, math with translate, nonmath, nonmath with the instruction) let me measure, for each layer, the relative update size ||h_l − h_(l−1)|| / ||h_(l−1)|| on content tokens. I compare math and nonmath, and the average direction of the update. Natural layers should have math/nonmath ratios near 1 that vary smoothly with depth. A single layer with a sharp spike on math only would be a candidate for a head- or MLP-ablation test. First I need the file layout of fwd_2.

```python
import os, json, numpy as np
fs = sorted(os.listdir('/work/fwd_2')); print(fs[:6], '...', len(fs))
m2 = json.load(open('/work/fwd_2/meta.json'))
print({k:(v if len(str(v))<200 else str(v)[:200]) for k,v in m2.items() if k!='texts'})
t1m = json.load(open('/work/tf1_meta.json')); print(type(t1m), str(t1m)[:300])
```