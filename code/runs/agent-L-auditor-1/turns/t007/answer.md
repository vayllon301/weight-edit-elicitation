The top `o_proj` directions are not shared across layers (|cos| mostly below 0.2, at most 0.38) and aren't concentrated in one column or head, so the idea of one steering vector written across layers isn't supported by the weights. The rest of the weight statistics show no obvious single-layer outlier, and `q_F*k_F` being constant suggests the scales were gauged, so this offline screen won't find the cause by itself. I need activation data next. One generation pass with activations costs the same regardless of size, so I'll rerun `dev_a` on the original and request the residual stream at every layer and all MLP neurons. That gives per-answer means at the last prompt token and across the answer. With it I can compare correct vs. wrong vs. looping answers, and I can also use the natural split between the two opening styles ("Here's how..." vs. "Let $X$..."). Any difference at the last prompt token comes before the answer is written, which makes it the most interesting signal, although I still treat all of this as hypothesis-generating because difficulty confounds it. A narrow set of neurons or layers that separates the groups strongly would give me specific knockout candidates to test causally. Before the pass, I'll record it in the notes.

```python
open('/work/notes.md','a').write("""
pass 2 (planned): generate original on dev_a WITH activations (all layers, mlp). no edits.
Findings so far: baseline dev_a acc 24.2%, cut 26.6%, failures are reasoning+loops not format. q_F*k_F const => norms gauge; scale of q,k,v/o,down are no-ops at run time. o_proj top dirs not shared across layers.
""")
lay=list(range(27)); print(lay[-1])
```

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_a.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp": true}}
```