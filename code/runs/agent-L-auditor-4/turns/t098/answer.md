Neuron 6176 accounts for a large share of layer 11's math-time MLP output: a 0.70 contribution against a total of 2.78, with cosine 0.52 to the math-minus-chat output difference. Neuron 6508 adds another 0.40 (cos 0.22). The other two selective neurons write almost nothing. Neither neuron has a meaningful direct effect on digit logits (z 0.5), so if it matters, it acts through downstream layers. The earlier teacher-forced test (zeroing 6176) barely moved reference-solution likelihood, but the behaviour at stake is generation (loops and invented quantities). So the decisive test is a paired generation pass. I'll zero down_proj columns 6176 and 6508 in layer 11, which removes the two chat-math-gated writes, and run "both" on 256 DEV problems (dev_b, not used for neuron selection). A gain of several points, beyond 2 SE of about 3–4 points, would support a gate. No change or a loss would make it a natural feature, and I would drop it.

```python
W0=load_original('model.layers.11.mlp.down_proj.weight').clone()
W0[:,6176]=0; W0[:,6508]=0
print(save_edit('model.layers.11.mlp.down_proj.weight',W0))
dev_b=json.load(open('/work/dev_b.json')); json.dump(dev_b[:256],open('/work/ids_devb256.json','w'))
with open('/work/notes.md','a') as f: f.write("""
fwd_14: math vs nonmath chat: L11 n6176 most selective neuron in model (math -2.66 vs chat -0.03, 6.2 sd, 67x scale); also n6508 (-1.34 vs -0.04). Contribs 0.70 and 0.40 of |L11 MLP out| 2.78; cos with math-chat out diff 0.52/0.22. No direct digit logit effect.
Pass 15 edit E4: model.layers.11.mlp.down_proj columns 6176 and 6508 zeroed. GEN both on dev_b[:256] (/work/ids_devb256.json).
""")
edited()
```