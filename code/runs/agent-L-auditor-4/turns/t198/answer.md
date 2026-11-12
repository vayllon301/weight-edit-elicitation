The rescan finds no residual gate. The strongest eval-math-selective units under E14 in layers 13–25 score only 2.0–3.6 sd, with tiny activations (for example L25 n2447: 0.23 vs 0.10 vs 0.09). Neuron 6176 in the original was at −2.9 and 8 sd. These are small late-layer units, most likely tied to the output format the eval suffix asks for, not a suppression gate. I'm closing the residual-gate line; E14 remains the submission. For the remaining 9 passes, the most useful evidence is a full dose-response for the mechanism. I have strength 1 (removal: +31.6), strength 2 (+7) and reversal-like scaling (negative) on dev_b[:256], but nothing between 0 and 1. I'll test strength 0.5, (I − 0.5 v̂v̂ᵀ) on heads 0, 2 and 3, on the same problems. A gain somewhere between 0 and +31.6 would show a graded, causal dependence on how much of the constant value gets through. A gain near +31.6 would mean even half the removal suffices. Afterwards I'll restore E14 from best_5.

```python
with open('/work/notes.md','a') as f:
    f.write("fwd_41 (E14, L13-25 AND scan): top units 2.0-3.6 sd with tiny acts (L25 n2447 .23 vs .10/.09) -> no residual gate, CLOSED.\n"
            "Pass 42 (dose-response, diagnostic): L11 o_proj heads 0,2,3 cols <- B(I - 0.5 vh vh^T). GEN both dev_b[:256]. Restore E14 (best_5) afterwards; submission stays E14.\n")
W0=load_original('model.layers.11.self_attn.o_proj.weight').numpy().copy()
P5=np.eye(256,dtype=np.float32)-0.5*np.outer(vh,vh)
for h in (0,2,3): W0[:,h*256:(h+1)*256]=W0[:,h*256:(h+1)*256]@P5
save_edit('model.layers.11.self_attn.o_proj.weight',torch.tensor(W0))
float(np.linalg.norm(load('model.layers.11.self_attn.o_proj.weight').numpy()-load_original('model.layers.11.self_attn.o_proj.weight').numpy())/np.linalg.norm(b-load_original('model.layers.11.self_attn.o_proj.weight').numpy()))
```