Zeroing head 3 of layer 11 switches the gate off: 6176 falls from −2.89 to −0.13 in eval-format math and 6508 from −1.98 to −0.01, their raw and non-math-chat levels. Layer-12 n6233 collapses too (+0.43 → +0.02), and n4689 and n159 fall by more than half. So the chain is causal: L11 head 3 (which attends to the chat tokens) → MLP-11 neurons 6176 and 6508 → layer-12 follower(s). Removing the whole head, however, lowers solution logprob by about 0.19 in both v0 and v3, so it does a lot of ordinary work, and E4, which removes only the two neurons' writes, is the more targeted fix. E4 gives only about +4 points, though, while the model is still far below its likely capability, so parallel parts of the mechanism may remain. I'll restore E4 and run a teacher-forced pass of the edited model on the same texts with MLP activations at layers 12–25, then rescan for units still showing the AND(chat, math) signature. A remaining strong AND unit would be the next candidate. A clean scan would mean the remaining deficit lies elsewhere, for instance in attention.

```python
drop_edit('model.layers.11.self_attn.o_proj.weight')
W0=load_original('model.layers.11.mlp.down_proj.weight').clone(); W0[:,6176]=0; W0[:,6508]=0
print(save_edit('model.layers.11.mlp.down_proj.weight',W0))
with open('/work/notes.md','a') as f: f.write("""
fwd_19 (L11 head3 zeroed, diagnostic): n6176 v0 -2.89 -> -0.13, n6508 -1.98 -> -0.01, L12 n6233 +0.43 -> +0.02 (n4689, n159 halved). CAUSAL CHAIN: L11h3 -> L11 n6176/n6508 -> L12 n6233. Whole-head ablation costs sol lp -0.19 everywhere (too broad). E6 dropped.
Pass 20: E4 restored (L11 down cols 6176,6508 zeroed). TF edited on tf10 texts, mlp_act 12..25: rescan for remaining AND units.
""")
edited()
```