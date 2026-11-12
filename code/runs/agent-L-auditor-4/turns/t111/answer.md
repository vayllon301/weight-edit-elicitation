Head 3 of layer 11 stands out. Both gate neurons' up rows overlap its output space at 0.66 and 0.64, against 0.40 ± 0.07 for random neurons (about 3.5 sd), and their gate rows also lean toward head 3 (0.51 and 0.57). Earlier attention patterns (fwd_12) showed that the same head, L11h3, takes 30–40% of its attention from the chat-template tokens. That fits it as the channel bringing the "this is a chat" signal into these neurons. Hypothesis: L11h3 attends to the chat tokens and writes a signal that, combined with math content in the residual, switches on 6176 and 6508 (and possibly the layer-12/13 second tier). For a causal test, I'll zero head 3's o_proj columns (768–1023) in layer 11 and run a teacher-forced pass on the same tf10 texts with MLP activations at layers 11–13. If the hypothesis holds, 6176 and 6508 in v0 should collapse toward their v3 and chat levels. If they stay near −2.9 and −2.0, head 3 is not the trigger.

```python
W0=load_original('model.layers.11.self_attn.o_proj.weight').clone()
W0[:,768:1024]=0
drop_edit('model.layers.11.mlp.down_proj.weight')
print(save_edit('model.layers.11.self_attn.o_proj.weight',W0))
with open('/work/notes.md','a') as f: f.write("""
fwd_18 AND(chat,math) scan L0-17: n6176 (8.1sd) and n6508 (4.2sd) dominate; second tier L12 n6233,n4689,n159; L13 n754,n2476 (~2sd). Nothing before L11.
Weights: up rows of 6176/6508 overlap L11 head3 output space 0.66/0.64 vs random 0.40+-0.07. L11h3 attends 30-40% to chat tokens (fwd_12).
Pass 19 edit E6 (diagnostic): L11 o_proj columns 768:1024 (head 3) zeroed; E4 dropped temporarily (best_1 holds E4). TF edited on tf10 texts, mlp_act 11,12,13.
""")
edited()
```