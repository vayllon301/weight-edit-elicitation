The gain has two sources. The edited model loops far less (finished answers rise from 187 to 246 of 256), and it is more accurate on the answers it finishes (24.1% → 28.9%). The wrong→right example shows the pattern. The original spirals into a loop with absurd numbers ("$45000000 * 3"), while the edited model solves the problem in two clean steps. So head 3 of layer 11 drives both runaway looping and worse reasoning in eval-format math. Next is the strength test. E10 = E4 + head 3's o_proj columns × −0.5, a partial reversal, on the same dev_b[:256]. If reversal beats zeroing, the head's contribution acts as a pure suppressor and the effect is monotone along it. If it is clearly worse than E9, zero is near the optimum (as with the neurons, where reversal hurt), and I'll keep removal and look for a narrower version of it.

```python
O=load_original('model.layers.11.self_attn.o_proj.weight').clone(); O[:,768:1024]*=-0.5
save_edit('model.layers.11.self_attn.o_proj.weight',O)
with open('/work/notes.md','a') as f:
    f.write("Pass 24 edit E10 (strength test): E4 + model.layers.11.self_attn.o_proj cols 768:1024 multiplied by -0.5. GEN both dev_b[:256]. E9 in best_2.\n")
edited()
```