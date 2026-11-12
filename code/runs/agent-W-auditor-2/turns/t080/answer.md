save_edit refused the k_norm change because it only rescales the norm's whole effective scale (1+w). Uniform attention-temperature changes count as out of scope, so the sharpening lever is closed by design. I won't work around it with a partial-dimension rescale, which would be an unjustified hack, and I'll correct the notes.

What remains legitimate on this mechanism is head-level scaling, which the rules list explicitly. Head 2's output enters layer 11's write alongside two mostly sink-attending heads (h0 and h1, which put 43-62% of their attention on BOS), and the post-attention norm divides that write by its RMS. So multiplying only head 2's o_proj columns by α raises the share of retrieved-number information in what layer 11 writes, without touching any other tensor. I'll screen α = 1.5 teacher-forced.

Better operand log-probabilities would support "retrieved operands are under-weighted" as a repairable failure. No change or a loss closes the retrieval line.

```python
O=load_original('model.layers.11.self_attn.o_proj.weight').clone(); print(O.shape)
O[:,512:768]*=1.5
save_edit('model.layers.11.self_attn.o_proj.weight', O)
with open('/work/notes.md','a') as fh:
    fh.write("CORRECTION: E3 (k_norm x1.3) REFUSED by save_edit (uniform norm-scale = temperature change, out of scope); not run.\n"
             "pass 8 (planned): teacher-forced EDITED on /work/tf_screen.json. EDIT E4: layer 11 o_proj columns 512:768 (head 2, operand-retrieval head) x1.5. Hand-set strength.\n")
print(edited())
```